import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch.cuda.memory as cuda_memory

from sglang.test.ci.ci_register import register_cpu_ci

for name in (
    "_cuda_beginAllocateCurrentThreadToPool",
    "_cuda_endAllocateToPool",
    "_cuda_releasePool",
):
    if not hasattr(cuda_memory, name):
        setattr(cuda_memory, name, lambda *_, **__: None)

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _TraceCtx:
    def __init__(self):
        self.abort_info = None

    def abort(self, abort_info):
        self.abort_info = abort_info


def _req(rid: str, priority=None):
    return SimpleNamespace(
        rid=rid,
        priority=0 if priority is None else priority,
        time_stats=SimpleNamespace(trace_ctx=_TraceCtx(), wait_queue_entry_time=0.0),
    )


def _scheduler(
    *,
    waiting=0,
    running=0,
    current=0,
    previous=0,
    grammar=0,
    chunked=False,
    max_running_requests=4,
    max_queued_requests=1,
    limit_admitted_requests_to_running_capacity=True,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting_queue = [_req(f"waiting-{i}") for i in range(waiting)]
    scheduler.running_batch = SimpleNamespace(
        reqs=[_req(f"running-{i}") for i in range(running)]
    )
    scheduler.cur_batch = SimpleNamespace(
        reqs=[_req(f"current-{i}") for i in range(current)]
    )
    scheduler.last_batch = SimpleNamespace(
        reqs=[_req(f"previous-{i}") for i in range(previous)]
    )
    scheduler.chunked_req = _req("chunked") if chunked else None
    scheduler.grammar_manager = SimpleNamespace(
        grammar_queue=[_req(f"grammar-{i}") for i in range(grammar)]
    )
    scheduler.max_running_requests = max_running_requests
    scheduler.max_queued_requests = max_queued_requests
    scheduler.limit_admitted_requests_to_running_capacity = (
        limit_admitted_requests_to_running_capacity
    )
    scheduler.enable_priority_scheduling = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    scheduler.send_to_tokenizer = MagicMock()
    return scheduler


class TestSchedulerAdmissionLimit(unittest.TestCase):
    def test_active_admitted_count_includes_scheduler_live_state(self):
        scheduler = _scheduler(
            waiting=2,
            running=3,
            current=1,
            previous=1,
            grammar=1,
            chunked=True,
        )

        self.assertEqual(Scheduler._active_admitted_request_count(scheduler), 9)

    def test_active_admitted_count_deduplicates_by_rid(self):
        scheduler = _scheduler(waiting=0, running=0, grammar=0)
        shared = _req("same-rid")
        scheduler.waiting_queue = [shared]
        scheduler.running_batch.reqs = [shared, _req("same-rid")]
        scheduler.cur_batch.reqs = [_req("same-rid")]
        scheduler.last_batch.reqs = [_req("same-rid")]
        scheduler.chunked_req = _req("same-rid")
        scheduler.grammar_manager.grammar_queue = [_req("other-rid")]

        self.assertEqual(Scheduler._active_admitted_request_count(scheduler), 2)

    def test_capacity_limit_replaces_static_queue_limit_when_enabled(self):
        scheduler = _scheduler(
            waiting=3,
            running=0,
            grammar=0,
            max_running_requests=4,
            max_queued_requests=1,
            limit_admitted_requests_to_running_capacity=True,
        )

        self.assertFalse(Scheduler._abort_on_queued_limit(scheduler, _req("incoming")))
        scheduler.send_to_tokenizer.send_output.assert_not_called()

    def test_capacity_limit_rejects_when_selected_concurrency_is_full(self):
        scheduler = _scheduler(
            waiting=1,
            running=2,
            grammar=1,
            max_running_requests=4,
            max_queued_requests=16,
            limit_admitted_requests_to_running_capacity=True,
        )
        incoming = _req("incoming")

        self.assertTrue(Scheduler._abort_on_queued_limit(scheduler, incoming))

        out, req = scheduler.send_to_tokenizer.send_output.call_args.args
        self.assertIs(req, incoming)
        self.assertEqual(out.finished_reason["status_code"], 429)
        self.assertEqual(out.finished_reason["message"], "The request queue is full.")
        self.assertEqual(
            incoming.time_stats.trace_ctx.abort_info["reason"],
            "The request queue is full.",
        )

    def test_static_queue_limit_still_applies_when_capacity_mode_disabled(self):
        scheduler = _scheduler(
            waiting=1,
            running=0,
            grammar=0,
            max_running_requests=128,
            max_queued_requests=1,
            limit_admitted_requests_to_running_capacity=False,
        )

        self.assertTrue(Scheduler._abort_on_queued_limit(scheduler, _req("incoming")))

    def test_priority_rejects_incoming_queue_full_as_429(self):
        scheduler = _scheduler(
            waiting=1,
            running=0,
            max_running_requests=128,
            max_queued_requests=1,
            limit_admitted_requests_to_running_capacity=False,
        )
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = False
        scheduler.waiting_queue = [_req("waiting", priority=10)]
        incoming = _req("incoming", priority=1)

        self.assertTrue(Scheduler._abort_on_queued_limit(scheduler, incoming))

        out, req = scheduler.send_to_tokenizer.send_output.call_args.args
        self.assertIs(req, incoming)
        self.assertEqual(out.finished_reason["status_code"], 429)
        self.assertEqual(out.finished_reason["message"], "The request queue is full.")

    def test_priority_preempts_existing_request_as_503(self):
        scheduler = _scheduler(
            waiting=1,
            running=0,
            max_running_requests=128,
            max_queued_requests=1,
            limit_admitted_requests_to_running_capacity=False,
        )
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = False
        waiting_req = _req("waiting", priority=1)
        scheduler.waiting_queue = [waiting_req]
        incoming = _req("incoming", priority=10)

        self.assertFalse(Scheduler._abort_on_queued_limit(scheduler, incoming))

        out, req = scheduler.send_to_tokenizer.send_output.call_args.args
        self.assertIs(req, waiting_req)
        self.assertEqual(out.finished_reason["status_code"], 503)
        self.assertEqual(
            out.finished_reason["message"],
            "The request is aborted by a higher priority request.",
        )

    def test_server_arg_enables_capacity_limit(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--limit-admitted-requests-to-running-capacity",
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.limit_admitted_requests_to_running_capacity)


if __name__ == "__main__":
    unittest.main()
