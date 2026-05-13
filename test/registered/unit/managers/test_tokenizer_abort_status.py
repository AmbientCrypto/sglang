import unittest
from http import HTTPStatus
from types import SimpleNamespace

import fastapi

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _coerce_http_status,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _manager_and_state():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(enable_lora=False)
    state = SimpleNamespace(obj=SimpleNamespace(rid="rid-1", lora_path=None))
    manager.rid_to_state = {state.obj.rid: state}
    return manager, state


def _abort_output(status_code):
    return {
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "status_code": status_code,
                "message": "The request queue is full.",
            }
        }
    }


class TestTokenizerAbortStatus(unittest.IsolatedAsyncioTestCase):
    def test_coerce_http_status_accepts_int_and_enum(self):
        self.assertEqual(_coerce_http_status(429), HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(
            _coerce_http_status(HTTPStatus.TOO_MANY_REQUESTS),
            HTTPStatus.TOO_MANY_REQUESTS,
        )
        self.assertIsNone(_coerce_http_status(999))

    async def test_non_stream_queue_full_abort_raises_429(self):
        manager, state = _manager_and_state()

        with self.assertRaises(fastapi.HTTPException) as raised:
            await TokenizerManager._handle_abort_finish_reason(
                manager,
                _abort_output(HTTPStatus.TOO_MANY_REQUESTS),
                state,
                is_stream=False,
            )

        self.assertEqual(raised.exception.status_code, 429)
        self.assertNotIn(state.obj.rid, manager.rid_to_state)

    async def test_non_stream_queue_full_abort_accepts_int_status(self):
        manager, state = _manager_and_state()

        with self.assertRaises(fastapi.HTTPException) as raised:
            await TokenizerManager._handle_abort_finish_reason(
                manager,
                _abort_output(429),
                state,
                is_stream=False,
            )

        self.assertEqual(raised.exception.status_code, 429)

    async def test_stream_queue_full_abort_is_yielded_for_sse_error(self):
        manager, state = _manager_and_state()
        out = _abort_output(429)

        returned = await TokenizerManager._handle_abort_finish_reason(
            manager,
            out,
            state,
            is_stream=True,
        )

        self.assertIs(returned, out)


if __name__ == "__main__":
    unittest.main()
