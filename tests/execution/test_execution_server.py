"""ServerBackend: logprob normalization, thinking verification, and batched generation.

Verifies chat-completion logprobs normalize to the common structure, responses carry a thinking
block, ``enable_thinking`` is verified against a mock, concurrent posts run in parallel, and
``generate_many`` issues a single n-request but falls back to sequential on backend error. The
thinking-enabled verification is a signal-validity precondition: C1/C2 reasoning and their TLE/VC
signals only mean what RQ1 assumes if the server actually ran with thinking on, so an unverified
backend could silently produce no-think outputs.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.execution.backend.server import (
    ServerBackend,
    response_has_thinking_block,
    verify_enable_thinking,
)
from src.utils.inference.vllm_shared import normalize_chat_completion_logprobs


def test_normalize_chat_completion_logprobs_structure():
    raw = {
        "content": [
            {
                "token": "go",
                "logprob": 0.0,
                "top_logprobs": [
                    {"token": "go", "logprob": 0.0},
                    {"token": "look", "logprob": -1.0},
                ],
            }
        ]
    }
    out = normalize_chat_completion_logprobs(raw)
    assert out is not None
    assert out[0]["token"] == "go"
    assert len(out[0]["top_logprobs"]) == 2


def test_response_has_thinking_block():
    assert response_has_thinking_block("<think>plan</think>\nnorth")
    assert not response_has_thinking_block("north only")


def test_verify_enable_thinking_mock():
    class _Backend:
        def generate(self, *a, **k):
            return "<think>plan</think>\nnorth", None

    ok, _ = verify_enable_thinking(_Backend())
    assert ok is True


def test_concurrent_generate_allows_parallel_posts():
    in_flight = 0
    max_in_flight = 0
    counter_lock = threading.Lock()

    backend = ServerBackend(
        server_url="http://127.0.0.1:8000/v1",
        model_name="test-model",
    )

    def _slow_post(payload: dict) -> dict:
        nonlocal in_flight, max_in_flight
        with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with counter_lock:
            in_flight -= 1
        return {"choices": [{"index": 0, "message": {"content": "go north"}, "logprobs": None}]}

    backend._post_chat = _slow_post  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: backend.generate("obs"), range(4)))
    assert max_in_flight > 1


def test_generate_many_uses_single_n_request():
    payloads: list[dict] = []

    backend = ServerBackend(
        server_url="http://127.0.0.1:8000/v1",
        model_name="test-model",
    )

    def _capture_post(payload: dict) -> dict:
        payloads.append(payload)
        n = int(payload.get("n", 1))
        choices = []
        for i in range(n):
            choices.append(
                {
                    "index": i,
                    "message": {"content": f"go north {i}"},
                    "logprobs": {"content": [{"token": "go", "logprob": -0.1, "top_logprobs": []}]},
                }
            )
        return {"choices": choices}

    backend._post_chat = _capture_post  # type: ignore[method-assign]
    outs = backend.generate_many("obs", n=3, logprobs=True, temperature=0.7)
    assert len(outs) == 3
    assert len(payloads) == 1
    assert payloads[0]["n"] == 3
    assert outs[0][0] == "go north 0"
    assert outs[2][0] == "go north 2"
    assert outs[0][1] is not None


def test_generate_many_falls_back_to_sequential_on_backend_error():
    calls = {"n": 0}

    backend = ServerBackend(
        server_url="http://127.0.0.1:8000/v1",
        model_name="test-model",
    )

    def _fail_batched_then_ok(payload: dict) -> dict:
        if payload.get("n", 1) > 1:
            calls["n"] += 1
            from src.utils.errors import BackendError

            raise BackendError("batched n unsupported")
        return {"choices": [{"index": 0, "message": {"content": "go east"}, "logprobs": None}]}

    backend._post_chat = _fail_batched_then_ok  # type: ignore[method-assign]
    outs = backend.generate_many("obs", n=2, logprobs=False)
    assert len(outs) == 2
    assert calls["n"] == 1
    assert outs[0][0] == "go east"
    assert outs[1][0] == "go east"
