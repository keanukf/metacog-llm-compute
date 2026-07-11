"""Tests for ServerBackend logprob normalization and thinking verification."""

from __future__ import annotations

from src.execution.backend.server import response_has_thinking_block, verify_enable_thinking
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
