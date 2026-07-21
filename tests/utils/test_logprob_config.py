"""Tests for resolve_top_logprobs config helper."""

from __future__ import annotations

from src.utils.inference.logprob_config import DEFAULT_TOP_LOGPROBS, resolve_top_logprobs


def test_resolve_default():
    assert resolve_top_logprobs(None) == DEFAULT_TOP_LOGPROBS
    assert DEFAULT_TOP_LOGPROBS == 20


def test_resolve_from_inference_cfg():
    assert resolve_top_logprobs({"top_logprobs": 12}) == 12


def test_resolve_deprecated_lmstudio_alias():
    assert resolve_top_logprobs({"lmstudio_top_logprobs": 7}) == 7


def test_resolve_precedence_top_over_alias():
    assert resolve_top_logprobs({"top_logprobs": 15, "lmstudio_top_logprobs": 7}) == 15


def test_resolve_kwarg_over_cfg():
    assert resolve_top_logprobs({"top_logprobs": 3}, top_logprobs=9) == 9


def test_resolve_clamps_to_at_least_one():
    assert resolve_top_logprobs({"top_logprobs": 0}) == 1
