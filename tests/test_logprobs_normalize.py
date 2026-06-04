"""Tests for vLLM-style logprob normalization."""

from __future__ import annotations

from src.utils.inference.logprobs import logprob_token_coverage, normalize_logprobs


class _FakeVllmLogprob:
    def __init__(self, logprob: float, decoded_token: str) -> None:
        self.logprob = logprob
        self.decoded_token = decoded_token


def test_normalize_vllm_token_id_map_per_step():
    raw = [{42: _FakeVllmLogprob(-0.5, "go")}, {7: _FakeVllmLogprob(-0.2, " north")}]
    out = normalize_logprobs(raw)
    assert out is not None
    assert len(out) == 2
    assert out[0]["logprob"] == -0.5
    assert out[0]["token"] == "go"


def test_normalize_skips_none_positions():
    raw = [None, {"logprob": -0.1, "token": "x"}]
    out = normalize_logprobs(raw)
    assert out is not None
    assert len(out) == 1


def test_logprob_token_coverage():
    recs = [{"logprob": -0.1, "token": "a"}, {"logprob": -0.2}]
    cov = logprob_token_coverage(recs)
    assert cov["n_tokens"] == 2
    assert cov["n_with_token"] == 1
    assert cov["token_field_rate"] == 0.5


def test_normalize_logprob_object_directly():
    raw = [_FakeVllmLogprob(-1.0, "hi")]
    out = normalize_logprobs(raw)
    assert out == [{"logprob": -1.0, "token": "hi"}]
