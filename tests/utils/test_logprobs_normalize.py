"""Normalization of vLLM logprob payloads into the common per-step schema.

Verifies token-id maps per step, skipping of null positions, multi-candidate top-logprob handling,
and the case where the sampled token is not the highest-logprob one. This normalized shape is what
token-level entropy is computed from, so a normalization bug would corrupt the RQ1 signal at the
source while leaving the downstream math looking correct.
"""

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


def test_normalize_vllm_multi_candidate_top_logprobs():
    raw = [
        {
            42: _FakeVllmLogprob(-0.1, "go"),
            7: _FakeVllmLogprob(-2.5, " stop"),
            99: _FakeVllmLogprob(-1.8, " north"),
        }
    ]
    out = normalize_logprobs(raw, sampled_token_ids=[42])
    assert out is not None
    assert len(out) == 1
    rec = out[0]
    assert rec["token"] == "go"
    assert rec["logprob"] == -0.1
    assert len(rec["top_logprobs"]) == 3
    assert rec["top_logprobs"][0]["logprob"] == -0.1


def test_normalize_vllm_sampled_token_not_highest_logprob():
    raw = [{1: _FakeVllmLogprob(-0.05, "a"), 2: _FakeVllmLogprob(-3.0, "b")}]
    out = normalize_logprobs(raw, sampled_token_ids=[2])
    assert out is not None
    assert out[0]["token"] == "b"
    assert out[0]["logprob"] == -3.0
