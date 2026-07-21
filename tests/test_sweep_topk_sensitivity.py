"""Tests for top-K sensitivity sidecar parsing and H1a alignment."""

from __future__ import annotations

from scripts.instrument_validation.sweep_topk_sensitivity import (
    _pick_logprob_tokens,
    _sidecar_step_entry,
    _step_logprob_token_lists,
    _tle_score_at_k,
)
from src.signals.token_entropy import (
    mean_entropy_at_top_k,
    slice_action_logprob_tokens,
    tle_mean_entropy_at_k_from_logprob_tokens,
)

_TOP = [{"token": "go", "logprob": 0.0}, {"token": "look", "logprob": -1.0}]


def test_action_top_logprobs_legacy_list_sidecar() -> None:
    payload = [{"token": "go", "logprob": 0.0, "top_logprobs": _TOP}]
    entry = _sidecar_step_entry(payload, step_index=0)
    assert entry is not None
    assert _step_logprob_token_lists(entry) == [payload]


def test_action_top_logprobs_v1_dict_sidecar() -> None:
    payload = {
        "schema_version": 1,
        "steps": [
            {
                "step_index": 2,
                "logprob_tokens": [{"token": "go", "logprob": 0.0, "top_logprobs": _TOP}],
            }
        ],
    }
    entry = _sidecar_step_entry(payload, step_index=2)
    assert entry is not None
    toks = _step_logprob_token_lists(entry)[0]
    assert toks[0]["top_logprobs"] == _TOP


def test_action_top_logprobs_v2_multi_sample_sidecar() -> None:
    payload = {
        "schema_version": 2,
        "steps": [
            {
                "step_index": 1,
                "samples": [
                    {
                        "sample_index": 0,
                        "logprob_tokens": [
                            {"token": "north", "logprob": 0.0, "top_logprobs": _TOP}
                        ],
                    }
                ],
            }
        ],
    }
    entry = _sidecar_step_entry(payload, step_index=1)
    assert entry is not None
    assert len(_step_logprob_token_lists(entry)) == 1


def test_slice_action_skips_thinking_prefix() -> None:
    toks = [
        {"token": "<think>", "logprob": 0.0, "top_logprobs": _TOP},
        {"token": "reason", "logprob": 0.0, "top_logprobs": _TOP},
        {"token": "</think>", "logprob": 0.0, "top_logprobs": _TOP},
        {"token": "go", "logprob": 0.0, "top_logprobs": _TOP},
        {"token": " north", "logprob": 0.0, "top_logprobs": _TOP},
        {"token": "\n", "logprob": 0.0, "top_logprobs": _TOP},
    ]
    sliced = slice_action_logprob_tokens(toks)
    assert [t["token"] for t in sliced] == ["go", " north", "\n"]


def test_tle_score_at_k_matches_episode_mean_at_k20() -> None:
    top = [
        {"token": "A", "logprob": -0.1},
        {"token": "B", "logprob": -0.5},
        {"token": "C", "logprob": -2.0},
    ]
    toks = [{"token": "A->C", "logprob": 0.0, "top_logprobs": top}]
    ent = tle_mean_entropy_at_k_from_logprob_tokens(toks, k=20)
    assert ent is not None
    payload = {"steps": [{"step_index": 0, "logprob_tokens": toks}]}
    score = _tle_score_at_k(payload, 0, 20, reference_mean_entropy=ent)
    assert score == ent


def test_pick_logprob_tokens_prefers_reference_match() -> None:
    high_action_top = [
        {"token": "a", "logprob": -0.01},
        {"token": "b", "logprob": -0.01},
        {"token": "c", "logprob": -0.01},
    ]
    low_action_top = [
        {"token": "A", "logprob": -0.001},
        {"token": "B", "logprob": -8.0},
    ]
    high = [{"token": "A->C", "logprob": 0.0, "top_logprobs": high_action_top}]
    low = [{"token": "A->B", "logprob": 0.0, "top_logprobs": low_action_top}]
    ref = tle_mean_entropy_at_k_from_logprob_tokens(high, k=20)
    picked = _pick_logprob_tokens([low, high], reference_mean_entropy=ref, k=20)
    assert picked == high


def test_mean_entropy_at_top_k_requires_two_candidates() -> None:
    one = [{"token": "x", "logprob": 0.0, "top_logprobs": [{"token": "x", "logprob": 0.0}]}]
    assert mean_entropy_at_top_k(one, k=20) is None
