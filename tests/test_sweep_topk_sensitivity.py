"""Tests for top-K sensitivity sidecar parsing."""

from __future__ import annotations

from scripts.sweep_topk_sensitivity import _action_top_logprobs

_TOP = [{"token": "go", "logprob": 0.0}, {"token": "look", "logprob": -1.0}]


def test_action_top_logprobs_legacy_list_sidecar() -> None:
    payload = [{"token": "go", "logprob": 0.0, "top_logprobs": _TOP}]
    assert _action_top_logprobs(payload, step_index=0) == _TOP


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
    assert _action_top_logprobs(payload, step_index=2) == _TOP


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
    assert _action_top_logprobs(payload, step_index=1) == _TOP
