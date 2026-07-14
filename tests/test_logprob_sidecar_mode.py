"""Tests for logprob sidecar mode resolution and action-window filtering."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_utils import write_logprob_distribution_artifacts
from src.utils.logprob_sidecar import (
    LogprobSidecarConfig,
    filter_logprob_raw_for_sidecar,
    parse_full_instance_overrides,
    parse_logprob_sidecar_mode,
)

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def _c1_full_tokens() -> list[dict]:
    think = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
    ]
    action = [
        {"token": "go", "logprob": -0.5, "top_logprobs": [{"token": "x", "logprob": -0.6}]},
        {"token": " north\n", "logprob": -0.4, "top_logprobs": [{"token": "y", "logprob": -0.5}]},
    ]
    return think + action


def test_parse_mode_explicit_and_default_off():
    assert parse_logprob_sidecar_mode({"logprob_sidecar_mode": "action_window"}) == "action_window"
    assert parse_logprob_sidecar_mode({}) == "off"
    assert parse_logprob_sidecar_mode({"logprob_sidecar_mode": "off"}) == "off"


def test_parse_mode_rejects_legacy_save_logprob_distributions():
    import pytest

    with pytest.raises(ValueError, match="save_logprob_distributions"):
        parse_logprob_sidecar_mode({"save_logprob_distributions": True})
    with pytest.raises(ValueError, match="save_logprob_distributions"):
        parse_logprob_sidecar_mode({"save_logprob_distributions": False})
    with pytest.raises(ValueError, match="save_logprob_distributions"):
        parse_logprob_sidecar_mode(
            {"logprob_sidecar_mode": "action_window", "save_logprob_distributions": False}
        )


def test_full_instance_override_per_domain():
    overrides = parse_full_instance_overrides(
        {"logprob_sidecar_full_instances": {"textworld": [0], "tower_of_hanoi": [1]}}
    )
    cfg = LogprobSidecarConfig(
        default_mode="action_window",
        full_instances_by_domain=overrides,
    )
    assert cfg.mode_for("textworld", 0) == "full"
    assert cfg.mode_for("textworld", 1) == "action_window"
    assert cfg.mode_for("tower_of_hanoi", 1) == "full"


def test_filter_action_window_strips_reasoning():
    full = _c1_full_tokens()
    filtered = filter_logprob_raw_for_sidecar([full], "action_window")
    assert len(filtered[0]) == 2
    assert filtered[0][0]["token"] == "go"
    assert THINK_OPEN not in "".join(t["token"] for t in filtered[0])


def test_write_sidecar_action_window_metadata(tmp_path: Path):
    full = _c1_full_tokens()
    filtered = filter_logprob_raw_for_sidecar([full], "action_window")
    paths = write_logprob_distribution_artifacts(
        "ep_test",
        filtered,
        tmp_path,
        sidecar_scope="action_window",
    )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["sidecar_scope"] == "action_window"
    stored = payload["steps"][0]["logprob_tokens"]
    assert len(stored) == 2


def test_write_sidecar_full_retains_reasoning(tmp_path: Path):
    full = _c1_full_tokens()
    paths = write_logprob_distribution_artifacts(
        "ep_test",
        [full],
        tmp_path,
        sidecar_scope="full",
    )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["sidecar_scope"] == "full"
    assert len(payload["steps"][0]["logprob_tokens"]) == len(full)
