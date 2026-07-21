"""Compact episode fixture must expose analysis-join fields from a real smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "episode_compact_real.json"

_EPISODE_JOIN_KEYS = ("episode_id", "domain", "instance", "run")
_STEP_REQUIRED_KEYS = (
    "step_index",
    "compute_stage",
    "tle",
    "vc",
    "tokens_generated",
    "correctness",
)


@pytest.fixture
def compact_episode() -> dict:
    if not FIXTURE.is_file():
        pytest.skip(
            f"Missing real smoke fixture {FIXTURE.name}: run Pod smoke with "
            "configs/dev/smoke.yaml, copy one compact episode JSON to tests/fixtures/, "
            f"and check in as {FIXTURE.name}"
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_compact_episode_has_analysis_fields(compact_episode: dict) -> None:
    ep = compact_episode
    assert isinstance(ep.get("holdout"), bool), "holdout must be a bool on episode level"
    assert ep.get("difficulty_tier") is not None, "difficulty_tier must be set on episode level"
    synthesized = ep.get("_steps_detail_synthesized")
    assert synthesized is None or synthesized is False, (
        "_steps_detail_synthesized must be absent or false for persisted steps_detail"
    )

    for key in _EPISODE_JOIN_KEYS:
        assert key in ep, f"missing episode join key {key!r}"

    steps_detail = ep.get("steps_detail")
    assert isinstance(steps_detail, list) and steps_detail, "steps_detail must be a non-empty list"

    for sd in steps_detail:
        assert isinstance(sd, dict), "each steps_detail entry must be a dict"
        for key in _STEP_REQUIRED_KEYS:
            assert key in sd, f"missing step key {key!r} in steps_detail"
        assert "lm_calls" in sd or "lm_calls_this_step" in sd, (
            "step must include lm_calls or lm_calls_this_step"
        )
        assert sd.get("compute_stage") == ep.get("compute_stage") or sd.get("compute_stage"), (
            "compute_stage must be present on each step"
        )

    join_values = {k: ep[k] for k in _EPISODE_JOIN_KEYS}
    assert all(join_values[k] is not None for k in _EPISODE_JOIN_KEYS), (
        "episode join keys must be non-null for dataset flattening"
    )
