from __future__ import annotations

import json
from pathlib import Path


def test_episode_schema_fixture_contains_required_keys() -> None:
    fixture = Path("tests/fixtures/episode_schema_v1.json")
    with open(fixture) as f:
        payload = json.load(f)

    required = {
        "schema_version",
        "episode_id",
        "compute_stage",
        "task_success",
        "steps_detail",
        "tle_per_step",
        "vc_per_step",
    }
    assert required.issubset(payload.keys())
    assert payload["schema_version"] == "episode.v1"
    assert isinstance(payload["steps_detail"], list) and payload["steps_detail"]
