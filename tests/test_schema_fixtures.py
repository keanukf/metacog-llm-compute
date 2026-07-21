"""Pins the persisted episode artifact schema (episode.v1; ADR-003).

Asserts the checked-in fixture carries the required top-level keys and the ``episode.v1`` version
marker, so an accidental change to the on-disk episode schema -- which every downstream analysis
reads -- fails fast here.
"""

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
