"""Tests for apply_production_n.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.apply_production_n import _recommended_n


def test_recommended_n_from_report(tmp_path: Path) -> None:
    report = {
        "recommended": {"max_concurrent_episodes": 16, "episodes_per_hour": 200.0},
        "results": [],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _recommended_n(json.loads(path.read_text())) == 16


def test_recommended_n_fallback_to_best_viable(tmp_path: Path) -> None:
    report = {
        "results": [
            {"max_concurrent_episodes": 8, "smoke_go": True, "episodes_per_hour": 100},
            {"max_concurrent_episodes": 16, "smoke_go": True, "episodes_per_hour": 180},
        ]
    }
    assert _recommended_n(report) == 16
