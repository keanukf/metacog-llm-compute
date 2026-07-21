"""Tests for threshold gridsearch and compact episode persistence."""

from __future__ import annotations

from src.analysis.thresholds import (
    _match_proxy,
    build_ecdf_ref,
    build_ecdf_ref_by_stage,
    grid_search_thresholds,
)
from src.utils.logging_utils import compact_episode_for_storage


def test_build_ecdf_ref_by_stage_from_holdout():
    rows = []
    for stage in ("C0", "C1", "C2"):
        for i in range(5):
            rows.append(
                {
                    "holdout": True,
                    "compute_stage": stage,
                    "tle_mean_entropy": (0.5 if stage == "C0" else 1e-6) + i * 0.01,
                }
            )
    by_stage = build_ecdf_ref_by_stage(rows, signal="tle_mean_entropy")
    assert set(by_stage) == {"C0", "C1", "C2"}
    assert len(by_stage["C0"]) == 5
    assert by_stage["C1"][0] < 1e-3


def test_build_ecdf_ref_from_holdout():
    rows = [{"holdout": True, "tle_mean_entropy": 0.5 + i * 0.01} for i in range(20)]
    ref = build_ecdf_ref(rows, signal="tle_mean_entropy")
    assert len(ref) == 20
    assert ref == tuple(sorted(ref))


def test_grid_search_returns_36_candidates():
    holdout = []
    pool = []
    for inst in range(2):
        for run in range(2):
            for step in range(3):
                for stage in ("C0", "C1", "C2"):
                    row = {
                        "domain": "textworld",
                        "instance": inst,
                        "run": run,
                        "step_index": step,
                        "compute_stage": stage,
                        "holdout": inst == 0,
                        "tle_mean_entropy": 0.1 + step * 0.2 + (0.05 if stage == "C2" else 0),
                        "y_optimal": 1 if step == 0 else 0,
                        "tokens_generated": 10 if stage == "C0" else 30,
                    }
                    pool.append(row)
                    if inst == 0:
                        holdout.append(row)
    out = grid_search_thresholds(holdout, pool, signal="tle_mean_entropy")
    assert out["theta1"] is not None
    assert len(out["grid_table"]) == 36
    assert out["objective_definition"] == "step_level_proxy_v1"
    assert set(out["ecdf_by_stage"]) == {"C0", "C1", "C2"}


def test_match_proxy_exact():
    pool = {
        (1, 0, 2, "C1"): [{"y": 1.0, "tokens": 20.0, "step_index": 2}],
    }
    y, tokens, level = _match_proxy(pool, instance=1, run=0, step_index=2, stage="C1")
    assert level == "exact"
    assert y == 1.0
    assert tokens == 20.0


def test_match_proxy_mean_run():
    pool = {
        (1, 0, 2, "C0"): [{"y": 0.0, "tokens": 10.0, "step_index": 2}],
        (1, 1, 2, "C0"): [{"y": 1.0, "tokens": 30.0, "step_index": 2}],
    }
    y, tokens, level = _match_proxy(pool, instance=1, run=2, step_index=2, stage="C0")
    assert level == "mean_run"
    assert y == 0.5
    assert tokens == 20.0


def test_match_proxy_nearest_position_prefers_smaller_step_index_on_tie():
    pool = {
        (0, 0, 1, "C0"): [{"y": 0.0, "tokens": 10.0, "step_index": 1}],
        (0, 0, 3, "C0"): [{"y": 1.0, "tokens": 30.0, "step_index": 3}],
    }
    y, tokens, level = _match_proxy(pool, instance=0, run=0, step_index=2, stage="C0")
    assert level == "nearest_position"
    assert y == 0.0
    assert tokens == 10.0


def test_compact_episode_keeps_minimal_steps_detail():
    data = {
        "episode_id": "ep_x",
        "steps_detail": [
            {
                "step_index": 0,
                "compute_stage": "C0",
                "tle": {"mean_entropy": 0.2},
                "vc": 80.0,
                "tokens_generated": 5,
                "lm_calls": 1,
                "correctness": "optimal",
                "prompt_full": "should be stripped if not in minimal",
            }
        ],
        "vc_detail_per_step": [{"vc_raw_text": "big"}],
    }
    compact = compact_episode_for_storage(data)
    assert "vc_detail_per_step" not in compact
    assert len(compact["steps_detail"]) == 1
    assert compact["steps_detail"][0]["step_index"] == 0
    assert "prompt_full" not in compact["steps_detail"][0]
