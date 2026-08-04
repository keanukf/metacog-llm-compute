"""Unit tests for the RQ1 calibration/discrimination metrics and dataset loading.

Covers the metric primitives the signal-quality analysis depends on (AUROC at perfect/no
separation, per-step-position calibration for H3, signal discrimination, strategy efficiency) and
the backward-compatible loader that synthesizes ``steps_detail`` from older compact episode JSONs.
"""

from __future__ import annotations

import json

import pytest

from src.analysis.calibration import (
    calibration_by_step_position,
    compute_auroc,
    compute_brier,
    compute_strategy_efficiency,
    signal_discrimination_report,
)
from src.utils.logging_utils import load_episodes, load_steps


def test_compute_auroc_perfect_separation():
    scores = [0.1, 0.2, 0.9, 1.0]
    labels = [0, 0, 1, 1]
    assert compute_auroc(scores, labels) == pytest.approx(1.0)


def test_compute_auroc_no_separation():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert compute_auroc(scores, labels) == pytest.approx(0.5)


def test_compute_auroc_matches_sklearn_on_random_data_with_ties():
    """General-case regression, not just the perfect/no-separation edge cases above -- catches a
    tie-handling or rank-sum regression that those two trivial cases can't."""
    import random

    from sklearn.metrics import roc_auc_score

    rng = random.Random(20260803)
    for _ in range(50):
        n = rng.randint(5, 100)
        # Coarse rounding forces genuine ties, exercising the average-rank tie-handling path.
        scores = [round(rng.gauss(0, 1), 1) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        if len(set(labels)) < 2:
            continue
        assert compute_auroc(scores, labels) == pytest.approx(
            roc_auc_score(labels, scores), abs=1e-9
        )


def test_compute_auroc_undefined_returns_chance_not_sklearn_valueerror():
    """sklearn.roc_auc_score raises ValueError with only one class present; this repo's own
    convention is to return 0.5 (chance) instead -- must stay true after the sklearn swap."""
    assert compute_auroc([0.1, 0.2, 0.3], [1, 1, 1]) == pytest.approx(0.5)
    assert compute_auroc([], []) == pytest.approx(0.5)


def test_compute_brier_perfect_predictions():
    assert compute_brier([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_compute_brier_matches_sklearn_on_random_data():
    """No test existed for compute_brier at all before this (2026-08-03 sklearn swap) -- closes
    the same kind of general-case gap test_compute_auroc_matches_sklearn... closes for AUROC."""
    import random

    from sklearn.metrics import brier_score_loss

    rng = random.Random(20260803)
    for _ in range(50):
        n = rng.randint(1, 100)
        preds = [rng.random() for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        assert compute_brier(preds, labels) == pytest.approx(
            brier_score_loss(labels, preds), abs=1e-9
        )


def test_compute_brier_empty_or_mismatched_returns_zero_not_sklearn_error():
    """sklearn.brier_score_loss raises on empty input; this repo's own convention returns 0.0."""
    assert compute_brier([], []) == pytest.approx(0.0)
    assert compute_brier([0.5], [1, 0]) == pytest.approx(0.0)  # mismatched length


def test_calibration_by_step_position_smoke():
    episodes = [
        {
            "episode_id": "ep_x",
            "steps_detail": [
                {
                    "step_index": 0,
                    "tle": {"mean_entropy": 0.2, "max_entropy": 0.3},
                    "vc": 90.0,
                    "correctness": "legal",
                },
                {
                    "step_index": 1,
                    "tle": {"mean_entropy": 0.9, "max_entropy": 1.0},
                    "vc": 10.0,
                    "correctness": "illegal",
                },
            ],
        }
    ]
    out = calibration_by_step_position(episodes, signal="vc", n_bins=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all("ece" in d and "brier" in d and "n_steps" in d for d in out)


def test_signal_discrimination_report_smoke():
    episodes = [
        {
            "episode_id": "ep_x",
            "steps_detail": [
                {
                    "step_index": 0,
                    "tle": {"mean_entropy": 0.2, "max_entropy": 0.3},
                    "vc": 90.0,
                    "correctness": "legal",
                },
                {
                    "step_index": 1,
                    "tle": {"mean_entropy": 0.9, "max_entropy": 1.0},
                    "vc": 10.0,
                    "correctness": "illegal",
                },
            ],
        }
    ]
    r = signal_discrimination_report(episodes, signal="vc")
    assert "auroc" in r and 0.0 <= r["auroc"] <= 1.0
    assert r["n_steps"] == 2


def test_compute_strategy_efficiency_groups():
    episodes = [
        {"strategy": "always_c0", "task_success": True, "normalized_compute_cost": 0.2},
        {"strategy": "always_c0", "task_success": False, "normalized_compute_cost": 0.2},
        {"strategy": "always_c2", "task_success": True, "normalized_compute_cost": 1.0},
    ]
    rows = compute_strategy_efficiency(episodes)
    assert any(r.get("strategy") == "always_c0" for r in rows)
    assert any(r.get("strategy") == "always_c2" for r in rows)


def test_load_episodes_backward_compat_synthesizes_steps_detail(tmp_path):
    ep = {
        "episode_id": "ep_legacy",
        "domain": "textworld",
        "instance": 0,
        "compute_stage": "C0",
        "run": 0,
        "task_success": True,
        "steps": 2,
        "tle_per_step": [
            {"mean_entropy": 0.1, "max_entropy": 0.1},
            {"mean_entropy": 0.2, "max_entropy": 0.2},
        ],
        "vc_per_step": [50.0, 60.0],
    }
    (tmp_path / "ep_legacy.json").write_text(json.dumps(ep), encoding="utf-8")
    eps = load_episodes(tmp_path)
    assert len(eps) == 1
    assert "steps_detail" in eps[0]
    assert len(eps[0]["steps_detail"]) == 2


def test_load_steps_flattens(tmp_path):
    ep = {
        "episode_id": "ep_ok",
        "domain": "textworld",
        "instance": 0,
        "compute_stage": "C0",
        "run": 0,
        "task_success": True,
        "steps": 1,
        "steps_detail": [
            {
                "step_index": 0,
                "compute_stage": "C0",
                "action": "go north",
                "tokens_generated": 5,
                "lm_calls_this_step": 1,
                "step_wall_time_s": 0.1,
                "tle": {"mean_entropy": 0.1, "max_entropy": 0.2},
                "vc": 80.0,
                "correctness": "legal",
                "observation_length_chars": 10,
            }
        ],
    }
    (tmp_path / "ep_ok.json").write_text(json.dumps(ep), encoding="utf-8")
    df = load_steps(tmp_path)
    # load_steps prefers pandas, but falls back to list[dict] in minimal environments
    if isinstance(df, list):
        assert len(df) == 1
        assert df[0]["episode_id"] == "ep_ok"
        assert df[0]["step_index"] == 0
        assert "tle_mean_entropy" in df[0]
    else:
        assert len(df) == 1
        assert "episode_id" in df.columns
        assert "step_index" in df.columns
        assert "tle_mean_entropy" in df.columns
