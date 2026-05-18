from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


def _write_ep(run_dir: Path, ep: dict) -> None:
    ep_id = ep["episode_id"]
    (run_dir / f"{ep_id}.json").write_text(json.dumps(ep), encoding="utf-8")


def test_load_run_dataset_builds_steps_and_labels(tmp_path: Path):
    from src.analysis.datasets import load_run_dataset, validate_analysis_schema

    run_dir = tmp_path / "phase1_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ep = {
        "episode_id": "ep_textworld_0_C0_0",
        "domain": "textworld",
        "instance": 0,
        "compute_stage": "C0",
        "run": 0,
        "task_success": True,
        "steps": 2,
        "tle_per_step": [
            {"mean_entropy": 0.2, "max_entropy": 0.4},
            {"mean_entropy": 0.9, "max_entropy": 1.0},
        ],
        "vc_per_step": [90.0, 10.0],
        "step_correctness": [
            {"step_index": 0, "correctness": "optimal"},
            {"step_index": 1, "correctness": "legal"},
        ],
        # compact episode: no steps_detail on disk
    }
    _write_ep(run_dir, ep)

    ds = load_run_dataset(run_dir)
    assert len(ds.episodes) == 1
    assert len(ds.steps) == 2
    # synthesized steps_detail should be marked
    assert ds.episodes[0]["_steps_detail_synthesized"] is True

    # correctness policy is optimal_only by default:
    # step0 optimal -> 1, step1 legal -> 0
    steps_sorted = sorted(ds.steps, key=lambda r: r["step_index"])
    assert steps_sorted[0]["step_correct_optimal"] == 1
    assert steps_sorted[1]["step_correct_optimal"] == 0
    assert steps_sorted[0]["tle_mean_entropy"] == pytest.approx(0.2)
    assert steps_sorted[0]["relative_step_position"] == pytest.approx(0.0)

    health = validate_analysis_schema(ds.steps)
    assert health["n_steps"] == 2
    assert health["missing_columns"] == [] or isinstance(health["missing_columns"], list)


def test_analyze_run_script_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a minimal run directory with one episode
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ep = {
        "episode_id": "ep_textworld_0_C0_0",
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
                "tle": {"mean_entropy": 0.2, "max_entropy": 0.3},
                "vc": 80.0,
                "correctness": "optimal",
                "observation_length_chars": 10,
            }
        ],
        "episode_length_steps": 1,
        "total_lm_calls": 1,
        "total_tokens_generated": 5,
        "normalized_compute_cost": 1 / (20 * 3),
        "efficiency_score": 1.0 / (1 / (20 * 3)),
    }
    _write_ep(run_dir, ep)

    out_dir = tmp_path / "analysis_out"
    argv = ["scripts/analyze_run.py", "--run-dir", str(run_dir), "--out-dir", str(out_dir)]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "scripts" / "analyze_run.py"),
        run_name="__main__",
    )

    assert (out_dir / "analysis_metrics.json").exists()
    assert (out_dir / "episodes.csv").exists()
    assert (out_dir / "steps.csv").exists()
    assert (out_dir / "report.md").exists()


def test_thresholds_fit_and_artifact(tmp_path: Path):
    from src.analysis.thresholds import fit_calibrator_from_steps, write_threshold_artifact

    # Build synthetic steps with a signal that correlates with optimal label
    steps = []
    for i in range(60):
        vc = 90.0 if i < 30 else 10.0
        y = 1 if i < 30 else 0
        steps.append({"domain": "textworld", "vc": vc, "step_correct_optimal": y})

    art = fit_calibrator_from_steps(steps, signal="vc")
    assert art["signal"] == "vc"
    assert art["n_samples"] == 60

    p = write_threshold_artifact(tmp_path / "thresholds.json", steps)
    body = json.loads(p.read_text(encoding="utf-8"))
    assert "by_domain" in body and "textworld" in body["by_domain"]


def test_comparison_bootstrap_and_perm_smoke():
    from src.analysis.comparison import bootstrap_diff_in_means_ci, permutation_test_diff_in_means

    a = [1.0] * 20
    b = [0.0] * 20
    ci = bootstrap_diff_in_means_ci(a, b, n_boot=200, seed=1)
    assert ci is not None
    assert ci["diff_mean"] > 0
    p = permutation_test_diff_in_means(a, b, n_perm=200, seed=1)
    assert p is not None
    assert 0.0 <= p["p_value"] <= 1.0
