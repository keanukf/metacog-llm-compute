"""Phase 2 prep: TextWorld threshold-sensitivity analysis
(scripts/phase2_prep/threshold_sensitivity_analysis.py).

CLI smoke test against a tiny synthetic Phase 1 manifest + Phase 2 checkpoint fixture (not the
real data). Verifies: the script fits two distinct (theta1, theta2) pairs from two distinct
holdout instance sets, looks the deployed pair up on the true-holdout grid, and reports a
routing-distribution comparison over synthetic "observed" Phase 2 steps that sums to 1.0 on both
sides.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.phase2_prep import threshold_sensitivity_analysis as tsa
from src.analysis.phase1_canonical import (
    TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL,
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
)

DOMAIN = "textworld"
UNION_INSTANCES = sorted(
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES | TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL
)


def _write_phase1_episode(
    dir_: Path, *, episode_id: str, instance: int, stage: str, seed: int
) -> None:
    rng = random.Random(seed)
    steps_detail = []
    for i in range(6):
        tle = rng.uniform(0.0, 1.0)
        p_correct = max(0.05, min(0.95, 1.0 - tle))
        correct = rng.random() < p_correct
        steps_detail.append(
            {
                "step_index": i,
                "compute_stage": stage,
                "tle": {"mean_entropy": tle},
                "vc": rng.uniform(0.0, 100.0),
                "correctness": "optimal" if correct else "illegal",
            }
        )
    (dir_ / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "domain": DOMAIN,
                "instance": instance,
                "compute_stage": stage,
                "run": 0,
                "holdout": True,  # overwritten by the correction regardless of this value
                "task_success": True,
                "episode_length_steps": 6,
                "steps_detail": steps_detail,
            }
        ),
        encoding="utf-8",
    )


def _build_phase1_manifest(tmp_path: Path) -> Path:
    source_dir = tmp_path / "phase1_source"
    source_dir.mkdir()
    entries = []
    for inst in UNION_INSTANCES:
        for stage in ("C0", "C1", "C2"):
            ep_id = f"ep_{DOMAIN}_{inst}_{stage}_0"
            _write_phase1_episode(
                source_dir,
                episode_id=ep_id,
                instance=inst,
                stage=stage,
                seed=inst * 7 + hash(stage) % 5,
            )
            entries.append({"episode_id": ep_id, "domain": DOMAIN, "source_dir": str(source_dir)})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_rule": "test fixture",
                "sources": {DOMAIN: str(source_dir)},
                "n_episodes": len(entries),
                "content_hash": "test",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _build_phase2_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "phase2_checkpoint"
    ckpt_dir.mkdir()
    rng = random.Random(99)
    for strategy in ("adaptive_tle", "adaptive_vc"):
        for inst in range(3):
            steps_detail = []
            for i in range(10):
                stage = rng.choice(["C0", "C1", "C2"])
                steps_detail.append(
                    {
                        "step_index": i,
                        "compute_stage": stage,
                        "tle": {"mean_entropy": rng.uniform(0.0, 1.0)},
                        "vc": rng.uniform(0.0, 100.0),
                        "correctness": rng.choice(["optimal", "legal", "illegal"]),
                    }
                )
            ep_id = f"ep_{DOMAIN}_{inst}_{strategy}_0"
            (ckpt_dir / f"{ep_id}.json").write_text(
                json.dumps(
                    {
                        "episode_id": ep_id,
                        "domain": DOMAIN,
                        "instance": inst,
                        "strategy": strategy,
                        "run": 0,
                        "holdout": False,
                        "task_success": True,
                        "episode_length_steps": 10,
                        "steps_detail": steps_detail,
                    }
                ),
                encoding="utf-8",
            )
    return ckpt_dir


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["threshold_sensitivity_analysis.py"] + argv)
    return tsa.main()


def test_reports_distinct_fits_and_valid_routing_distributions(tmp_path, monkeypatch, capsys):
    manifest_path = _build_phase1_manifest(tmp_path)
    checkpoint_dir = _build_phase2_checkpoint(tmp_path)
    out_path = tmp_path / "sensitivity.json"

    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--phase2-checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(out_path),
        ],
        monkeypatch,
    )
    assert rc == 0
    assert out_path.exists()

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["true_holdout_instances"] == sorted(TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    assert obj["deployed_wrong_holdout_instances_historical"] == sorted(
        TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL
    )

    for signal in ("tle_mean_entropy", "vc"):
        block = obj["by_signal"][signal]
        true_fit = block["true_holdout_fit"]
        deployed_fit = block["deployed_reconstructed_fit"]
        assert true_fit["theta1"] is not None
        assert deployed_fit["theta1"] is not None
        assert true_fit["theta1"] < true_fit["theta2"]
        assert deployed_fit["theta1"] < deployed_fit["theta2"]

        rr = block["realized_routing"]
        assert rr["n_observed_steps"] > 0
        for policy_key in ("true_holdout_policy", "deployed_policy"):
            frac = rr[policy_key]["fraction"]
            assert abs(sum(frac.values()) - 1.0) < 1e-9

    captured = capsys.readouterr()
    assert "OK -- threshold sensitivity analysis written to" in captured.out


def test_fails_loudly_on_missing_manifest(tmp_path, monkeypatch):
    rc = _run_main(
        [
            "--manifest",
            str(tmp_path / "missing.json"),
            "--phase2-checkpoint-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
        monkeypatch,
    )
    assert rc == 1


def test_fails_loudly_on_missing_checkpoint_dir(tmp_path, monkeypatch):
    manifest_path = _build_phase1_manifest(tmp_path)
    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--phase2-checkpoint-dir",
            str(tmp_path / "missing_checkpoint"),
            "--output",
            str(tmp_path / "out.json"),
        ],
        monkeypatch,
    )
    assert rc == 1
