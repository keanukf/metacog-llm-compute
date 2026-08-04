"""Phase 1 analysis pipeline, Stage 6 (visualizations).

CLI smoke test: feeds a tiny synthetic manifest + Stage 2/Stage 4 JSON in, asserts the figures
manifest and at least one PNG get written, and that a missing input stage fails loudly.

The manifest/episode fixture is built explicitly (not left to the CLI's default path) so this
test never silently depends on the real, host-specific canonical dataset -- Stage 6 now loads the
Stage 0 manifest too (to overlay empirical data on the H3 marginal-effect plot), and without an
explicit fixture here the default ``--manifest`` path would resolve to whatever real data happens
to already be on disk in this repo checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.phase1_analysis import stage6_visualizations as stage6

_H1A = {
    "family": "A",
    "by_domain": {
        "tower_of_hanoi": {"point": 0.09, "ci_low": 0.08, "ci_high": 0.11, "decision_holds": True},
    },
    "descriptive_cross_check": {
        "tower_of_hanoi": {"optimal_only": {"tle": {"auroc": 0.71}, "vc": {"auroc": 0.62}}},
    },
}

_H3 = {
    "family": "E",
    "confirmatory_domain": "textworld",
    "exploratory_domain": "tower_of_hanoi",
    "results": {
        "textworld": {
            "tle": {"converged": True, "params": {"const": -1.6, "z_c": 0.6, "p_c": -0.02, "interaction": -1.37}},
            "vc": {"converged": False, "note": "insufficient data"},
        },
        "tower_of_hanoi": {"tle": {"converged": False, "note": "insufficient data"}, "vc": {"converged": False, "note": "insufficient data"}},
    },
}


def _write_episode_with_steps(dir_: Path, *, episode_id: str, domain: str, instance: int, n_steps: int, seed: int) -> None:
    import random

    rng = random.Random(seed)
    steps_detail = []
    for i in range(n_steps):
        steps_detail.append(
            {
                "step_index": i,
                "compute_stage": "C0",
                "tle": {"mean_entropy": rng.uniform(0.0, 1.0)},
                "vc": rng.uniform(0.0, 100.0),
                "correctness": "optimal" if rng.random() < 0.5 else "illegal",
            }
        )
    (dir_ / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "domain": domain,
                "instance": instance,
                "compute_stage": "C0",
                "run": 0,
                "holdout": False,
                "task_success": True,
                "episode_length_steps": n_steps,
                "steps_detail": steps_detail,
            }
        ),
        encoding="utf-8",
    )


def _build_manifest_fixture(tmp_path: Path) -> Path:
    """Enough textworld episodes/steps for build_h3_frame's own thresholds (>=20 rows, >=3
    clusters) so the empirical-overlay code path actually runs, not just gets skipped."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    entries = []
    for inst in range(5):
        ep_id = f"ep_textworld_{inst}_C0_0"
        _write_episode_with_steps(source_dir, episode_id=ep_id, domain="textworld", instance=inst, n_steps=10, seed=inst)
        entries.append({"episode_id": ep_id, "domain": "textworld", "source_dir": str(source_dir)})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_rule": "test fixture",
                "sources": {"textworld": str(source_dir)},
                "n_episodes": len(entries),
                "content_hash": "test",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage6_visualizations.py"] + argv)
    return stage6.main()


def test_stage6_writes_figures_and_manifest(tmp_path, monkeypatch):
    manifest_path = _build_manifest_fixture(tmp_path)
    stage2_path = tmp_path / "h1a.json"
    stage4_path = tmp_path / "h3.json"
    out_dir = tmp_path / "figures"
    stage2_path.write_text(json.dumps(_H1A), encoding="utf-8")
    stage4_path.write_text(json.dumps(_H3), encoding="utf-8")

    rc = _run_main(
        [
            "--manifest", str(manifest_path),
            "--stage2", str(stage2_path),
            "--stage4", str(stage4_path),
            "--output-dir", str(out_dir),
        ],
        monkeypatch,
    )
    assert rc == 0

    manifest = json.loads((out_dir / "figures_manifest.json").read_text(encoding="utf-8"))
    assert "h1a_auroc_comparison" in manifest
    assert "h3_marginal_effect_textworld_tle" in manifest
    for path_str in manifest.values():
        assert Path(path_str).exists()


def test_stage6_fails_loudly_on_missing_manifest(tmp_path, monkeypatch):
    rc = _run_main(["--manifest", str(tmp_path / "missing_manifest.json")], monkeypatch)
    assert rc == 1


def test_stage6_fails_loudly_on_missing_stage_input(tmp_path, monkeypatch):
    manifest_path = _build_manifest_fixture(tmp_path)
    rc = _run_main(
        [
            "--manifest", str(manifest_path),
            "--stage2", str(tmp_path / "missing.json"),
            "--stage4", str(tmp_path / "missing2.json"),
        ],
        monkeypatch,
    )
    assert rc == 1
