"""Phase 1 analysis pipeline, Stage 6 (visualizations).

CLI smoke test: feeds tiny synthetic Stage 2 / Stage 4 JSON files in, asserts the figures
manifest and at least one PNG get written, and that a missing input stage fails loudly.
"""

from __future__ import annotations

import json
import sys

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


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage6_visualizations.py"] + argv)
    return stage6.main()


def test_stage6_writes_figures_and_manifest(tmp_path, monkeypatch):
    stage2_path = tmp_path / "h1a.json"
    stage4_path = tmp_path / "h3.json"
    out_dir = tmp_path / "figures"
    stage2_path.write_text(json.dumps(_H1A), encoding="utf-8")
    stage4_path.write_text(json.dumps(_H3), encoding="utf-8")

    rc = _run_main(
        ["--stage2", str(stage2_path), "--stage4", str(stage4_path), "--output-dir", str(out_dir)],
        monkeypatch,
    )
    assert rc == 0

    manifest = json.loads((out_dir / "figures_manifest.json").read_text(encoding="utf-8"))
    assert "h1a_auroc_comparison" in manifest
    assert "h3_marginal_effect_textworld_tle" in manifest
    from pathlib import Path

    for path_str in manifest.values():
        assert Path(path_str).exists()


def test_stage6_fails_loudly_on_missing_stage_input(tmp_path, monkeypatch):
    rc = _run_main(
        ["--stage2", str(tmp_path / "missing.json"), "--stage4", str(tmp_path / "missing2.json")],
        monkeypatch,
    )
    assert rc == 1
