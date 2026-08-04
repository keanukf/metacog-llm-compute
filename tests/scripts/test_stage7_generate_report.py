"""Phase 1 analysis pipeline, Stage 7 (report generation).

CLI smoke test: feeds tiny synthetic Stage 0-6 JSON output in, asserts the report is written
with the expected section headers and that figures get copied into the docs-committed location.
"""

from __future__ import annotations

import json
import sys

from scripts.phase1_analysis import stage7_generate_report as stage7

_MANIFEST = {
    "selection_rule": "tower_of_hanoi from dirA; textworld from dirB",
    "sources": {},
    "n_episodes": 1500,
    "content_hash": "abc123" * 10,
    "entries": [],
}

_SCREEN = {
    "by_domain": {
        "tower_of_hanoi": {
            "n_steps": 100,
            "n_clusters": 10,
            "vc_missing_rate": 0.01,
            "icc": {"icc_gee": 0.02},
            "episode_length_distribution": {"n": 10, "min": 1, "q1": 2, "median": 3, "q3": 4, "max": 5},
            "position_correctness": {"n_empty_cells": 0},
        },
        "textworld": {
            "n_steps": 90,
            "n_clusters": 10,
            "vc_missing_rate": 0.05,
            "icc": {"icc_gee": 0.03},
            "episode_length_distribution": {"n": 10, "min": 1, "q1": 2, "median": 3, "q3": 4, "max": 5},
            "position_correctness": {"n_empty_cells": 1},
        },
    }
}

_H1A = {
    "by_domain": {
        "tower_of_hanoi": {
            "point": 0.09,
            "ci_low": 0.08,
            "ci_high": 0.11,
            "one_sided_pvalue": 0.0002,
            "holm": {"adjusted": 0.0004},
            "decision_holds": True,
        },
        "textworld": {
            "point": -0.001,
            "ci_low": -0.03,
            "ci_high": 0.02,
            "one_sided_pvalue": 0.5,
            "holm": {"adjusted": 0.5},
            "decision_holds": False,
        },
    },
    "descriptive_cross_check": {
        "tower_of_hanoi": {"optimal_only": {"tle": {"auroc": 0.71}, "vc": {"auroc": 0.62}}},
        "textworld": {"optimal_only": {"tle": {"auroc": 0.55}, "vc": {"auroc": 0.54}}},
    },
}

_H1B = {
    "by_domain": {
        "tower_of_hanoi": {
            "point": -0.15,
            "ci_low": -0.17,
            "ci_high": -0.14,
            "calibrator_slope": -9.3,
            "calibrator_converged": True,
            "n_holdout_steps": 2128,
            "holm": {"adjusted": 0.0004},
            "decision_holds": True,
        },
        "textworld": {
            "point": -0.10,
            "ci_low": -0.11,
            "ci_high": -0.09,
            "calibrator_slope": -5.2,
            "calibrator_converged": True,
            "n_holdout_steps": 2398,
            "holm": {"adjusted": 0.0004},
            "decision_holds": True,
        },
    }
}

_H3 = {
    "confirmatory_domain": "textworld",
    "exploratory_domain": "tower_of_hanoi",
    "results": {
        "textworld": {
            "tle": {
                "converged": True,
                "interaction_coef": -1.37,
                "one_sided_pvalue_degradation": 0.0000021,
                "holm": {"adjusted": 0.0000043},
                "decision_holds": True,
            },
            "vc": {"converged": False, "note": "insufficient data"},
        },
        "tower_of_hanoi": {
            "tle": {"converged": True, "interaction_coef": 0.6, "note": "exploratory only"},
            "vc": {"converged": False, "note": "insufficient data"},
        },
    },
}

_H4 = {
    "result": {
        "point": 0.095,
        "ci_low": 0.068,
        "ci_high": 0.125,
        "one_sided_pvalue": 0.0002,
        "holm": {"adjusted": 0.0002},
        "decision_holds": True,
    }
}


def _write_fake_figures(figures_dir, names):
    figures_dir.mkdir(parents=True)
    manifest = {}
    for name in names:
        p = figures_dir / f"{name}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        manifest[name] = str(p)
    (figures_dir / "figures_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_all(stage_dir):
    (stage_dir / "stage0").mkdir(parents=True)
    (stage_dir / "stage0" / "canonical_manifest.json").write_text(json.dumps(_MANIFEST), encoding="utf-8")
    (stage_dir / "stage1").mkdir(parents=True)
    (stage_dir / "stage1" / "preanalysis_screen.json").write_text(json.dumps(_SCREEN), encoding="utf-8")
    (stage_dir / "stage1" / "variable_codebook.md").write_text(
        "*Table 1*\n\n*Variable roles and measurement scales*\n\n| Variable |\n|---|\n| domain |\n",
        encoding="utf-8",
    )
    (stage_dir / "stage2").mkdir(parents=True)
    (stage_dir / "stage2" / "h1a_discrimination.json").write_text(json.dumps(_H1A), encoding="utf-8")
    (stage_dir / "stage3").mkdir(parents=True)
    (stage_dir / "stage3" / "h1b_calibration.json").write_text(json.dumps(_H1B), encoding="utf-8")
    (stage_dir / "stage4").mkdir(parents=True)
    (stage_dir / "stage4" / "h3_temporal.json").write_text(json.dumps(_H3), encoding="utf-8")
    (stage_dir / "stage5").mkdir(parents=True)
    (stage_dir / "stage5" / "h4_domain_modulation.json").write_text(json.dumps(_H4), encoding="utf-8")

    _write_fake_figures(
        stage_dir / "stage1" / "figures",
        ["hist_signals_tower_of_hanoi", "hist_signals_textworld", "boxplot_signals_by_domain", "boxplot_episode_length"],
    )
    _write_fake_figures(
        stage_dir / "stage2" / "figures", ["bootstrap_dist_h1a_tower_of_hanoi", "bootstrap_dist_h1a_textworld"]
    )
    _write_fake_figures(
        stage_dir / "stage3" / "figures",
        [
            "bootstrap_dist_h1b_tower_of_hanoi",
            "bootstrap_dist_h1b_textworld",
            "reliability_tle_mapped_tower_of_hanoi",
            "reliability_vc_tower_of_hanoi",
            "reliability_tle_mapped_textworld",
            "reliability_vc_textworld",
        ],
    )
    _write_fake_figures(stage_dir / "stage5" / "figures", ["bootstrap_dist_h4"])
    _write_fake_figures(
        stage_dir / "stage6" / "figures",
        ["h1a_auroc_comparison", "h3_marginal_effect_textworld_tle", "h3_marginal_effect_tower_of_hanoi_tle"],
    )


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage7_generate_report.py"] + argv)
    return stage7.main()


def test_stage7_generates_report_with_all_sections(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stages"
    _write_all(stage_dir)
    report_out = tmp_path / "report.md"
    figures_out = tmp_path / "docs_figures"

    rc = _run_main(
        [
            "--stage-dir", str(stage_dir),
            "--report-out", str(report_out),
            "--figures-out", str(figures_out),
        ],
        monkeypatch,
    )
    assert rc == 0
    text = report_out.read_text(encoding="utf-8")
    assert "# Phase 1 Real-Data Analysis Report" in text
    assert "## H1a" in text
    assert "## H1b" in text
    assert "## H3" in text
    assert "## H4" in text
    assert (figures_out / "h1a_auroc_comparison.png").exists()

    # Every stage's figures got copied and embedded, not just Stage 6's.
    assert (figures_out / "bootstrap_dist_h1a_tower_of_hanoi.png").exists()
    assert (figures_out / "bootstrap_dist_h1b_tower_of_hanoi.png").exists()
    assert (figures_out / "reliability_tle_mapped_tower_of_hanoi.png").exists()
    assert (figures_out / "bootstrap_dist_h4.png").exists()
    assert (figures_out / "hist_signals_textworld.png").exists()
    assert "bootstrap_dist_h1a_tower_of_hanoi.png" in text
    assert "reliability_tle_mapped_tower_of_hanoi.png" in text
    assert "bootstrap_dist_h4.png" in text

    # The full variable codebook (all 9 descriptive-stats tables) is embedded, not just the
    # curated preanalysis summary table.
    assert "Full variable codebook" in text
    assert "Variable roles and measurement scales" in text


def test_stage7_fails_loudly_on_missing_stage_input(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stages"
    stage_dir.mkdir()
    rc = _run_main(["--stage-dir", str(stage_dir)], monkeypatch)
    assert rc == 1


def test_stage7_report_is_idempotent(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stages"
    _write_all(stage_dir)
    report_out = tmp_path / "report.md"
    figures_out = tmp_path / "docs_figures"
    argv = [
        "--stage-dir", str(stage_dir),
        "--report-out", str(report_out),
        "--figures-out", str(figures_out),
    ]
    _run_main(argv, monkeypatch)
    first = report_out.read_text(encoding="utf-8")
    _run_main(argv, monkeypatch)
    second = report_out.read_text(encoding="utf-8")
    assert first == second
