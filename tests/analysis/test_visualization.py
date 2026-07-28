"""Phase 1 analysis Stage 6 plotting helpers (src.analysis.visualization).

Exercises plot_auroc_comparison_bars / plot_h3_marginal_effect against small synthetic
Stage 2 / Stage 4 result shapes, confirming they write real PNG files and degrade gracefully
without matplotlib.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.visualization import plot_auroc_comparison_bars, plot_h3_marginal_effect


def _h1a_fixture() -> dict:
    return {
        "family": "A",
        "by_domain": {
            "tower_of_hanoi": {"point": 0.09, "ci_low": 0.08, "ci_high": 0.11, "decision_holds": True},
            "textworld": {"point": -0.001, "ci_low": -0.03, "ci_high": 0.02, "decision_holds": False},
        },
        "descriptive_cross_check": {
            "tower_of_hanoi": {"optimal_only": {"tle": {"auroc": 0.71}, "vc": {"auroc": 0.62}}},
            "textworld": {"optimal_only": {"tle": {"auroc": 0.55}, "vc": {"auroc": 0.54}}},
        },
    }


def _h3_fixture() -> dict:
    return {
        "family": "E",
        "confirmatory_domain": "textworld",
        "exploratory_domain": "tower_of_hanoi",
        "results": {
            "textworld": {
                "tle": {
                    "converged": True,
                    "params": {"const": -1.6, "z_c": 0.6, "p_c": -0.02, "interaction": -1.37},
                },
                "vc": {"converged": False, "note": "insufficient data"},
            },
            "tower_of_hanoi": {
                "tle": {
                    "converged": True,
                    "params": {"const": -1.1, "z_c": 0.1, "p_c": -0.14, "interaction": 0.6},
                },
                "vc": {"converged": True, "params": {"const": -0.9, "z_c": 0.2, "p_c": 0.05, "interaction": 0.1}},
            },
        },
    }


def test_plot_auroc_comparison_bars_writes_png(tmp_path):
    out = plot_auroc_comparison_bars(_h1a_fixture(), tmp_path)
    assert "h1a_auroc_comparison" in out
    p = Path(out["h1a_auroc_comparison"])
    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_h3_marginal_effect_writes_one_png_per_converged_domain_signal(tmp_path):
    out = plot_h3_marginal_effect(_h3_fixture(), tmp_path)
    # 1 converged in textworld (tle) + 2 converged in tower_of_hanoi (tle, vc) == 3.
    assert len(out) == 3
    for path_str in out.values():
        p = Path(path_str)
        assert p.exists()
        assert p.stat().st_size > 0
    assert "h3_marginal_effect_textworld_vc" not in out


def test_plot_auroc_comparison_bars_empty_domains_returns_empty(tmp_path):
    out = plot_auroc_comparison_bars({"by_domain": {}, "descriptive_cross_check": {}}, tmp_path)
    assert out == {}
