"""Phase 1 analysis Stage 6 plotting helpers (src.analysis.visualization).

Exercises plot_auroc_comparison_bars / plot_h3_marginal_effect against small synthetic
Stage 2 / Stage 4 result shapes, confirming they write real PNG files and degrade gracefully
without matplotlib.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.visualization import (
    plot_auroc_comparison_bars,
    plot_bootstrap_distribution,
    plot_episode_length_boxplot,
    plot_h3_marginal_effect,
    plot_signal_boxplots,
    plot_signal_histograms,
)


def _h1a_fixture() -> dict:
    return {
        "family": "A",
        "by_domain": {
            "tower_of_hanoi": {
                "point": 0.09,
                "ci_low": 0.08,
                "ci_high": 0.11,
                "decision_holds": True,
            },
            "textworld": {
                "point": -0.001,
                "ci_low": -0.03,
                "ci_high": 0.02,
                "decision_holds": False,
            },
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
                "vc": {
                    "converged": True,
                    "params": {"const": -0.9, "z_c": 0.2, "p_c": 0.05, "interaction": 0.1},
                },
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


def _h3_step_fixture() -> list[dict]:
    import random

    rng = random.Random(1)
    rows = []
    for inst in range(6):
        for t in range(10):
            pos = t / 9.0
            tle = rng.uniform(0.0, 1.0)
            rows.append(
                {
                    "domain": "textworld",
                    "instance_key": f"tw:{inst}",
                    "compute_stage": "C0",
                    "y_optimal": 1 if rng.random() < 0.5 else 0,
                    "tle_mean_entropy": tle,
                    "position_norm": pos,
                }
            )
    return rows


def test_plot_h3_marginal_effect_with_steps_overlays_empirical_data(tmp_path):
    """Same fitted-curve output as without steps, but with a real scatter overlay -- exercised via
    a real (if tiny) step table so build_h3_frame's stage-wise standardization actually runs."""
    out_with = plot_h3_marginal_effect(_h3_fixture(), tmp_path / "with", steps=_h3_step_fixture())
    out_without = plot_h3_marginal_effect(_h3_fixture(), tmp_path / "without", steps=None)
    assert set(out_with.keys()) == set(out_without.keys())
    for path_str in out_with.values():
        p = Path(path_str)
        assert p.exists() and p.stat().st_size > 0


def test_plot_h3_marginal_effect_steps_too_sparse_degrades_to_fitted_curve_only(tmp_path):
    """Insufficient real data for build_h3_frame (< 20 rows) must not crash -- just no overlay."""
    sparse_steps = [
        {
            "domain": "textworld",
            "instance_key": "tw:0",
            "compute_stage": "C0",
            "y_optimal": 1,
            "tle_mean_entropy": 0.2,
            "position_norm": 0.1,
        }
    ]
    out = plot_h3_marginal_effect(_h3_fixture(), tmp_path, steps=sparse_steps)
    assert len(out) == 3
    for path_str in out.values():
        assert Path(path_str).exists()


def test_plot_auroc_comparison_bars_empty_domains_returns_empty(tmp_path):
    out = plot_auroc_comparison_bars({"by_domain": {}, "descriptive_cross_check": {}}, tmp_path)
    assert out == {}


def _step_fixture() -> list[dict]:
    steps = []
    for dom, tle_base, vc_base in (("tower_of_hanoi", 0.1, 60), ("textworld", 0.3, 45)):
        for i in range(40):
            steps.append(
                {
                    "domain": dom,
                    "tle_mean_entropy": tle_base + 0.01 * i,
                    "vc": vc_base + i % 10,
                }
            )
    return steps


def test_plot_signal_histograms_writes_one_png_per_domain(tmp_path):
    out = plot_signal_histograms(_step_fixture(), tmp_path)
    assert set(out.keys()) == {"hist_signals_tower_of_hanoi", "hist_signals_textworld"}
    for path_str in out.values():
        p = Path(path_str)
        assert p.exists() and p.stat().st_size > 0


def test_plot_signal_histograms_empty_input_returns_empty(tmp_path):
    assert plot_signal_histograms([], tmp_path) == {}


def test_plot_signal_boxplots_writes_png(tmp_path):
    out = plot_signal_boxplots(_step_fixture(), tmp_path)
    assert "boxplot_signals_by_domain" in out
    p = Path(out["boxplot_signals_by_domain"])
    assert p.exists() and p.stat().st_size > 0


def test_plot_episode_length_boxplot_writes_png(tmp_path):
    episodes = [{"domain": "tower_of_hanoi", "episode_length_steps": 20 + i} for i in range(10)] + [
        {"domain": "textworld", "episode_length_steps": 30 + i} for i in range(10)
    ]
    out = plot_episode_length_boxplot(episodes, tmp_path)
    assert "boxplot_episode_length" in out
    p = Path(out["boxplot_episode_length"])
    assert p.exists() and p.stat().st_size > 0


def test_plot_episode_length_boxplot_empty_input_returns_empty(tmp_path):
    assert plot_episode_length_boxplot([], tmp_path) == {}


def test_plot_bootstrap_distribution_writes_png(tmp_path):
    import random

    reps = [random.gauss(0.05, 0.02) for _ in range(500)]
    out = plot_bootstrap_distribution(
        reps, tmp_path, name="h1a_tower_of_hanoi", point=0.05, ci_low=0.02, ci_high=0.08
    )
    assert "bootstrap_dist_h1a_tower_of_hanoi" in out
    p = Path(out["bootstrap_dist_h1a_tower_of_hanoi"])
    assert p.exists() and p.stat().st_size > 0


def test_plot_bootstrap_distribution_empty_reps_returns_empty(tmp_path):
    assert plot_bootstrap_distribution([], tmp_path, name="empty") == {}
