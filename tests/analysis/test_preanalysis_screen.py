"""Pre-analysis data quality screen (``src/analysis/preanalysis_screen.py``).

No test file existed for this module before 2026-07-28 (revision_audit P1-stat-8 cross-check).
Covers the pre-existing per-domain screen shape plus the three checks added to close that gap:
episode-length distribution (quartiles, not just mean), position x correctness empty-cell/
class-balance bins, and real ICC estimation on actual data (not just a power simulation).
"""

from __future__ import annotations

from src.analysis.preanalysis_screen import (
    _episode_length_distribution,
    _position_correctness_bins,
    _quantile,
    run_preanalysis_screen,
)


def test_quantile_matches_known_values():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _quantile(vals, 0.0) == 1.0
    assert _quantile(vals, 0.5) == 3.0
    assert _quantile(vals, 1.0) == 5.0


def test_episode_length_distribution_reports_full_spread_not_just_mean():
    lengths = [5, 10, 10, 15, 45, 45, 45]
    dist = _episode_length_distribution(lengths)
    assert dist["min"] == 5
    assert dist["max"] == 45
    assert dist["n"] == 7
    assert dist["median"] == 15
    assert dist["q1"] < dist["median"] < dist["q3"]


def test_episode_length_distribution_empty_input():
    dist = _episode_length_distribution([])
    assert dist["n"] == 0
    assert dist["min"] is None


def test_position_correctness_bins_flags_empty_cells():
    # All rows land in the first half of position_norm space -> the back half's bins are empty.
    rows = [
        {"position_norm": 0.05 * i, "y_optimal": i % 2}
        for i in range(10)  # positions 0.0 .. 0.45, so bins covering [0.5, 1.0) are all empty
    ]
    report = _position_correctness_bins(rows, n_bins=10)
    assert report["n_empty_cells"] > 0
    assert all(b["position_range"][0] >= 0.5 for b in report["bins"] if b["bin"] in report["empty_cell_bins"])


def test_position_correctness_bins_no_empty_cells_when_evenly_spread():
    rows = []
    for b in range(10):
        for _ in range(5):
            rows.append({"position_norm": b / 10.0 + 0.01, "y_optimal": 1})
    report = _position_correctness_bins(rows, n_bins=10)
    assert report["n_empty_cells"] == 0
    assert all(bin_["n"] == 5 for bin_ in report["bins"])


def test_run_preanalysis_screen_includes_new_checks_per_domain():
    rng_vals = [0.05, 0.12, 0.3, 0.08, 0.4, 0.15, 0.2, 0.35]
    steps = []
    episodes = []
    for i in range(60):
        dom = "textworld" if i % 2 == 0 else "tower_of_hanoi"
        steps.append(
            {
                "domain": dom,
                "instance_key": f"{dom}:{i % 6}",
                "y_optimal": i % 3 == 0,
                "tle_mean_entropy": rng_vals[i % len(rng_vals)],
                "vc": 50 + i,
                "position_norm": (i % 10) / 10.0,
            }
        )
    for i in range(10):
        dom = "textworld" if i % 2 == 0 else "tower_of_hanoi"
        episodes.append({"domain": dom, "episode_length_steps": 10 + i})

    screen = run_preanalysis_screen(steps, episodes)
    for dom, report in screen["by_domain"].items():
        assert "icc" in report
        assert "n_clusters" in report["icc"]
        assert "position_correctness" in report
        assert "bins" in report["position_correctness"]
        assert "episode_length_distribution" in report
        assert "median" in report["episode_length_distribution"]
    assert "length_distribution" in screen["episodes"]
