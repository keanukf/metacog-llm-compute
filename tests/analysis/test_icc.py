"""ICC estimation (``src/analysis/icc.py``).

Lifted 2026-07-28 from ``scripts/analysis_rehearsal/h3_power_simulation.py``'s private
``_anova_icc1``/``_gee_icc`` (revision_audit P1-stat-8) so the Stage 1 preanalysis screen can run
a real ICC check against actual data, not just a power simulation. Verifies both estimators
respond in the expected direction to cluster-level variance and degrade gracefully (return
``None``, never raise) on insufficient data.
"""

from __future__ import annotations

import random

from src.analysis.icc import anova_icc1, estimate_icc, gee_icc


def _clustered_binary_rows(
    *, n_clusters: int, n_per_cluster: int, cluster_spread: float, seed: int
) -> list[dict]:
    """Simulate binary outcomes with a per-cluster base rate offset by ``cluster_spread`` --
    larger spread means more between-cluster variance, i.e. a higher true ICC."""
    rng = random.Random(seed)
    rows = []
    for c in range(n_clusters):
        base_rate = 0.5 + cluster_spread * (rng.random() - 0.5)
        base_rate = min(max(base_rate, 0.01), 0.99)
        for _ in range(n_per_cluster):
            rows.append(
                {"instance_key": f"c{c}", "y_optimal": 1 if rng.random() < base_rate else 0}
            )
    return rows


def test_high_cluster_spread_yields_higher_icc_than_low_spread():
    low = _clustered_binary_rows(n_clusters=20, n_per_cluster=10, cluster_spread=0.02, seed=1)
    high = _clustered_binary_rows(n_clusters=20, n_per_cluster=10, cluster_spread=0.9, seed=1)
    icc_low = anova_icc1(low)
    icc_high = anova_icc1(high)
    assert icc_low is not None and icc_high is not None
    assert icc_high > icc_low


def test_gee_and_anova_roughly_agree():
    rows = _clustered_binary_rows(n_clusters=30, n_per_cluster=10, cluster_spread=0.6, seed=2)
    gee = gee_icc(rows)
    anova = anova_icc1(rows)
    assert gee is not None and anova is not None
    assert abs(gee - anova) < 0.25  # same underlying quantity, different estimators


def test_estimate_icc_returns_both_plus_counts():
    rows = _clustered_binary_rows(n_clusters=10, n_per_cluster=5, cluster_spread=0.5, seed=3)
    out = estimate_icc(rows)
    assert out["n_rows"] == 50
    assert out["n_clusters"] == 10
    assert out["icc_gee"] is not None
    assert out["icc_anova"] is not None


def test_insufficient_clusters_returns_none_not_raise():
    rows = [{"instance_key": "only_one", "y_optimal": 1}] * 5
    assert anova_icc1(rows) is None
    assert gee_icc(rows) is None
    out = estimate_icc(rows)
    assert out["icc_gee"] is None
    assert out["icc_anova"] is None


def test_missing_value_key_rows_are_skipped_not_erroring():
    rows = [{"instance_key": "c1"}, {"instance_key": "c1", "y_optimal": None}]
    assert anova_icc1(rows) is None
    assert gee_icc(rows) is None
