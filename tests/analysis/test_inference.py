"""Smoke tests for inference layer."""

from __future__ import annotations

import math

from src.analysis.inference import bh, cluster_bootstrap, h2_paired, holm


def test_cluster_bootstrap_smoke():
    rows = [
        {"instance_key": f"t:{i // 3}", "y_optimal": i % 2, "tle_mean_entropy": 0.1 * i}
        for i in range(30)
    ]
    out = cluster_bootstrap(
        rows,
        lambda rs: sum(int(r["y_optimal"]) for r in rs) / len(rs),
        n_boot=200,
        seed=1,
    )
    assert out["point"] is not None
    assert out["ci_low"] is not None


def test_cluster_bootstrap_drops_nonfinite_replicates():
    """Regression for a real Gate E anomaly: a stat_fn that can return nan/inf on a
    degenerate cluster resample (e.g. delta_auroc when a resample has only one label
    class) must not let that nan pollute the sorted percentile array. Before the fix,
    Python's list.sort() does not order nan consistently, which corrupted ci_low/ci_high
    and could put the point estimate outside the reported CI -- exactly what was observed
    (unexplained) on the real TextWorld pilot pool in docs/gate_e_rehearsal.md section 6.
    """
    rows = [
        {"instance_key": f"c{i}", "y_optimal": i % 2, "tle_mean_entropy": 0.1 * i} for i in range(6)
    ]

    def stat_fn(rs):
        ys = {int(r["y_optimal"]) for r in rs}
        if len(ys) < 2:
            return float("nan")
        return sum(int(r["y_optimal"]) for r in rs) / len(rs)

    out = cluster_bootstrap(rows, stat_fn, n_boot=500, seed=1)
    assert out["point"] is not None
    assert out["ci_low"] is not None and out["ci_high"] is not None
    assert not math.isnan(out["ci_low"])
    assert not math.isnan(out["ci_high"])
    assert out["ci_low"] <= out["ci_high"]
    # Some resamples of only 6 clusters over 500 draws are expected to be single-class.
    assert out["n_boot_nonfinite"] > 0
    assert out["n_boot_effective"] == 500 - out["n_boot_nonfinite"]


def test_h2_paired_delta_default():
    eps = [
        {
            "domain": "tw",
            "instance": 0,
            "strategy": "adaptive_tle",
            "task_success": True,
            "total_tokens_generated": 100,
            "holdout": False,
        },
        {
            "domain": "tw",
            "instance": 0,
            "strategy": "always_c2",
            "task_success": True,
            "total_tokens_generated": 200,
            "holdout": False,
        },
    ]
    r = h2_paired(eps)
    assert r["delta"] == 0.05
    assert r["n_pairs"] == 1


def test_holm_bh():
    assert len(holm([0.01, 0.04, 0.2])) == 3
    assert len(bh([0.01, 0.04, 0.2])) == 3


def test_holm_enforces_monotonicity():
    """Regression: Holm's step-down procedure requires adjusted p-values to be a running
    max over increasing raw p-value rank (Holm 1979); a version that just applies
    ``min(1, raw * (m - rank))`` per-rank without the running max can produce a smaller
    adjusted p-value for a larger raw p-value, which is not a valid Holm adjustment.
    Expected values cross-checked against
    ``statsmodels.stats.multitest.multipletests(pvals, method="holm")``.
    """
    pvals = [0.01, 0.011, 0.5]
    out = holm(pvals)
    adjusted_by_index = {o["index"]: o["adjusted"] for o in out}
    assert math.isclose(adjusted_by_index[0], 0.03, rel_tol=1e-9)
    assert math.isclose(adjusted_by_index[1], 0.03, rel_tol=1e-9)
    assert math.isclose(adjusted_by_index[2], 0.5, rel_tol=1e-9)
    # Adjusted p-values must be non-decreasing in raw-p-value rank order.
    ordered = sorted(out, key=lambda o: o["raw"])
    adj_in_rank_order = [o["adjusted"] for o in ordered]
    assert adj_in_rank_order == sorted(adj_in_rank_order)
