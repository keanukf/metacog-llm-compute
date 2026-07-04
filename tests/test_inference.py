"""Smoke tests for inference layer."""

from __future__ import annotations

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
