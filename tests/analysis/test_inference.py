"""Statistical inference primitives for the RQ2 analysis (``src.analysis.inference``).

Exercises the cluster bootstrap (including its drop of non-finite replicates), the H2 paired-delta
estimator, and Holm/BH multiple-comparison correction with enforced monotonicity. These are the
tools that turn per-episode outcomes into the confidence-interval and family-wise-corrected claims
the thesis reports, so a bug here would misstate significance rather than merely a value.
"""

from __future__ import annotations

import math

from src.analysis.inference import bh, cluster_bootstrap, fit_h3_model, h2_paired, h4_diff_in_diff, holm


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


def test_h2_paired_decides_on_ci_bound_not_point_estimate():
    """Regression for P0-5/P0-7 (revision_audit_2026-07.md): h2_paired must decide on the
    cluster-bootstrap CI bound, not the raw paired-mean point estimate. Construction: 10
    instances alternate which arm wins, so mean_success_diff = 0.0 -- a naive point-estimate
    check (0.0 > -delta) would say non-inferiority holds -- but the high instance-to-instance
    variance this alternation creates means the bootstrap CI lower bound is far below -delta,
    so the correct (CI-bound) decision is that non-inferiority does NOT hold.
    """
    episodes = []
    for i in range(10):
        succ_p = i % 2 == 0
        episodes.append(
            {
                "domain": "tw",
                "instance": i,
                "strategy": "adaptive_tle",
                "task_success": succ_p,
                "total_tokens_generated": 100,
                "holdout": False,
            }
        )
        episodes.append(
            {
                "domain": "tw",
                "instance": i,
                "strategy": "always_c2",
                "task_success": not succ_p,
                "total_tokens_generated": 100,
                "holdout": False,
            }
        )
    r = h2_paired(episodes, delta=0.05)
    assert r["mean_success_diff"] == 0.0
    assert r["success_ci_low"] is not None and r["success_ci_low"] < -0.05
    assert r["non_inferiority_holds"] is False


def test_h2_paired_holds_when_ci_bound_clears_threshold():
    """Sanity counterpart: a well-powered, consistent effect (policy matches baseline success
    everywhere and uses a quarter of the tokens everywhere) should show both decisions True,
    with the CI collapsed onto the point estimate since there is zero instance-to-instance
    variance in this construction."""
    episodes = []
    for i in range(10):
        episodes.append(
            {
                "domain": "tw",
                "instance": i,
                "strategy": "adaptive_tle",
                "task_success": True,
                "total_tokens_generated": 50,
                "holdout": False,
            }
        )
        episodes.append(
            {
                "domain": "tw",
                "instance": i,
                "strategy": "always_c2",
                "task_success": True,
                "total_tokens_generated": 200,
                "holdout": False,
            }
        )
    r = h2_paired(episodes, delta=0.05)
    assert r["non_inferiority_holds"] is True
    assert r["token_superiority_holds"] is True
    assert r["log_token_ci_low"] is not None and r["log_token_ci_low"] > 0


def test_h4_diff_in_diff_sign_matches_preregistered_direction():
    """Regression: h4_diff_in_diff must return
    [AUROC_TLE - AUROC_VC]_ToH - [AUROC_TLE - AUROC_VC]_TextWorld, matching thesis H_{1,4}
    (ch.4 4.2.2), not the reversed textworld-minus-toh order a prior version silently
    computed. Construction: ToH has a perfectly TLE-discriminating, VC-uninformative
    signal pair (delta = 1.0 - 0.5 = 0.5); TextWorld has neither signal discriminating
    (delta = 0.5 - 0.5 = 0.0). A correct implementation returns 0.5 - 0.0 = 0.5 (positive,
    the H4-supporting direction); the reversed order would instead return -0.5.
    """
    toh_rows = [
        {
            "domain": "tower_of_hanoi",
            "y_optimal": i % 2,
            # negated inside delta_auroc -> low entropy on the correct (y=1) steps
            # gives a perfectly TLE-discriminating pair (AUROC_TLE = 1.0).
            "tle_mean_entropy": 0.1 if i % 2 == 1 else 0.9,
            "vc": 50,  # constant -> uninformative, AUROC_VC = 0.5
        }
        for i in range(10)
    ]
    tw_rows = [
        {
            "domain": "textworld",
            "y_optimal": i % 2,
            "tle_mean_entropy": 50,  # constant -> AUROC_TLE = 0.5
            "vc": 50,  # constant -> AUROC_VC = 0.5
        }
        for i in range(10)
    ]
    result = h4_diff_in_diff(toh_rows + tw_rows)
    assert math.isclose(result, 0.5, abs_tol=1e-9)


def test_fit_h3_model_converges_with_multi_stage_data():
    rng_vals = [0.05, 0.12, 0.3, 0.08, 0.4, 0.15, 0.2, 0.35]
    rows = []
    for i in range(60):
        stage = ["C0", "C1", "C2"][i % 3]
        rows.append(
            {
                "domain": "textworld",
                "instance_key": f"tw:{i % 6}",
                "compute_stage": stage,
                "y_optimal": i % 2,
                "tle_mean_entropy": rng_vals[i % len(rng_vals)] + 0.01 * i,
                "position_norm": (i % 10) / 10.0,
            }
        )
    out = fit_h3_model(rows, signal="tle", domain="textworld")
    assert out["converged"] is True
    assert "z_c" in out["params"] or "interaction" in out["params"]


def test_fit_h3_model_standardizes_per_stage_not_pooled():
    """Regression for P0-5 (ADR-006): a signal that is constant WITHIN one compute_stage but
    varies across the pooled dataset (because a different stage varies) must be caught as
    zero-variance -- a domain-wide-only check would miss this, since pooling stage C1's
    variation in with C0's constant values gives nonzero variance overall. Before the fix,
    fit_h3_model didn't group by compute_stage at all (mean-centered the pooled column, never
    checked variance), so this exact case would have silently produced a degenerate,
    uninterpretable coefficient instead of the explicit "insufficient data" style failure
    stage-conditional standardization requires.
    """
    rows = []
    for i in range(15):
        rows.append(
            {
                "domain": "textworld",
                "instance_key": f"c0:{i % 3}",
                "compute_stage": "C0",
                "y_optimal": i % 2,
                "tle_mean_entropy": 0.5,  # constant within C0 -> zero variance in this stage
                "position_norm": 0.1,
            }
        )
    for i in range(15):
        rows.append(
            {
                "domain": "textworld",
                "instance_key": f"c1:{i % 3}",
                "compute_stage": "C1",
                "y_optimal": i % 2,
                "tle_mean_entropy": 0.1 + 0.05 * i,  # varies -> pooled variance is nonzero
                "position_norm": 0.1,
            }
        )
    out = fit_h3_model(rows, signal="tle", domain="textworld")
    assert out["converged"] is False
    assert "variance" in out["note"]


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
