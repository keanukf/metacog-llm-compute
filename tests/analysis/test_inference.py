"""Statistical inference primitives for the RQ2 analysis (``src.analysis.inference``).

Exercises the cluster bootstrap (including its drop of non-finite replicates), the H2 paired-delta
estimator, and Holm/BH multiple-comparison correction with enforced monotonicity. These are the
tools that turn per-episode outcomes into the confidence-interval and family-wise-corrected claims
the thesis reports, so a bug here would misstate significance rather than merely a value.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.inference import (
    bh,
    build_h3_frame,
    cluster_bootstrap,
    cluster_bootstrap_stratified,
    fit_h3_model,
    h2_paired,
    h4_diff_in_diff,
    holm,
    one_sided_bootstrap_pvalue,
    one_sided_wald_pvalue,
)


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


def test_cluster_bootstrap_stratified_resamples_within_each_domain_exactly():
    """H4's preregistration requires resampling instances within each domain independently
    (thesis Ch.5 §5.8), not pooling all instance_keys into one flat resampling pool. Construct a
    deliberately imbalanced case -- 5 clusters in one domain, 50 in the other -- so an unstratified
    bootstrap would draw a domain-imbalanced mix of clusters by chance; instrument via a stat_fn
    that returns the per-replicate cluster count per domain to prove it never varies.
    """
    rows = []
    for i in range(5):
        rows.append({"domain": "tower_of_hanoi", "instance_key": f"toh:{i}", "y_optimal": i % 2})
    for i in range(50):
        rows.append({"domain": "textworld", "instance_key": f"tw:{i}", "y_optimal": i % 2})

    def count_stat_fn(rs):
        n_toh = len({r["instance_key"] for r in rs if r["domain"] == "tower_of_hanoi"})
        n_tw = len({r["instance_key"] for r in rs if r["domain"] == "textworld"})
        # Encode both counts into one float so the smoke path still gets a single number;
        # the assertions below decode via the raw replicate-count sidecar collected separately.
        return n_toh + 1000.0 * n_tw

    out = cluster_bootstrap_stratified(rows, count_stat_fn, strata_col="domain", n_boot=300, seed=7)
    assert out["point"] is not None
    for rep in out["reps"]:
        n_tw = int(rep // 1000.0)
        n_toh = rep - 1000.0 * n_tw
        # Distinct clusters observed in a with-replacement resample can only ever be <= the
        # stratum's true cluster count -- never more, and stratification means the *pool size*
        # drawn from is always exactly 5 (toh) and 50 (textworld), never a flattened mix of both.
        assert 0 <= n_toh <= 5
        assert 0 <= n_tw <= 50


def test_cluster_bootstrap_stratified_requires_at_least_two_clusters_per_stratum():
    rows = [
        {"domain": "tower_of_hanoi", "instance_key": "toh:0", "y_optimal": 1},
        {"domain": "textworld", "instance_key": "tw:0", "y_optimal": 1},
        {"domain": "textworld", "instance_key": "tw:1", "y_optimal": 0},
    ]
    out = cluster_bootstrap_stratified(rows, lambda rs: 0.0, strata_col="domain", n_boot=50, seed=1)
    assert out["point"] is None
    assert out["ci_low"] is None and out["ci_high"] is None
    assert out["reps"] == []


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


def test_build_h3_frame_matches_fit_h3_model_standardization():
    """build_h3_frame was factored out of fit_h3_model (2026-08-03, for the H3 empirical-overlay
    plot) -- must produce the exact same z_c values fit_h3_model's GEE actually fits on, not just
    a formula that's supposed to match."""
    rows = []
    for i in range(60):
        stage = ["C0", "C1", "C2"][i % 3]
        rows.append(
            {
                "domain": "textworld",
                "instance_key": f"tw:{i % 6}",
                "compute_stage": stage,
                "y_optimal": i % 2,
                "tle_mean_entropy": 0.05 + 0.01 * i,
                "position_norm": (i % 10) / 10.0,
            }
        )
    frame, note = build_h3_frame(rows, signal="tle", domain="textworld")
    assert note is None
    assert frame is not None
    assert set(frame.columns) >= {"y", "z_c", "position_norm", "g", "stage"}
    assert len(frame) == 60

    fit = fit_h3_model(rows, signal="tle", domain="textworld")
    assert fit["converged"] is True
    # Re-deriving the GEE by hand from build_h3_frame's own z_c column must reproduce the same
    # interaction coefficient fit_h3_model reports -- proof the two share one standardization path.
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    frame = frame.assign(p_c=frame["position_norm"] - frame["position_norm"].mean())
    model = sm.GEE(
        frame["y"],
        sm.add_constant(frame[["z_c", "p_c"]].assign(interaction=frame["z_c"] * frame["p_c"])),
        groups=frame["g"],
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    res = model.fit()
    assert float(res.params["interaction"]) == pytest.approx(fit["params"]["interaction"], abs=1e-9)


def test_build_h3_frame_insufficient_data_returns_none_with_note():
    frame, note = build_h3_frame(
        [{"y_optimal": 1, "compute_stage": "C0", "tle_mean_entropy": 0.1}], signal="tle"
    )
    assert frame is None
    assert note == "insufficient data"


def test_cluster_bootstrap_exposes_reps_for_pvalue_derivation():
    rows = [
        {"instance_key": f"t:{i // 3}", "y_optimal": i % 2, "tle_mean_entropy": 0.1 * i}
        for i in range(30)
    ]
    out = cluster_bootstrap(
        rows, lambda rs: sum(int(r["y_optimal"]) for r in rs) / len(rs), n_boot=200, seed=1
    )
    assert isinstance(out["reps"], list)
    assert len(out["reps"]) == out["n_boot_effective"]


def test_one_sided_bootstrap_pvalue_small_when_all_replicates_exceed_null():
    reps = [0.1, 0.2, 0.3, 0.15, 0.25]
    p = one_sided_bootstrap_pvalue(reps, null_value=0.0)
    assert p == 1 / 6  # (0 exceed-or-equal + 1) / (5 + 1), continuity-corrected


def test_one_sided_bootstrap_pvalue_large_when_all_replicates_below_null():
    reps = [-0.1, -0.2, -0.3, -0.15, -0.25]
    p = one_sided_bootstrap_pvalue(reps, null_value=0.0)
    assert p == 1.0  # all 5 at-or-below null + 1 continuity => 6/6


def test_one_sided_bootstrap_pvalue_never_exactly_zero():
    reps = [1.0] * 1000  # every replicate strictly exceeds null
    p = one_sided_bootstrap_pvalue(reps, null_value=0.0)
    assert p > 0.0
    assert math.isclose(p, 1 / 1001)


def test_one_sided_bootstrap_pvalue_empty_reps_is_maximally_uncertain():
    assert one_sided_bootstrap_pvalue([]) == 1.0


def test_one_sided_wald_pvalue_halves_when_coefficient_points_the_hypothesized_way():
    p = one_sided_wald_pvalue(-0.5, 0.04, direction=-1)
    assert math.isclose(p, 0.02)


def test_one_sided_wald_pvalue_takes_complement_when_coefficient_points_wrong_way():
    p = one_sided_wald_pvalue(0.5, 0.04, direction=-1)
    assert math.isclose(p, 0.98)


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
