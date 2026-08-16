#!/usr/bin/env python3
"""B1 preregistered label-collapse sensitivity: repeat the focal Phase 1 analyses under the
alternative `legal_or_optimal` correctness collapse and report side-by-side with `optimal_only`.

The confirmatory analyses code only optimal steps as correct (`y_optimal`). Section 5.8 promises a
preregistered sensitivity under the alternative collapse that also counts distance-neutral legal
moves (examine/look/inventory) as correct (`y_legal_or_optimal`, precomputed in the canonical steps).
This script runs, per domain, under BOTH labels, reusing the exact confirmatory estimators (the
stage-stratified H1a/H4 estimand adopted after the compute-stage-confound correction, the
preregistered H1b ΔBrier, and the H3 GEE interaction), by relabelling `y_optimal := y_legal_or_optimal`
for the alternative run. Read-only, no GPU.

Run from repo root:
  python scripts/phase1_analysis/b1_label_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis.calibration import compute_brier, fit_tle_calibrator, vc_to_prob  # noqa: E402
from src.analysis.inference import (  # noqa: E402
    cluster_bootstrap,
    cluster_bootstrap_stratified,
    delta_auroc,
    fit_h3_model,
    holm,
    one_sided_bootstrap_pvalue,
    one_sided_wald_pvalue,
)
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)
from stage_stratified_sensitivity import h4, strat_dauroc  # noqa: E402

DOMS = ("tower_of_hanoi", "textworld")
STG = ("C0", "C1", "C2")
NB = 5000
SEED = 20260703
LABELS = ("optimal_only", "legal_or_optimal")


def _sig(a, b, x):
    import math

    return 1.0 / (1.0 + math.exp(-(a + b * x)))


def _relabel(steps, label_field):
    for r in steps:
        r["y_optimal"] = r.get(label_field)


def _occupancy(steps, dom):
    print(f"  {dom} class occupancy (correct rate) by stage:")
    for s in STG:
        ys = [
            int(r["y_optimal"])
            for r in steps
            if str(r.get("domain")) == dom
            and str(r.get("compute_stage")) == s
            and r.get("y_optimal") is not None
        ]
        rate = sum(ys) / len(ys) if ys else float("nan")
        print(f"    {s}: n={len(ys):6d}  correct_rate={rate:.3f}")


def _h1b_delta_brier(dom_steps):
    """Preregistered ΔBrier = Brier(TLE-mapped) - Brier(VC/100); fixed holdout calibrator,
    cluster-bootstrapped over non-holdout instances (mirrors stage3_h1b)."""
    holdout = [r for r in dom_steps if bool(r.get("holdout"))]
    cal = fit_tle_calibrator(holdout, label="y_optimal")
    if not hasattr(cal, "intercept"):
        return None
    nonh = [
        r
        for r in dom_steps
        if not bool(r.get("holdout"))
        and r.get("y_optimal") is not None
        and r.get("tle_mean_entropy") is not None
        and r.get("vc") is not None
    ]

    def stat(rows):
        ys = [int(r["y_optimal"]) for r in rows]
        p_tle = [_sig(cal.intercept, cal.slope, float(r["tle_mean_entropy"])) for r in rows]
        p_vc = [vc_to_prob(float(r["vc"])) for r in rows]
        return compute_brier(p_tle, ys) - compute_brier(p_vc, ys)

    return cluster_bootstrap(nonh, stat, n_boot=NB, seed=SEED)


def run_label(steps, label_field):
    print(f"\n########## LABEL = {label_field} ##########")
    _relabel(steps, label_field)
    for dom in DOMS:
        _occupancy(steps, dom)

    print("\n  H1a  stage-stratified ΔAUROC(TLE-VC) [90% CI]  (confirmatory estimand):")
    for dom in DOMS:
        dr = [r for r in steps if str(r.get("domain")) == dom]
        bs = cluster_bootstrap(dr, strat_dauroc, n_boot=NB, seed=SEED)
        bp = cluster_bootstrap(dr, delta_auroc, n_boot=NB, seed=SEED)
        print(
            f"    {dom:15s} stratified={bs['point']:+.4f} [{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}]"
            f"   pooled(ref)={bp['point']:+.4f} [{bp['ci_low']:+.4f},{bp['ci_high']:+.4f}]"
        )

    print("\n  H1b  ΔBrier(TLE-mapped - VC/100) [90% CI]  (preregistered, as-emitted VC):")
    for dom in DOMS:
        dr = [r for r in steps if str(r.get("domain")) == dom]
        b = _h1b_delta_brier(dr)
        if b:
            print(f"    {dom:15s} ΔBrier={b['point']:+.4f} [{b['ci_low']:+.4f},{b['ci_high']:+.4f}]")

    print("\n  H4  stage-stratified DiD (ToH-TextWorld) [90% CI]:")
    bd = cluster_bootstrap_stratified(steps, h4(strat_dauroc), n_boot=NB, seed=SEED)
    p = one_sided_bootstrap_pvalue(bd["reps"], null_value=0.0)
    print(f"    stratified DiD={bd['point']:+.4f} [{bd['ci_low']:+.4f},{bd['ci_high']:+.4f}]  p={p:.4f}")

    print("\n  H3  TextWorld interaction coefficient (signal x position):")
    for sig in ("tle", "vc"):
        fit = fit_h3_model(steps, signal=sig, domain="textworld")
        if isinstance(fit, dict) and fit.get("params") is not None:
            coef = fit["params"].get("interaction")
            pv = fit["pvalues"].get("interaction")
            print(f"    {sig.upper():3s} interaction={coef:+.4f}  (two-sided p={pv:.4g})")
        else:
            print(f"    {sig.upper():3s} interaction=NA ({fit})")


def main() -> int:
    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    # keep an untouched copy of both labels; relabel in place per run
    for r in ds.steps:
        r["_y_optimal_orig"] = r.get("y_optimal")
    for lab in LABELS:
        field = "_y_optimal_orig" if lab == "optimal_only" else "y_legal_or_optimal"
        run_label(ds.steps, field)
    return 0


if __name__ == "__main__":
    sys.exit(main())
