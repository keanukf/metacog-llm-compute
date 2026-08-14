#!/usr/bin/env python3
"""Stage-stratified sensitivity re-analysis for H1a / H4 (compute-stage confound).

The confirmatory H1a/H4 discrimination statistics pool all three compute stages. Because both
raw TLE and step correctness are near-deterministically separated by stage, the pooled AUROC is
largely a stage-identity classifier (a Simpson-type pooling confound). This script re-estimates
H1a and H4 within compute stage -- the estimand consistent with the allocator's own
stage-conditional ECDF normalisation (thesis Ch.5 §5.3/§5.4) -- using the SAME cluster-bootstrap
harness, and reports a stage-only AUROC floor as a manipulation check. It does not modify the
preregistered pooled analysis; it is an added, disclosed sensitivity analysis.

Run from repo root:
  python scripts/phase1_analysis/stage_stratified_sensitivity.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compute_auroc  # noqa: E402
from src.analysis.inference import (  # noqa: E402
    cluster_bootstrap,
    cluster_bootstrap_stratified,
    delta_auroc,
    holm,
    one_sided_bootstrap_pvalue,
)
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

DOMS = ("tower_of_hanoi", "textworld")
STG = ("C0", "C1", "C2")
NB = 5000
SEED = 20260703


def _paired_n(rows, s):
    return sum(
        1
        for r in rows
        if str(r.get("compute_stage")) == s
        and r.get("y_optimal") is not None
        and r.get("tle_mean_entropy") is not None
        and r.get("vc") is not None
    )


def strat_dauroc(rows):
    """N-weighted mean over stages of within-stage delta_auroc(TLE, VC)."""
    num, den = 0.0, 0
    for s in STG:
        n = _paired_n(rows, s)
        if n < 2:
            continue
        d = delta_auroc([r for r in rows if str(r.get("compute_stage")) == s])
        if math.isfinite(d):
            num += d * n
            den += n
    return num / den if den else float("nan")


def stage_only_auroc(rows):
    """AUROC using each step's stage-mean correctness as the score (manipulation-check floor)."""
    m = {}
    for s in STG:
        ys = [
            int(r["y_optimal"])
            for r in rows
            if str(r.get("compute_stage")) == s
            and r.get("y_optimal") is not None
            and r.get("tle_mean_entropy") is not None
            and r.get("vc") is not None
        ]
        m[s] = sum(ys) / len(ys) if ys else 0.0
    sc, lb = [], []
    for r in rows:
        if (
            r.get("y_optimal") is None
            or r.get("tle_mean_entropy") is None
            or r.get("vc") is None
        ):
            continue
        sc.append(m[str(r.get("compute_stage"))])
        lb.append(int(r["y_optimal"]))
    return compute_auroc(sc, lb) if len(set(lb)) > 1 else float("nan")


def h4(fn):
    def _stat(rows):
        toh = [r for r in rows if str(r.get("domain")) == "tower_of_hanoi"]
        tw = [r for r in rows if str(r.get("domain")) == "textworld"]
        return fn(toh) - fn(tw)

    return _stat


def main():
    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    steps = ds.steps

    print("H1a  (delta_auroc = AUROC(TLE) - AUROC(VC); decision ci_low > 0; Holm family A)")
    pv_str = []
    for dom in DOMS:
        dr = [r for r in steps if str(r.get("domain")) == dom]
        bp = cluster_bootstrap(dr, delta_auroc, n_boot=NB, seed=SEED)
        bs = cluster_bootstrap(dr, strat_dauroc, n_boot=NB, seed=SEED)
        bf = cluster_bootstrap(dr, stage_only_auroc, n_boot=NB, seed=SEED)
        pv_str.append(one_sided_bootstrap_pvalue(bs["reps"], null_value=0.0))
        print(f"\n  {dom}")
        print(
            f"    POOLED      dAUROC = {bp['point']:+.4f}  [{bp['ci_low']:+.4f}, {bp['ci_high']:+.4f}]"
            f"  holds={bp['ci_low'] is not None and bp['ci_low'] > 0}"
        )
        print(
            f"    STRATIFIED  dAUROC = {bs['point']:+.4f}  [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]"
            f"  holds={bs['ci_low'] is not None and bs['ci_low'] > 0}"
        )
        print(
            f"    stage-only  AUROC  = {bf['point']:.4f}  [{bf['ci_low']:.4f}, {bf['ci_high']:.4f}]  (floor)"
        )
    hs = holm(pv_str, family="A")
    for i, dom in enumerate(DOMS):
        print(f"    Holm(strat) {dom}: p_raw={pv_str[i]:.4f} adj={hs[i]['adjusted']:.4f}")

    print("\nH4  DiD (ToH - TextWorld); stratified resampling within domain; decision ci_low > 0")
    for name, fn in (("POOLED", delta_auroc), ("STRATIFIED", strat_dauroc)):
        b = cluster_bootstrap_stratified(steps, h4(fn), n_boot=NB, seed=SEED)
        p = one_sided_bootstrap_pvalue(b["reps"], null_value=0.0)
        print(
            f"  {name:10} DiD = {b['point']:+.4f}  [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"
            f"  holds={b['ci_low'] is not None and b['ci_low'] > 0}  p={p:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
