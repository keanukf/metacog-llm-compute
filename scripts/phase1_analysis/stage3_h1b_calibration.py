#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 3: H1b (calibration).

Per domain: fit a logistic TLE->probability calibrator on that domain's HOLDOUT steps (pooled
across runs and compute stages -- thesis Ch.5 §5.9, deliberately not stage-conditional like H3),
then evaluate the confirmatory DeltaBrier = Brier(TLE-mapped) - Brier(VC/100) on that domain's
NON-HOLDOUT steps via cluster_bootstrap, decided on the CI upper bound being fully below 0
(thesis Table 5.2: "Bound of DeltaBrier (TLE-VC) < 0, per domain"). Family D = the two domains,
Holm-corrected.

Note: src.analysis.inference.delta_brier_after_mapping computes only Brier(TLE-mapped) despite
its name -- not a difference. The actual DeltaBrier vs. VC/100 is computed here directly (pairwise
VC-missing exclusion, per thesis §5.2.2/§2.5 of the stats plan: a step needs both TLE and VC
present to enter this specific comparison).

Usage:
  python scripts/phase1_analysis/stage3_h1b_calibration.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage3/h1b_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compute_brier, fit_tle_calibrator, vc_to_prob  # noqa: E402
from src.analysis.inference import cluster_bootstrap, holm, one_sided_bootstrap_pvalue  # noqa: E402
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402

DOMAINS = ("tower_of_hanoi", "textworld")


def _brier_delta_stat_fn(rows: list[dict[str, Any]], calibrator: Any) -> float:
    ys: list[int] = []
    ps_tle: list[float] = []
    ps_vc: list[float] = []
    for r in rows:
        y = r.get("y_optimal")
        tle = r.get("tle_mean_entropy")
        vc = r.get("vc")
        if y is None or tle is None or vc is None:  # pairwise VC-missing exclusion
            continue
        ys.append(int(y))
        ps_tle.append(calibrator.predict_proba(float(tle)))
        ps_vc.append(vc_to_prob(float(vc)))
    if not ys:
        return float("nan")
    return compute_brier(ps_tle, ys) - compute_brier(ps_vc, ys)


def run_h1b(steps: list[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = {}
    pvalues: list[float] = []
    for dom in DOMAINS:
        dom_steps = [r for r in steps if str(r.get("domain")) == dom]
        holdout_steps = [r for r in dom_steps if bool(r.get("holdout"))]
        non_holdout_steps = [r for r in dom_steps if not bool(r.get("holdout"))]

        calibrator = fit_tle_calibrator(holdout_steps)
        if isinstance(calibrator, dict):  # fit failed -- record and skip this domain's decision
            by_domain[dom] = {
                "calibrator_converged": False,
                "calibrator_note": calibrator.get("note"),
                "point": None,
                "ci_low": None,
                "ci_high": None,
                "decision_holds": False,
                "one_sided_pvalue": 1.0,
            }
            pvalues.append(1.0)
            continue

        boot = cluster_bootstrap(
            non_holdout_steps,
            lambda rs: _brier_delta_stat_fn(rs, calibrator),
            n_boot=n_boot,
            seed=seed,
        )
        holds = boot["ci_high"] is not None and boot["ci_high"] < 0
        # one-sided test is "DeltaBrier < 0"; one_sided_bootstrap_pvalue tests ">", so negate.
        p = one_sided_bootstrap_pvalue([-r for r in boot["reps"]], null_value=0.0)
        by_domain[dom] = {k: v for k, v in boot.items() if k != "reps"}
        by_domain[dom]["calibrator_converged"] = True
        by_domain[dom]["calibrator_intercept"] = calibrator.intercept
        by_domain[dom]["calibrator_slope"] = calibrator.slope
        by_domain[dom]["n_holdout_steps"] = len(holdout_steps)
        by_domain[dom]["decision_holds"] = holds
        by_domain[dom]["one_sided_pvalue"] = p
        pvalues.append(p)

    holm_result = holm(pvalues, family="D")
    for i, dom in enumerate(DOMAINS):
        by_domain[dom]["holm"] = holm_result[i]

    return {"family": "D", "by_domain": by_domain}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json")
    parser.add_argument("--output", default="data/results/phase1_analysis/stage3/h1b_calibration.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_path.exists():
        print(f"Stage 3 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.", file=sys.stderr)
        return 1

    ds = load_canonical_dataset_from_manifest(manifest_path)
    result = run_h1b(ds.steps, n_boot=args.n_boot, seed=args.seed)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Stage 3 OK -- H1b written to {out_path}")
    for dom in DOMAINS:
        d = result["by_domain"][dom]
        if not d.get("calibrator_converged"):
            print(f"  {dom}: calibrator did not converge ({d.get('calibrator_note')})")
            continue
        print(
            f"  {dom}: DeltaBrier={d['point']:.4f} CI=[{d['ci_low']:.4f}, {d['ci_high']:.4f}] "
            f"holds={d['decision_holds']} holm_adjusted={d['holm']['adjusted']:.4f} "
            f"(n_holdout_steps={d['n_holdout_steps']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
