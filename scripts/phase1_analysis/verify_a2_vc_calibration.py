#!/usr/bin/env python3
"""Verify deep-check finding A2: is H1b's calibration advantage an artifact of the
asymmetric calibration (TLE holdout-Platt-calibrated vs VC scored as raw vc/100)?

Reproduces the confirmatory ΔBrier = Brier(TLE-mapped) - Brier(VC/100) exactly, then
computes the FAIR comparison ΔBrier = Brier(TLE-mapped) - Brier(VC-Platt), where VC gets
the identical holdout-fitted logistic calibration TLE gets. Also reports absolute Briers
and the base-rate (constant mean-correctness) floor. Same holdout, same non-holdout
evaluation subset, same pairing (y_optimal & tle & vc all present) as the H1b driver.

Read-only, no GPU. Run from repo root:
  python scripts/phase1_analysis/verify_a2_vc_calibration.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compute_brier, fit_tle_calibrator, vc_to_prob  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

DOMS = ("tower_of_hanoi", "textworld")


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _fit_logit(xs: list[float], ys: list[int]):
    """Logistic y ~ x via statsmodels, same engine as fit_tle_calibrator."""
    import statsmodels.api as sm

    design = sm.add_constant(xs)
    res = sm.Logit(ys, design).fit(disp=0)
    return float(res.params[0]), float(res.params[1])  # intercept, slope


def _paired(steps):
    """Non-holdout steps with y_optimal, tle, vc all present (the H1b eval subset)."""
    out = []
    for r in steps:
        if bool(r.get("holdout")):
            continue
        y = r.get("y_optimal")
        tle = r.get("tle_mean_entropy")
        vc = r.get("vc")
        if y is None or tle is None or vc is None:
            continue
        out.append((int(y), float(tle), float(vc)))
    return out


def main() -> int:
    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)

    for dom in DOMS:
        dom_steps = [r for r in ds.steps if str(r.get("domain")) == dom]
        holdout = [r for r in dom_steps if bool(r.get("holdout"))]

        # TLE calibrator (exactly the H1b path)
        tle_cal = fit_tle_calibrator(holdout)

        # VC calibrator: identical logistic, fit on the same holdout, feature = vc
        hv_x = [float(r["vc"]) for r in holdout if r.get("vc") is not None and r.get("y_optimal") is not None]
        hv_y = [int(r["y_optimal"]) for r in holdout if r.get("vc") is not None and r.get("y_optimal") is not None]
        a_vc, b_vc = _fit_logit(hv_x, hv_y)

        rows = _paired(dom_steps)
        ys = [y for y, _, _ in rows]
        p_tle = [_sigmoid(tle_cal.intercept + tle_cal.slope * t) for _, t, _ in rows]
        p_vc_raw = [vc_to_prob(v) for _, _, v in rows]
        p_vc_cal = [_sigmoid(a_vc + b_vc * v) for _, _, v in rows]

        base = sum(ys) / len(ys)
        b_tle = compute_brier(p_tle, ys)
        b_vc_raw = compute_brier(p_vc_raw, ys)
        b_vc_cal = compute_brier(p_vc_cal, ys)
        b_floor = compute_brier([base] * len(ys), ys)

        print(f"\n===== {dom}  (n_eval={len(ys)}, base rate correct={base:.3f}) =====")
        print(f"  Brier(TLE-mapped)      = {b_tle:.4f}")
        print(f"  Brier(VC/100, raw)     = {b_vc_raw:.4f}")
        print(f"  Brier(VC-Platt)        = {b_vc_cal:.4f}   [VC calibrator: p=sigmoid({a_vc:+.3f}{b_vc:+.4f}*vc)]")
        print(f"  Brier(base-rate floor) = {b_floor:.4f}")
        print(f"  --")
        print(f"  ΔBrier CURRENT (TLE - VC/100)   = {b_tle - b_vc_raw:+.4f}   (thesis H1b; <0 => TLE better)")
        print(f"  ΔBrier FAIR    (TLE - VC-Platt) = {b_tle - b_vc_cal:+.4f}   (both calibrated)")
        print(f"  VC calibration gain (VC/100 - VC-Platt) = {b_vc_raw - b_vc_cal:+.4f}")
        print(f"  TLE vs floor   (TLE - floor)    = {b_tle - b_floor:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
