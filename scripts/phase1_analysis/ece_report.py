#!/usr/bin/env python3
"""Expected Calibration Error for every signal form the thesis compares.

Section 5.8 defines ECE and names it the calibration component that the Murphy
decomposition isolates, but Chapter 6 reports only the Brier score. This closes
that gap. ECE is reported for the three forms the calibration analyses use:
TLE mapped by the holdout-fitted calibrator, VC as the model emits it, and VC
given the same one-parameter map, which is the symmetric comparison of 6.3.

Evaluated on the non-holdout steps, matching the Brier comparison exactly.

Read-only. Deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.calibration import compute_ece, fit_tle_calibrator, vc_to_prob  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

MANIFEST = ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
OUT = ROOT / "data/results/phase1_analysis/ece_report.json"
N_BINS = 10


def _fit_vc_calibrator(steps):
    """One-parameter logistic on VC, the symmetric counterpart of the TLE map."""
    import statsmodels.api as sm

    x = [[1.0, vc_to_prob(float(r["vc"]))] for r in steps]
    y = [1.0 if r.get("correctness") == "optimal" else 0.0 for r in steps]
    res = sm.Logit(y, x).fit(disp=0)
    b0, b1 = float(res.params[0]), float(res.params[1])
    import math
    return lambda v: 1.0 / (1.0 + math.exp(-(b0 + b1 * v)))


def main() -> int:
    ds = load_canonical_dataset_from_manifest(MANIFEST)
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)

    out: dict[str, dict[str, float]] = {}
    for dom in sorted({r["domain"] for r in ds.steps}):
        dsteps = [r for r in ds.steps if r["domain"] == dom
                  and (r.get("tle") or {}).get("mean_entropy") is not None
                  and r.get("vc") is not None]
        hold = [r for r in dsteps if bool(r.get("holdout"))]
        evalset = [r for r in dsteps if not bool(r.get("holdout"))]

        tle_cal = fit_tle_calibrator(hold)
        vc_cal = _fit_vc_calibrator(hold)
        y = [1.0 if r.get("correctness") == "optimal" else 0.0 for r in evalset]

        out[dom] = {
            "n_eval_steps": len(evalset),
            "ece_tle_mapped": compute_ece(
                [tle_cal.predict_proba(float(r["tle"]["mean_entropy"])) for r in evalset],
                y, N_BINS),
            "ece_vc_as_emitted": compute_ece(
                [vc_to_prob(float(r["vc"])) for r in evalset], y, N_BINS),
            "ece_vc_mapped": compute_ece(
                [vc_cal(vc_to_prob(float(r["vc"]))) for r in evalset], y, N_BINS),
            "base_rate": sum(y) / len(y),
        }
        d = out[dom]
        print(f"{dom:<16} n={d['n_eval_steps']:>6}  base={d['base_rate']:.3f}  "
              f"TLE-mapped={d['ece_tle_mapped']:.3f}  "
              f"VC-emitted={d['ece_vc_as_emitted']:.3f}  "
              f"VC-mapped={d['ece_vc_mapped']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
