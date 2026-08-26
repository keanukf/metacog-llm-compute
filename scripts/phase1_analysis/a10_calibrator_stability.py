#!/usr/bin/env python3
"""A10: quantify the instability of the TLE-to-probability calibrator.

Section 8.5 states that the calibration mapping rests on a five-instance holdout
and calls the resulting variance a limitation, but reports no magnitude. Section
7.5 does the equivalent for the allocator thresholds. This script closes the gap
by refitting the calibrator on every leave-one-instance-out subset of the holdout
and reporting the spread of the fitted slope and intercept.

Read-only. Deterministic.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.calibration import fit_tle_calibrator  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,

    load_canonical_dataset_from_manifest,
)

MANIFEST = ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
OUT = ROOT / "data/results/phase1_analysis/a10_calibrator_stability.json"


def main() -> int:
    ds = load_canonical_dataset_from_manifest(MANIFEST)
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)

    out: dict[str, dict] = {}
    for dom in sorted({r["domain"] for r in ds.steps}):
        hold = [r for r in ds.steps if r["domain"] == dom and bool(r.get("holdout"))]
        insts = sorted({r["instance_key"] for r in hold})
        full = fit_tle_calibrator(hold)

        slopes, intercepts = [], []
        for drop in insts:
            sub = [r for r in hold if r["instance_key"] != drop]
            cal = fit_tle_calibrator(sub)
            slopes.append(float(cal.slope))
            intercepts.append(float(cal.intercept))

        out[dom] = {
            "n_holdout_instances": len(insts),
            "n_holdout_steps": len(hold),
            "slope_full": float(full.slope),
            "slope_loo": slopes,
            "slope_min": min(slopes),
            "slope_max": max(slopes),
            "slope_range_pct_of_full": 100 * (max(slopes) - min(slopes)) / abs(float(full.slope)),
            "slope_sd": statistics.stdev(slopes) if len(slopes) > 1 else 0.0,
            "intercept_full": float(full.intercept),
            "intercept_min": min(intercepts),
            "intercept_max": max(intercepts),
        }
        d = out[dom]
        print(f"{dom}: {d['n_holdout_instances']} holdout instances, "
              f"slope {d['slope_full']:.3f} -> LOO range "
              f"[{d['slope_min']:.3f}, {d['slope_max']:.3f}] "
              f"= {d['slope_range_pct_of_full']:.1f}% of the full-fit value")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
