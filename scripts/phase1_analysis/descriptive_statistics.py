#!/usr/bin/env python3
"""Descriptive statistics for the signals, outcome and covariates (Phase 1).

The results chapters report inferential quantities but never characterise the
variables themselves. This produces the table an examiner from a psychology
background expects before any test: distribution of both signals by domain and
compute stage, the outcome base rate, episode length, and the signal-signal and
signal-outcome associations.

Read-only. Deterministic.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

MANIFEST = ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
OUT = ROOT / "data/results/phase1_analysis/descriptive_statistics.json"


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.fmean(ra), st.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def summarise(xs):
    if not xs:
        return None
    return {
        "n": len(xs), "mean": st.fmean(xs),
        "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs), "p25": pct(xs, .25), "median": pct(xs, .50),
        "p75": pct(xs, .75), "max": max(xs),
    }


def main() -> int:
    ds = load_canonical_dataset_from_manifest(MANIFEST)
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    steps = [r for r in ds.steps if (r.get("tle") or {}).get("mean_entropy") is not None and r.get("vc") is not None]

    out: dict = {"n_steps_with_both_signals": len(steps), "by_cell": {}, "by_domain": {}}
    for dom in sorted({r["domain"] for r in steps}):
        dsteps = [r for r in steps if r["domain"] == dom]
        for stage in sorted({r["compute_stage"] for r in dsteps}):
            cell = [r for r in dsteps if r["compute_stage"] == stage]
            tle = [float(r["tle"]["mean_entropy"]) for r in cell]
            vc = [float(r["vc"]) / 100.0 for r in cell]
            y = [1 if r.get("correctness") == "optimal" else 0 for r in cell]
            out["by_cell"][f"{dom}|{stage}"] = {
                "tle": summarise(tle), "vc": summarise(vc),
                "optimal_rate": st.fmean(y),
                "spearman_tle_vc": spearman(tle, vc),
                "spearman_tle_y": spearman(tle, [float(v) for v in y]),
                "spearman_vc_y": spearman(vc, [float(v) for v in y]),
                "vc_distinct_values": len({r["vc"] for r in cell}),
            }
        lens: dict = {}
        for r in dsteps:
            lens[r["episode_id"]] = lens.get(r["episode_id"], 0) + 1
        out["by_domain"][dom] = {
            "n_episodes": len(lens),
            "episode_length": summarise(list(lens.values())),
            "vc_distinct_values_overall": len({r["vc"] for r in dsteps}),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in out["by_cell"].items():
        print(f"{k:<28} TLE M={v['tle']['mean']:.4f} SD={v['tle']['sd']:.4f} | "
              f"VC M={v['vc']['mean']:.3f} SD={v['vc']['sd']:.3f} | "
              f"optimal={v['optimal_rate']:.3f} | rho(TLE,VC)={v['spearman_tle_vc']:+.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
