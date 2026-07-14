#!/usr/bin/env python3
"""TLE distribution screen for §5.4 ECDF/threshold viability (Phase 0 diagnostic)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.datasets import load_run_dataset

NEAR_ZERO_THRESHOLD = 1e-3


def _round_key(v: float, decimals: int = 12) -> float:
    return round(float(v), decimals)


def diagnose_run(run_dir: Path, *, near_zero: float = NEAR_ZERO_THRESHOLD) -> dict:
    ds = load_run_dataset(run_dir)
    buckets: dict[str, list[float]] = defaultdict(list)
    for ep in ds.episodes:
        dom = str(ep.get("domain", "unknown"))
        stage = str(ep.get("compute_stage", "unknown"))
        for sd in ep.get("steps_detail") or []:
            if not isinstance(sd, dict):
                continue
            tle = sd.get("tle")
            if not isinstance(tle, dict):
                continue
            v = tle.get("mean_entropy")
            if not isinstance(v, (int, float)):
                continue
            key = f"{dom}|{sd.get('compute_stage', stage)}"
            buckets[key].append(float(v))
            buckets[f"{dom}|all"].append(float(v))
            buckets["all|all"].append(float(v))

    report: dict[str, dict] = {}
    for key, vals in sorted(buckets.items()):
        if not vals:
            continue
        distinct = len({_round_key(v) for v in vals})
        under = sum(1 for v in vals if v < near_zero)
        sorted_vals = sorted(vals)
        n = len(vals)
        hist_bins = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, float("inf")]
        hist: dict[str, int] = {}
        for lo, hi in zip(hist_bins[:-1], hist_bins[1:]):
            label = f"[{lo:g},{hi:g})"
            hist[label] = sum(1 for v in vals if lo <= v < hi)
        report[key] = {
            "n_steps": n,
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / n,
            "median": sorted_vals[n // 2],
            "p95": sorted_vals[int(0.95 * (n - 1))] if n > 1 else sorted_vals[0],
            "n_distinct_rounded": distinct,
            "n_under_near_zero": under,
            "pct_under_near_zero": 100.0 * under / n,
            "histogram": hist,
        }
    return {"run": str(run_dir.name), "near_zero_threshold": near_zero, "by_group": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="TLE distribution diagnostic for a Phase1 run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--near-zero", type=float, default=NEAR_ZERO_THRESHOLD)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 2

    report = diagnose_run(run_dir, near_zero=args.near_zero)
    text = json.dumps(report, indent=2)
    if args.output:
        out = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)

    # Human-readable summary lines
    for key, d in report.get("by_group", {}).items():
        if "|all" not in key:
            continue
        print(
            f"  {key}: n={d['n_steps']} distinct={d['n_distinct_rounded']} "
            f"pct<{args.near_zero:g}={d['pct_under_near_zero']:.1f}% "
            f"median={d['median']:.6g} max={d['max']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
