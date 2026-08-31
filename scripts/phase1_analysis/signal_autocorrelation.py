#!/usr/bin/env python3
"""Lag-one autocorrelation of both signals, within compute stage and pooled.

Sections 6.4 and 8.1 rest on this diagnostic: an allocator that reads a signal at
one step to gate the next needs that reading to persist across the interval it
bridges. The quantity is the Pearson correlation between the signal value at a
step and its value at the immediately following step of the same episode.

Because a Phase 1 episode runs at a single compute stage, consecutive steps
always share a stage. Pooling nevertheless mixes pairs drawn from episodes at
different stages, which adds the between-stage level difference to both variables
and inflates the correlation. Both are reported so the difference is visible.

Read-only. Deterministic.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

MANIFEST = ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
OUT = ROOT / "data/results/phase1_analysis/signal_autocorrelation.json"


def pearson(pairs):
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = st.fmean(xs), st.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else None


def consecutive_pairs(steps, value):
    """(x_t, x_{t+1}) for steps adjacent within the same episode."""
    by_ep = defaultdict(list)
    for r in steps:
        v = value(r)
        if v is not None:
            by_ep[r["episode_id"]].append((r["step_index"], v))
    out = []
    for seq in by_ep.values():
        seq.sort()
        for (i, a), (j, b) in zip(seq, seq[1:]):
            if j == i + 1:
                out.append((a, b))
    return out


def main() -> int:
    ds = load_canonical_dataset_from_manifest(MANIFEST)
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)

    tle = lambda r: (r.get("tle") or {}).get("mean_entropy")
    vc = lambda r: (float(r["vc"]) / 100.0) if r.get("vc") is not None else None

    result: dict = {}
    for dom in sorted({r["domain"] for r in ds.steps}):
        dsteps = [r for r in ds.steps if r["domain"] == dom]
        cell: dict = {}
        for stage in sorted({r["compute_stage"] for r in dsteps}):
            sub = [r for r in dsteps if r["compute_stage"] == stage]
            cell[stage] = {
                "tle": pearson(consecutive_pairs(sub, tle)),
                "vc": pearson(consecutive_pairs(sub, vc)),
                "n_pairs_tle": len(consecutive_pairs(sub, tle)),
            }
        cell["pooled"] = {
            "tle": pearson(consecutive_pairs(dsteps, tle)),
            "vc": pearson(consecutive_pairs(dsteps, vc)),
            "n_pairs_tle": len(consecutive_pairs(dsteps, tle)),
        }
        result[dom] = cell

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print(f"{'domain':<16}{'cell':<9}{'TLE':>8}{'VC':>8}{'pairs':>9}")
    for dom, cells in result.items():
        for name, v in cells.items():
            t = f"{v['tle']:.3f}" if v["tle"] is not None else "  n/a"
            c = f"{v['vc']:.3f}" if v["vc"] is not None else "  n/a"
            print(f"{dom:<16}{name:<9}{t:>8}{c:>8}{v['n_pairs_tle']:>9}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
