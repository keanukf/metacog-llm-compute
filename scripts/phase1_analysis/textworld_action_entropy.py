#!/usr/bin/env python3
"""Mean token-level entropy by action type in TextWorld.

Section 6.3 explains the TextWorld discrimination null by observing that entropy
rises on information-gathering actions, which the primary correctness label codes
as not correct because they do not reduce distance to the goal. That observation
was originally read off individual traces. This computes it over every canonical
TextWorld step, so the claim rests on the full sample rather than on examples.

The canonical dataset carries no action text, so steps are joined back to the
trace files named by the manifest.

Read-only. Deterministic.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
OUT = ROOT / "data/results/phase1_analysis/textworld_action_entropy.json"
def _median_vc(group):
    """Median verbalized confidence over the steps of one verb, or None if none carry a value."""
    vals = [float(g["vc"]) for g in group if g.get("vc") is not None]
    return st.median(vals) if vals else None


MIN_N = 50  # verbs rarer than this are pooled into "other"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    steps = []
    for entry in manifest["entries"]:
        if entry["domain"] != "textworld":
            continue
        f = Path(entry["source_dir"]) / f"trace_{entry['episode_id']}.jsonl"
        if not f.exists():
            continue
        for line in f.open():
            try:
                d = json.loads(line)
            except Exception:
                continue
            action = d.get("action_parsed")
            entropy = (d.get("tle") or {}).get("mean_entropy")
            if action is None or entropy is None:
                continue
            steps.append({
                "verb": str(action).strip().lower().split()[0],
                "entropy": float(entropy),
                "optimal": d.get("correctness") == "optimal",
                "vc": d.get("vc"),
                "stage": d.get("compute_stage"),
            })

    by_verb: dict[str, list] = defaultdict(list)
    for s in steps:
        by_verb[s["verb"]].append(s)

    rows = []
    pooled = []
    for verb, group in by_verb.items():
        if len(group) < MIN_N:
            pooled.extend(group)
            continue
        e = [g["entropy"] for g in group]
        rows.append({
            "verb": verb,
            "n": len(group),
            "mean_entropy": st.fmean(e),
            "median_entropy": st.median(e),
            "max_entropy": max(e),
            "median_vc": _median_vc(group),
            "optimal_rate": sum(g["optimal"] for g in group) / len(group),
        })
    if pooled:
        e = [g["entropy"] for g in pooled]
        rows.append({
            "verb": f"(other, <{MIN_N} steps each)", "n": len(pooled),
            "mean_entropy": st.fmean(e), "median_entropy": st.median(e),
            "max_entropy": max(e),
            "median_vc": _median_vc(pooled),
            "optimal_rate": sum(g["optimal"] for g in pooled) / len(pooled),
        })
    rows.sort(key=lambda r: -r["mean_entropy"])

    result = {
        "n_steps": len(steps),
        "min_n_per_verb": MIN_N,
        "max_entropy_any_step": max(s["entropy"] for s in steps),
        "by_verb": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print(f"{len(steps)} canonical TextWorld steps with an action and an entropy value\n")
    print(f"{'verb':<26}{'n':>7}{'M entropy':>12}{'Mdn':>9}{'max':>8}{'optimal':>10}")
    for r in rows:
        print(f"{r['verb']:<26}{r['n']:>7}{r['mean_entropy']:>12.4f}"
              f"{r['median_entropy']:>9.4f}{r['max_entropy']:>8.3f}{r['optimal_rate']:>10.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
