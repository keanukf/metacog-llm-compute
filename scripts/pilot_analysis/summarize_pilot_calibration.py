#!/usr/bin/env python3
"""Summarize TextWorld / ToH pilot episode JSONs for baseline calibration checks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _mean_tle(ep: dict) -> float | None:
    rows = ep.get("tle_per_step") or []
    vals: list[float] = []
    for t in rows:
        if isinstance(t, dict) and t.get("mean_entropy") is not None:
            vals.append(float(t["mean_entropy"]))
    return statistics.mean(vals) if vals else None


def _vc_values(ep: dict) -> list[float]:
    out: list[float] = []
    for v in ep.get("vc_per_step") or []:
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _step_correctness(ep: dict) -> list[float]:
    sd = ep.get("steps_detail") or []
    out: list[float] = []
    for row in sd:
        c = row.get("correctness")
        if c == "optimal":
            out.append(1.0)
        elif c == "legal":
            out.append(0.5)
        elif c == "illegal":
            out.append(0.0)
    return out


def summarize_dir(output_dir: Path) -> None:
    tw = sorted(output_dir.glob("ep_textworld_*.json"))
    toh = sorted(output_dir.glob("ep_tower_of_hanoi_*.json"))
    print(f"Directory: {output_dir}")
    print(f"TextWorld episodes: {len(tw)}")
    print(f"TowerOfHanoi episodes: {len(toh)}")

    def _by_stage(eps: list[Path]) -> None:
        by: dict[str, list[dict]] = {}
        for p in eps:
            d = _load_json(p)
            st = str(d.get("compute_stage", "?"))
            by.setdefault(st, []).append(d)
        for st in sorted(by.keys()):
            group = by[st]
            succ = sum(1 for e in group if e.get("task_success"))
            print(f"  {st}: n={len(group)} success={succ}/{len(group)}")
            mtes = [_mean_tle(e) for e in group]
            mtes = [x for x in mtes if x is not None]
            if mtes:
                print(
                    f"    mean episode mean-TLE: {statistics.mean(mtes):.4f} (std {statistics.pstdev(mtes):.4f})"
                )
            all_vc: list[float] = []
            for e in group:
                all_vc.extend(_vc_values(e))
            if all_vc:
                print(
                    f"    VC (non-null steps): n={len(all_vc)} "
                    f"min={min(all_vc):.1f} max={max(all_vc):.1f} mean={statistics.mean(all_vc):.1f}"
                )

    if tw:
        print("TextWorld by stage:")
        _by_stage(tw)
    if toh:
        print("TowerOfHanoi by stage:")
        _by_stage(toh)

    # Optional: ECE proxy (VC vs step correctness) when both exist
    try:
        from src.analysis.calibration import compute_ece
    except ImportError:
        compute_ece = None  # type: ignore[assignment]
    if compute_ece is not None:
        preds: list[float] = []
        corr: list[float] = []
        for p in tw + toh:
            ep = _load_json(p)
            vcs = ep.get("vc_per_step") or []
            sds = ep.get("steps_detail") or []
            n = min(len(vcs), len(sds))
            for i in range(n):
                v = vcs[i]
                c = sds[i].get("correctness")
                if v is None or c is None:
                    continue
                preds.append(float(v) / 100.0)
                if c == "optimal":
                    corr.append(1.0)
                elif c == "legal":
                    corr.append(0.0)
                elif c == "illegal":
                    corr.append(0.0)
                else:
                    continue
        if preds:
            ece = compute_ece(preds, corr, n_bins=10)
            print(
                f"ECE (VC/100 vs optimal=1 else 0, coarse): {ece:.4f} over {len(preds)} step pairs"
            )
        else:
            print("ECE: skipped (no vc_per_step + steps_detail pairs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", type=Path, help="pilot output directory containing ep_*.json")
    args = ap.parse_args()
    summarize_dir(args.output_dir.resolve())


if __name__ == "__main__":
    main()
