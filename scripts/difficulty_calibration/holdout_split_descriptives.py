#!/usr/bin/env python3
"""Descriptive Holdout vs non-holdout table for Gate D manifests."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _summarize(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    floats = [float(v) for v in values]
    return {
        "n": len(floats),
        "mean": round(sum(floats) / len(floats), 3),
        "median": round(float(statistics.median(floats)), 3),
        "min": round(min(floats), 3),
        "max": round(max(floats), 3),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tw_rows(manifest: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    holdout, non = [], []
    for e in manifest.get("entries") or []:
        sidecar_path = e.get("sidecar_file")
        row = {
            "instance_id": int(e.get("instance_id", 0)),
            "difficulty_tier": e.get("difficulty_tier"),
            "expected_step_count": e.get("expected_step_count"),
            "num_rooms": (e.get("generation_parameters") or {}).get("num_rooms"),
        }
        if sidecar_path:
            sp = Path(sidecar_path)
            if not sp.is_absolute():
                sp = REPO_ROOT / sp
            if sp.is_file():
                sc = _load_json(sp)
                row["expected_step_count"] = sc.get(
                    "expected_step_count", row["expected_step_count"]
                )
                gp = sc.get("generation_parameters") or {}
                row["num_rooms"] = gp.get("num_rooms", row["num_rooms"])
        (holdout if e.get("holdout") else non).append(row)
    return holdout, non


def _toh_rows(manifest: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    holdout, non = [], []
    for e in manifest.get("entries") or []:
        row = {
            "instance_id": int(e.get("instance_id", 0)),
            "difficulty_tier": e.get("difficulty_tier"),
            "num_disks": e.get("num_disks"),
            "partial_start_moves": e.get("partial_start_moves"),
            "optimal_steps": e.get("optimal_steps"),
        }
        (holdout if e.get("holdout") else non).append(row)
    return holdout, non


def _table(domain: str, holdout: list[dict], non: list[dict]) -> dict[str, Any]:
    if domain == "textworld":
        fields = ["expected_step_count", "num_rooms"]
    else:
        fields = ["num_disks", "partial_start_moves", "optimal_steps"]
    tiers_h = [r.get("difficulty_tier") for r in holdout]
    tiers_n = [r.get("difficulty_tier") for r in non]
    numeric: dict[str, Any] = {}
    for f in fields:
        numeric[f] = {
            "holdout": _summarize([r[f] for r in holdout if r.get(f) is not None]),
            "non_holdout": _summarize([r[f] for r in non if r.get(f) is not None]),
        }
    return {
        "domain": domain,
        "holdout_count": len(holdout),
        "non_holdout_count": len(non),
        "difficulty_tier_counts": {
            "holdout": dict(__import__("collections").Counter(tiers_h)),
            "non_holdout": dict(__import__("collections").Counter(tiers_n)),
        },
        "numeric_summaries": numeric,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D holdout descriptives.")
    parser.add_argument(
        "--tw-manifest",
        default="data/tasks/textworld/difficulty_manifest.json",
    )
    parser.add_argument(
        "--toh-manifest",
        default="data/tasks/tower_of_hanoi/difficulty_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="data/results/gate_d_calibration/reports/holdout_descriptives.json",
    )
    args = parser.parse_args()

    tw_path = REPO_ROOT / args.tw_manifest
    toh_path = REPO_ROOT / args.toh_manifest
    tables = []
    if tw_path.is_file():
        tw_m = _load_json(tw_path)
        h, n = _tw_rows(tw_m)
        tables.append(_table("textworld", h, n))
    if toh_path.is_file():
        toh_m = _load_json(toh_path)
        h, n = _toh_rows(toh_m)
        tables.append(_table("tower_of_hanoi", h, n))

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tables": tables}, indent=2), encoding="utf-8")

    md_lines = ["# Gate D — Holdout descriptives (n=5 vs n=45)\n"]
    for t in tables:
        md_lines.append(f"## {t['domain']}\n")
        md_lines.append(f"- holdout: {t['holdout_count']}, non-holdout: {t['non_holdout_count']}\n")
        md_lines.append(f"- tiers (holdout): {t['difficulty_tier_counts']['holdout']}\n")
        md_lines.append(f"- tiers (non-holdout): {t['difficulty_tier_counts']['non_holdout']}\n")
        for field, summ in t["numeric_summaries"].items():
            md_lines.append(
                f"- **{field}**: holdout {summ['holdout']}, non-holdout {summ['non_holdout']}\n"
            )
    md_path = out_path.with_suffix(".md")
    md_path.write_text("".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_path} and {md_path}")


if __name__ == "__main__":
    main()
