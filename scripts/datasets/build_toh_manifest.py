#!/usr/bin/env python3
"""Build difficulty_manifest.json for Tower of Hanoi task instances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environments.tower_of_hanoi import generate_instances


def _difficulty_tier(num_disks: int) -> str:
    if num_disks <= 3:
        return "easy"
    if num_disks == 4:
        return "medium"
    return "hard"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ToH difficulty_manifest.json with holdout split."
    )
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-count", type=int, default=5)
    parser.add_argument(
        "--holdout-policy",
        choices=["first-n", "mod-10"],
        default="first-n",
    )
    parser.add_argument(
        "--output",
        default="data/tasks/tower_of_hanoi/difficulty_manifest.json",
    )
    parser.add_argument("--num-disks-range", nargs=2, type=int, default=[3, 4])
    parser.add_argument("--partial-start-range", nargs=2, type=int, default=[0, 3])
    parser.add_argument(
        "--partial-start-mode",
        choices=["optimal_prefix", "random_scramble"],
        default="optimal_prefix",
        help="Must match what make_experiment_env() will use to reconstruct instances at "
        "runtime -- stored per-entry below precisely so the runtime reconstruction reads it "
        "from the manifest instead of silently defaulting back to 'optimal_prefix'.",
    )
    args = parser.parse_args()

    instances = generate_instances(
        int(args.num_instances),
        seed=int(args.seed),
        num_disks_range=(int(args.num_disks_range[0]), int(args.num_disks_range[1])),
        partial_start_range=(int(args.partial_start_range[0]), int(args.partial_start_range[1])),
        partial_start_mode=args.partial_start_mode,
    )
    entries: list[dict[str, Any]] = []
    for i, inst in enumerate(instances):
        entries.append(
            {
                "instance_id": i,
                "num_disks": int(inst["num_disks"]),
                "partial_start_moves": int(inst.get("partial_start_moves", 0)),
                "optimal_steps": int(inst.get("optimal_steps", 0)),
                "difficulty_tier": _difficulty_tier(int(inst["num_disks"])),
                "task_generation_seed": int(args.seed),
                "num_disks_range": [int(args.num_disks_range[0]), int(args.num_disks_range[1])],
                "partial_start_range": [
                    int(args.partial_start_range[0]),
                    int(args.partial_start_range[1]),
                ],
                "partial_start_mode": args.partial_start_mode,
            }
        )
    entries.sort(key=lambda x: int(x["instance_id"]))
    holdout_count = max(0, min(int(args.holdout_count), len(entries)))
    if args.holdout_policy == "mod-10":
        mod_candidates = [int(e["instance_id"]) for e in entries if int(e["instance_id"]) % 10 == 0]
        holdout_ids = set(mod_candidates[:holdout_count])
        if len(holdout_ids) < holdout_count:
            for e in entries:
                iid = int(e["instance_id"])
                if iid not in holdout_ids:
                    holdout_ids.add(iid)
                if len(holdout_ids) >= holdout_count:
                    break
    else:
        holdout_ids = {int(e["instance_id"]) for e in entries[:holdout_count]}
    for e in entries:
        e["holdout"] = int(e["instance_id"]) in holdout_ids

    manifest = {
        "dataset": "tower_of_hanoi",
        "num_instances": len(entries),
        "holdout_count": holdout_count,
        "non_holdout_count": len(entries) - holdout_count,
        "holdout_policy": args.holdout_policy,
        "task_generation_seed": int(args.seed),
        "entries": entries,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Holdout: {holdout_count} | Non-holdout: {len(entries) - holdout_count}")


if __name__ == "__main__":
    main()
