#!/usr/bin/env python3
"""
Analyze a ToH state-diversity/feasibility probe run: per-stage success rate,
TLE-AUROC vs. optimal (Hanley-McNeil SE + normal-approx CI), and a peg-source
bias breakdown (how often each stage sources a move from each peg).

Pure local analysis over already-collected episode JSONs -- no GPU/model call,
fully idempotent. Reproduces the numbers computed ad hoc during the 2026-07-18
ToH investigation (docs/consistency_log.md) as a committed, reusable script.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _auroc_with_ci(tle_vals: list[float], labels: list[int], *, z: float = 1.645) -> dict[str, Any]:
    """AUROC (probability a positive step has LOWER entropy than a negative step,
    since lower entropy = higher confidence) with Hanley-McNeil SE and a
    normal-approximation CI (z=1.645 -> ~90%, matching the thesis's one-sided
    alpha=.05 / 90%-CI convention, Section 5.8)."""
    pos = [t for t, label in zip(tle_vals, labels) if label == 1]
    neg = [t for t, label in zip(tle_vals, labels) if label == 0]
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return {
            "auroc": None,
            "n_pos": n1,
            "n_neg": n2,
            "se": None,
            "ci_low": None,
            "ci_high": None,
        }
    wins = 0.0
    for tp in pos:
        for tn in neg:
            if tp < tn:
                wins += 1
            elif tp == tn:
                wins += 0.5
    auc = wins / (n1 * n2)
    q1 = auc / (2 - auc)
    q2 = (2 * auc * auc) / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2)
    se = math.sqrt(max(var, 0.0))
    return {
        "auroc": auc,
        "n_pos": n1,
        "n_neg": n2,
        "se": se,
        "ci_low": max(0.0, auc - z * se),
        "ci_high": min(1.0, auc + z * se),
    }


def _load_episode_steps(path: Path) -> list[dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    step_correctness = d.get("step_correctness") or []
    tle_per_step = d.get("tle_per_step") or []
    rows = []
    for j, s in enumerate(step_correctness):
        t = tle_per_step[j] if j < len(tle_per_step) else None
        tle_val = t.get("mean_entropy") if isinstance(t, dict) else None
        action = s.get("action_parsed")
        source_peg = action[0] if isinstance(action, list) and action else None
        rows.append(
            {
                "correctness": s.get("correctness"),
                "tle": tle_val,
                "source_peg": source_peg,
                "task_success": bool(d.get("task_success")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a ToH diversity/feasibility probe run.")
    parser.add_argument(
        "--input-dir",
        default="data/results/gate_d_calibration/toh_state_diversity_probe",
        help="Directory containing per-episode JSON files (e.g. toh_diversity_<STAGE>_<i>.json)",
    )
    parser.add_argument(
        "--pattern", default="toh_diversity_*.json", help="Glob for episode files in --input-dir"
    )
    args = parser.parse_args()

    in_dir_arg = Path(args.input_dir)
    in_dir = in_dir_arg if in_dir_arg.is_absolute() else REPO_ROOT / in_dir_arg
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No episode files matching {args.pattern!r} in {in_dir}")

    by_stage: dict[str, list[Path]] = {}
    for f in files:
        parts = f.stem.split("_")
        stage = parts[2] if len(parts) >= 3 else "UNKNOWN"
        by_stage.setdefault(stage, []).append(f)

    print(f"Input: {in_dir} ({len(files)} episodes across {len(by_stage)} stage(s))\n")

    report: dict[str, Any] = {}
    for stage in sorted(by_stage):
        ep_files = by_stage[stage]
        all_rows: list[dict[str, Any]] = []
        n_episodes = len(ep_files)
        n_success = 0
        for f in ep_files:
            rows = _load_episode_steps(f)
            all_rows.extend(rows)
            if rows and rows[0]["task_success"]:
                n_success += 1

        tle_vals = [r["tle"] for r in all_rows if r["tle"] is not None]
        labels = [
            1 if r["correctness"] == "optimal" else 0 for r in all_rows if r["tle"] is not None
        ]
        auroc_stats = _auroc_with_ci(tle_vals, labels)

        source_counts = Counter(r["source_peg"] for r in all_rows if r["source_peg"])
        total_moves = sum(source_counts.values())

        print(f"=== Stage {stage} ===")
        print(
            f"  Episodes: {n_episodes}, success: {n_success}/{n_episodes} "
            f"({n_success / n_episodes:.1%})"
        )
        if auroc_stats["auroc"] is not None:
            print(
                f"  TLE-AUROC vs optimal: {auroc_stats['auroc']:.3f} "
                f"(n_optimal={auroc_stats['n_pos']}, n_other={auroc_stats['n_neg']}, "
                f"SE={auroc_stats['se']:.3f}, "
                f"~90% CI=[{auroc_stats['ci_low']:.3f}, {auroc_stats['ci_high']:.3f}])"
            )
        else:
            print("  TLE-AUROC: not computable (no TLE values or single-class labels)")
        print(f"  Move source-peg distribution ({total_moves} moves): {dict(source_counts)}")
        print()

        report[stage] = {
            "n_episodes": n_episodes,
            "n_success": n_success,
            "success_rate": n_success / n_episodes if n_episodes else None,
            "auroc": auroc_stats,
            "source_peg_counts": dict(source_counts),
        }

    out_path = in_dir / "diversity_probe_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
