#!/usr/bin/env python3
"""
Gate D ToH feasibility diagnostic: C0/C1/C2 on N-disk instances, no signal analysis.

Companion to run_gate_d_feasibility.py (TextWorld); sweep_toh_difficulty.py only covers
C0, so this fills the C1/C2 gap for Tower of Hanoi. Reports per-stage success rate only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_textworld_difficulty import _create_model, _load_merged_config


def _run_stage(
    *,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    use_real: bool,
    stage: str,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    model = _create_model(config, use_real)
    step_cfg = resolve_step_fn_kwargs(config, "tower_of_hanoi")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    step_fn = get_step_fn(stage, **step_cfg)
    episodes: list[dict[str, Any]] = []
    try:
        for i, inst in enumerate(instances):
            max_steps = int(inst.get("max_steps", 50))
            env = TowerOfHanoiEnv(task=inst, max_steps=max_steps)
            result = run_episode(
                env, model, stage, step_fn=step_fn, max_steps=max_steps, **history_cfg
            )
            episodes.append(
                {
                    "instance_index": i,
                    "task_success": bool(result.get("task_success")),
                    "episode_length_steps": int(result.get("episode_length_steps", 0)),
                    "max_steps": max_steps,
                }
            )
            print(
                f"  {stage} inst={i} success={episodes[-1]['task_success']} "
                f"steps={episodes[-1]['episode_length_steps']}/{max_steps}",
                flush=True,
            )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()

    n_succ = sum(1 for e in episodes if e["task_success"])
    return {
        "n": len(episodes),
        "successes": n_succ,
        "success_rate": n_succ / len(episodes) if episodes else 0.0,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D ToH feasibility diagnostic (C0/C1/C2).")
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--num-disks", type=int, default=3)
    parser.add_argument("--partial-start-lo", type=int, default=0)
    parser.add_argument("--partial-start-hi", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/toh_feasibility",
    )
    args = parser.parse_args()

    from src.environments.tower_of_hanoi import generate_instances

    config = _load_merged_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = generate_instances(
        int(args.num_instances),
        seed=int(args.seed),
        num_disks_range=(int(args.num_disks), int(args.num_disks)),
        partial_start_range=(int(args.partial_start_lo), int(args.partial_start_hi)),
    )
    print(f"Generated {len(instances)} {args.num_disks}-disk ToH instances", flush=True)

    by_stage: dict[str, Any] = {}
    for stage in ["C0", "C1", "C2"]:
        by_stage[stage] = _run_stage(
            instances=instances, config=config, use_real=bool(args.real), stage=stage
        )
        row = by_stage[stage]
        print(f"=== {stage}: {row['successes']}/{row['n']} = {row['success_rate']:.1%} ===")

    report = {
        "seed": int(args.seed),
        "num_disks": int(args.num_disks),
        "num_instances": int(args.num_instances),
        "partial_start_range": [int(args.partial_start_lo), int(args.partial_start_hi)],
        "config": str(args.config),
        "by_stage": by_stage,
    }
    out_path = out_dir / "toh_feasibility_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
