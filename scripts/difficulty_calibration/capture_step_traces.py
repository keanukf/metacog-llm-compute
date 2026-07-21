#!/usr/bin/env python3
"""
Gate D trace probe: capture a handful of episodes with full per-step traces
(complete prompt/response, including the reasoning text) for manual debugging.

Unlike the other Gate D dev scripts, this deliberately turns on
save_step_traces + episode_id + trace_output_dir, so trace_{episode_id}.jsonl
files are written under <output-dir>/traces/ with the full LM call detail
(prompt, response, tokens, temperature) per step. Cheap for a handful of
episodes; not meant for large sweeps (verbose, no aggregate stripping).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.difficulty_calibration.sweep_textworld_difficulty import (
    _create_model,
    _generate_combo_games,
    _instance_seed,
    _load_merged_config,
)


def _run_textworld_stage(
    *,
    stage: str,
    num_instances: int,
    obs_ceiling: int,
    seed: int,
    config: dict[str, Any],
    trace_dir: Path,
    out_dir: Path,
) -> None:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    work = Path(tempfile.mkdtemp(prefix="trace_probe_tw_"))
    combo_seed = _instance_seed(seed, 1)
    games = _generate_combo_games(
        base_dir=work / "r3_i1_take-only",
        num_instances=num_instances,
        seed=combo_seed,
        num_rooms=3,
        num_ingredients=1,
        cut=False,
        cook=False,
        open_=False,
    )
    step_cfg = resolve_step_fn_kwargs(config, "textworld")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    model = _create_model(config, True)
    step_fn = get_step_fn(stage, **step_cfg)
    try:
        for i, gf in enumerate(games):
            ep_id = f"trace_tw_{stage}_{i}"
            env = TextWorldEnv(game_file=str(gf), max_steps=obs_ceiling)
            result = run_episode(
                env,
                model,
                stage,
                step_fn=step_fn,
                max_steps=obs_ceiling,
                save_step_traces=True,
                episode_id=ep_id,
                trace_output_dir=str(trace_dir),
                **history_cfg,
            )
            (out_dir / f"{ep_id}.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"{ep_id}: task_success={result.get('task_success')} "
                f"steps={result.get('episode_length_steps')}",
                flush=True,
            )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


def _run_toh_stage(
    *,
    stage: str,
    num_instances: int,
    num_disks: int,
    seed: int,
    config: dict[str, Any],
    trace_dir: Path,
    out_dir: Path,
) -> None:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    instances = generate_instances(
        num_instances, seed=seed, num_disks_range=(num_disks, num_disks), partial_start_range=(0, 0)
    )
    step_cfg = resolve_step_fn_kwargs(config, "tower_of_hanoi")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    model = _create_model(config, True)
    step_fn = get_step_fn(stage, **step_cfg)
    try:
        for i, inst in enumerate(instances):
            ep_id = f"trace_toh_{stage}_{i}"
            max_steps = int(inst.get("max_steps", 50))
            env = TowerOfHanoiEnv(task=inst, max_steps=max_steps)
            result = run_episode(
                env,
                model,
                stage,
                step_fn=step_fn,
                max_steps=max_steps,
                save_step_traces=True,
                episode_id=ep_id,
                trace_output_dir=str(trace_dir),
                **history_cfg,
            )
            (out_dir / f"{ep_id}.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"{ep_id}: task_success={result.get('task_success')} "
                f"steps={result.get('episode_length_steps')}",
                flush=True,
            )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D trace probe (full token-chain capture).")
    parser.add_argument("--tw-config", default="configs/dev/gate_d_diagnostic.yaml")
    parser.add_argument("--toh-config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--tw-stages", default="C1,C2", help="Comma-separated TextWorld stages")
    parser.add_argument("--toh-stages", default="C0", help="Comma-separated ToH stages")
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--tw-obs-ceiling", type=int, default=45)
    parser.add_argument("--toh-num-disks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/results/gate_d_calibration/trace_probe")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    tw_stages = [s.strip() for s in args.tw_stages.split(",") if s.strip()]
    toh_stages = [s.strip() for s in args.toh_stages.split(",") if s.strip()]

    if tw_stages:
        tw_config = _load_merged_config(REPO_ROOT / args.tw_config)
        for stage in tw_stages:
            _run_textworld_stage(
                stage=stage,
                num_instances=int(args.num_instances),
                obs_ceiling=int(args.tw_obs_ceiling),
                seed=int(args.seed),
                config=tw_config,
                trace_dir=trace_dir,
                out_dir=out_dir,
            )

    if toh_stages:
        toh_config = _load_merged_config(REPO_ROOT / args.toh_config)
        for stage in toh_stages:
            _run_toh_stage(
                stage=stage,
                num_instances=int(args.num_instances),
                num_disks=int(args.toh_num_disks),
                seed=int(args.seed),
                config=toh_config,
                trace_dir=trace_dir,
                out_dir=out_dir,
            )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
