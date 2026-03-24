#!/usr/bin/env python3
"""
Phase 1 — Calibration: run domains x instances x compute_stages x runs.
Supports --resume via checkpoint_dir; skips already completed episodes.
Usage: python scripts/run_phase1.py --config configs/experiment_core.yaml [--resume] [--real]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--checkpoint-dir", default="data/results/phase1")
    parser.add_argument("--resume", action="store_true", help="Skip completed episodes")
    parser.add_argument("--real", action="store_true", help="Use real model (vLLM/HF) when available")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from src.utils.checkpointing import list_completed_episodes, save_episode_checkpoint
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.utils.experiment_env import create_experiment_model, make_experiment_env

    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
    phase1 = config.get("phase1", {})
    domains = phase1.get("domains", ["textworld", "tower_of_hanoi"])
    instances_per_domain = phase1.get("instances_per_domain", 50)
    stages = ["C0", "C1", "C2"]
    runs = phase1.get("runs_per_condition", 5)
    max_steps = config.get("episode", {}).get("max_steps_per_episode", 20)
    total = len(domains) * instances_per_domain * len(stages) * runs
    print(f"Phase 1: {len(domains)} domains x {instances_per_domain} instances x {len(stages)} stages x {runs} runs = {total} episodes")
    print(f"Completed so far: {len(completed)}. Resume={args.resume}. Real model={args.real}.")

    model = create_experiment_model(config, args.real)
    done_count = 0
    for domain in domains:
        for inst in range(instances_per_domain):
            for stage in stages:
                for run in range(runs):
                    ep_id = f"ep_{domain}_{inst}_{stage}_{run}"
                    if ep_id in completed:
                        continue
                    env = make_experiment_env(domain, inst, config, max_steps, REPO_ROOT)
                    step_fn = get_step_fn(stage)
                    result = run_episode(env, model, stage, step_fn=step_fn, max_steps=max_steps)
                    data = {
                        "episode_id": ep_id,
                        "domain": domain,
                        "instance": inst,
                        "compute_stage": stage,
                        "run": run,
                        "task_success": result["task_success"],
                        "steps": result["steps"],
                        "lm_calls": result["lm_calls"],
                        "tokens": result["tokens"],
                        "wall_clock_time": result["wall_clock_time"],
                        "tle_per_step": result.get("tle_per_step"),
                        "vc_per_step": result.get("vc_per_step"),
                    }
                    if result.get("step_correctness") is not None:
                        data["step_correctness"] = result["step_correctness"]
                    save_episode_checkpoint(checkpoint_dir, ep_id, data)
                    done_count += 1
                    if done_count % 50 == 0:
                        print(f"  Completed {done_count} new episodes (total in dir: {len(completed) + done_count})")
    print(f"Phase 1 done. New episodes: {done_count}. Total checkpoints: {len(list_completed_episodes(checkpoint_dir))}.")


if __name__ == "__main__":
    main()
