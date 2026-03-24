#!/usr/bin/env python3
"""
Phase 2 — Adaptive Allocation: run domains x instances x strategies x runs.
Supports --resume via checkpoint_dir.
Usage: python scripts/run_phase2.py --config configs/experiment_core.yaml [--resume]
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


def _make_env(domain: str, instance: int, config: dict, max_steps: int):
    """Create env for domain/instance. Mirrors Phase 1 domain handling."""
    from src.environments.textworld_env import TextWorldEnv

    if domain == "textworld":
        tasks_dir = Path(config.get("paths", {}).get("tasks_dir", "data/tasks"))
        game_file = tasks_dir / f"textworld_{instance}.ulx"
        if not game_file.is_absolute():
            game_file = REPO_ROOT / game_file
        if not game_file.exists():
            game_file = None
        return TextWorldEnv(game_file=str(game_file) if game_file else None, max_steps=max_steps)
    if domain == "tower_of_hanoi":
        from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances

        cfg = config.get("tower_of_hanoi", {})
        num_disks_range = cfg.get("num_disks_range", [3, 4])
        partial_start_range = cfg.get("partial_start_range", [0, 3])
        base_seed = int(cfg.get("task_generation_seed", 42))
        seed = base_seed + instance * 10007
        task_instance = generate_instances(
            1,
            seed=seed,
            num_disks_range=(int(num_disks_range[0]), int(num_disks_range[1])),
            partial_start_range=(int(partial_start_range[0]), int(partial_start_range[1])),
        )[0]
        return TowerOfHanoiEnv(task=task_instance, max_steps=max_steps)
    return TextWorldEnv(game_file=None, max_steps=max_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--checkpoint-dir", default="data/results/phase2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from src.utils.checkpointing import list_completed_episodes

    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
    phase2 = config.get("phase2", {})
    domains = phase2.get("domains", ["textworld", "tower_of_hanoi"])
    instances_per_domain = phase2.get("instances_per_domain", 50)
    strategies = phase2.get("strategies", ["adaptive_tle", "always_c0", "always_c2", "random", "eager_style", "adaptive_vc"])
    runs = phase2.get("runs_per_condition", 5)
    total = len(domains) * instances_per_domain * len(strategies) * runs
    print(f"Phase 2: {len(domains)} domains x {instances_per_domain} instances x {len(strategies)} strategies x {runs} runs = {total} episodes")
    print(f"Completed: {len(completed)}. Resume={args.resume}.")

    print("Stub: episode loop not run. Implement with allocator.allocate + base_agent.run_episode + checkpointing.")


if __name__ == "__main__":
    main()
