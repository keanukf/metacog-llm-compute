#!/usr/bin/env python3
"""
Phase 1 — Calibration: run domains x instances x compute_stages x runs.
Supports --resume via checkpoint_dir; skips already completed episodes.
Usage: python scripts/run_phase1.py --config configs/experiment_core.yaml --checkpoint-dir data/results/phase1 [--resume]
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
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from src.utils.checkpointing import list_completed_episodes

    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
    phase1 = config.get("phase1", {})
    domains = phase1.get("domains", ["textworld", "delayed_cue"])
    instances_per_domain = phase1.get("instances_per_domain", 50)
    stages = ["C0", "C1", "C2"]
    runs = phase1.get("runs_per_condition", 5)
    total = len(domains) * instances_per_domain * len(stages) * runs
    print(f"Phase 1: {len(domains)} domains x {instances_per_domain} instances x {len(stages)} stages x {runs} runs = {total} episodes")
    print(f"Completed so far: {len(completed)}. Resume={args.resume}.")
    # Stub: no actual episode loop; real impl would iterate and call base_agent + checkpointing
    print("Stub: episode loop not run. Implement with base_agent.run_episode and checkpointing.save_episode_checkpoint.")


if __name__ == "__main__":
    main()
