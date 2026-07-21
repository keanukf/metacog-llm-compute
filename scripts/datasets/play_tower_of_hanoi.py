#!/usr/bin/env python3
"""
Interactive terminal play for Tower of Hanoi — sanity-check the environment without a model.

Uses TowerOfHanoiEnv from src.environments.tower_of_hanoi; does not modify that module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Tower of Hanoi interactively in the terminal (manual sanity check)."
    )
    parser.add_argument(
        "--num-disks",
        type=int,
        default=3,
        help="Number of disks (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for instance generation (default: 42)",
    )
    parser.add_argument(
        "--partial-moves",
        type=int,
        default=0,
        help="Apply this many moves along an optimal path from the classic start (0 = all disks on A).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode step cap (default: from generated task, usually len(optimal)×3)",
    )
    args = parser.parse_args()

    if args.num_disks < 1:
        print("--num-disks must be >= 1", file=sys.stderr)
        sys.exit(2)

    repo = _repo_root()
    sys.path.insert(0, str(repo))

    from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances

    p_lo = max(0, args.partial_moves)
    p_hi = max(0, args.partial_moves)
    instances = generate_instances(
        1,
        args.seed,
        num_disks_range=(args.num_disks, args.num_disks),
        partial_start_range=(p_lo, p_hi),
    )
    task = instances[0]
    max_steps = int(args.max_steps) if args.max_steps is not None else int(task["max_steps"])
    env = TowerOfHanoiEnv(task=task, max_steps=max_steps)

    print(
        f"[instance] id={task['id']} disks={task['num_disks']} "
        f"partial_start_moves={task.get('partial_start_moves', 0)} max_steps={max_steps}"
    )
    print(env.reset())
    print("Commands: type a move (e.g. A->C or 'move disk from A to C'), or quit / exit.")

    while not env.done:
        try:
            cmd = input("> ").strip()
        except EOFError:
            print("\nExiting.")
            return
        if cmd.lower() in {"quit", "exit", ":q"}:
            print("Exiting.")
            return
        if not cmd:
            continue
        env.step(cmd)
        print(env.observation)
        if env.done:
            if env.task_success:
                print("SOLVED — all disks on peg C.")
            else:
                print("Episode ended — max steps reached without solving.")
            return


if __name__ == "__main__":
    main()
