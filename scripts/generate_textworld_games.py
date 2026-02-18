#!/usr/bin/env python3
"""
Generate small TextWorld games for pilot/phase1. Requires textworld and tw-make on PATH.
Usage: python scripts/generate_textworld_games.py [--output-dir data/tasks] [--num-games 3] [--rooms 5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mini TextWorld games via tw-make")
    parser.add_argument("--output-dir", default="data/tasks", help="Directory for game files")
    parser.add_argument("--num-games", type=int, default=3, help="Number of games to generate")
    parser.add_argument("--rooms", type=int, default=5, help="World size (rooms) per game")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_games):
        path = out_dir / f"mini_{args.rooms}r_{i}.ulx"
        cmd = [
            sys.executable, "-m", "textworld.challenges",
            "custom",
            "--world-size", str(args.rooms),
            "--quest-length", "2",
            "--output", str(path),
            "--seed", str(42 + i),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
            print(f"Generated {path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: could not generate {path}: {e}. Install textworld (pip install textworld).")
            break
    print(f"Games in {out_dir}")


if __name__ == "__main__":
    main()
