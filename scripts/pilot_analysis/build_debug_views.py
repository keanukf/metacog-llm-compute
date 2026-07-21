#!/usr/bin/env python3
"""Build compact debug_views/ JSON from existing trace_*.jsonl in a run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.trace_debug_view import build_run_debug_views  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate debug_views/ from trace_*.jsonl under a pilot or phase run folder."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing trace_*.jsonl (e.g. data/results/pilot_20260527_152026)",
    )
    parser.add_argument(
        "--head-chars",
        type=int,
        default=800,
        help="Characters kept from the start of each prompt/response (default 800)",
    )
    parser.add_argument(
        "--tail-chars",
        type=int,
        default=800,
        help="Characters kept from the end of each prompt/response (default 800)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    summary = build_run_debug_views(
        run_dir,
        head_chars=int(args.head_chars),
        tail_chars=int(args.tail_chars),
    )
    if summary is None:
        print(f"No trace_*.jsonl found in {run_dir}")
        raise SystemExit(1)

    debug_dir = run_dir / "debug_views"
    print(f"Wrote {debug_dir / 'run_summary.json'}")
    print(f"Episodes: {summary.get('episodes_built', 0)}; steps: {summary.get('total_steps', 0)}")


if __name__ == "__main__":
    main()
