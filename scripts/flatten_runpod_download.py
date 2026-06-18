#!/usr/bin/env python3
"""Flatten nested ``results/`` folders from RunPod scp downloads into the parent directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def flatten(dest: Path, *, dry_run: bool = False) -> list[str]:
    """
    Move ``dest/results/*`` up into ``dest/`` when present.

    Returns human-readable action lines for logging.
    """
    actions: list[str] = []
    nested = dest / "results"
    if not nested.is_dir():
        return actions
    for child in sorted(nested.iterdir()):
        target = dest / child.name
        if target.exists():
            actions.append(f"skip {child.name} (already exists in {dest})")
            continue
        if dry_run:
            actions.append(f"would move {child} -> {target}")
        else:
            shutil.move(str(child), str(target))
            actions.append(f"moved {child.name} -> {dest}/")
    if not dry_run and nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()
        actions.append(f"removed empty {nested}")
    return actions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dest",
        type=Path,
        nargs="?",
        default=Path("data/results/runpod_pilot"),
        help="Download directory (default: data/results/runpod_pilot)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without moving files")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dest = args.dest if args.dest.is_absolute() else (repo_root / args.dest).resolve()
    if not dest.is_dir():
        print(f"error: not a directory: {dest}", file=sys.stderr)
        return 2

    actions = flatten(dest, dry_run=args.dry_run)
    if not actions:
        print(f"No nested results/ folder under {dest}")
        return 0
    for line in actions:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
