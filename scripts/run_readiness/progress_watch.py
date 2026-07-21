#!/usr/bin/env python3
"""Lightweight, run-agnostic progress watcher — polls an output directory from the outside.

Doesn't hook into run_phase1.py, run_phase2.py, or any probe script -- every one of them already
writes one JSON file per finished episode plus (when save_step_traces is on) one growing
trace_*.jsonl per in-flight episode into a checkpoint/output directory. This just watches that
directory and reports counts, grouped by (domain, stage/strategy) parsed from filenames, plus a
rough steps/min throughput. Safe to point at a live directory on the pod (over SSH) or locally.

Usage:
    python scripts/progress_watch.py --dir data/results/phase1/phase1_20260721_120000
    python scripts/progress_watch.py --dir <dir> --expected 1500 --interval 10
    python scripts/progress_watch.py --dir <dir> --once   # single snapshot, no loop
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

DOMAINS = ("textworld", "tower_of_hanoi")
STAGES = (
    "C0",
    "C1",
    "C2",
    "always_c0",
    "always_c2",
    "random",
    "adaptive_tle",
    "adaptive_vc",
    "eager_style",
)

_DOMAIN_RE = re.compile("|".join(re.escape(d) for d in DOMAINS))
_STAGE_RE = re.compile(r"(?:^|_)(" + "|".join(re.escape(s) for s in STAGES) + r")(?:_|$)")


def _classify(stem: str) -> tuple[str, str]:
    dom_m = _DOMAIN_RE.search(stem)
    domain = dom_m.group(0) if dom_m else "?"
    stage_m = _STAGE_RE.search(stem)
    stage = stage_m.group(1) if stage_m else "?"
    return domain, stage


def snapshot(out_dir: Path) -> dict:
    """One poll: completed episodes (ep_*.json / qc_*.json) + in-flight step counts (trace_*.jsonl)."""
    done: dict[tuple[str, str], int] = defaultdict(int)
    done_total = 0
    inflight: dict[tuple[str, str], int] = defaultdict(int)
    inflight_steps_total = 0
    total_steps_observed = 0  # done episodes' own step count + in-flight trace line counts
    earliest_mtime: float | None = None

    for p in out_dir.glob("*.json"):
        stem = p.stem
        if not (stem.startswith("ep_") or stem.startswith("qc_")):
            continue
        domain, stage = _classify(stem)
        done[(domain, stage)] += 1
        done_total += 1
        try:
            mtime = p.stat().st_mtime
            earliest_mtime = mtime if earliest_mtime is None else min(earliest_mtime, mtime)
            data = json.loads(p.read_text())
            total_steps_observed += int(data.get("episode_length_steps") or data.get("steps") or 0)
        except (json.JSONDecodeError, OSError):
            pass

    for p in out_dir.glob("trace_*.jsonl"):
        ep_id = p.stem[len("trace_") :]
        # Skip traces whose episode already finished (avoid double-counting steps).
        if (out_dir / f"{ep_id}.json").exists():
            continue
        domain, stage = _classify(ep_id)
        try:
            mtime = p.stat().st_mtime
            earliest_mtime = mtime if earliest_mtime is None else min(earliest_mtime, mtime)
            n_lines = sum(1 for line in p.read_text().splitlines() if line.strip())
        except OSError:
            n_lines = 0
        inflight[(domain, stage)] += 1
        inflight_steps_total += n_lines
        total_steps_observed += n_lines

    return {
        "done": dict(done),
        "done_total": done_total,
        "inflight": dict(inflight),
        "inflight_steps_total": inflight_steps_total,
        "total_steps_observed": total_steps_observed,
        "earliest_mtime": earliest_mtime,
    }


def render(snap: dict, *, expected: int | None, elapsed_s: float) -> str:
    lines = []
    cells = sorted(set(snap["done"]) | set(snap["inflight"]))
    for cell in cells:
        d, s = cell
        n_done = snap["done"].get(cell, 0)
        n_inflight = snap["inflight"].get(cell, 0)
        suffix = f" (+{n_inflight} in flight)" if n_inflight else ""
        lines.append(f"  {d}/{s}: {n_done} done{suffix}")

    total_str = (
        f"{snap['done_total']}/{expected}" if expected is not None else str(snap["done_total"])
    )
    rate = (snap["total_steps_observed"] / elapsed_s * 60) if elapsed_s > 0 else 0.0
    header = (
        f"[{time.strftime('%H:%M:%S')}] episodes done: {total_str} | "
        f"in-flight episodes: {sum(snap['inflight'].values())} | "
        f"~{rate:.1f} steps/min (elapsed {elapsed_s / 60:.1f}m)"
    )
    return header + ("\n" + "\n".join(lines) if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", required=True, help="Checkpoint/output directory to watch.")
    parser.add_argument(
        "--expected", type=int, default=None, help="Total episodes expected (optional)."
    )
    parser.add_argument("--interval", type=float, default=10.0, help="Poll interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Single snapshot, no loop.")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    t0 = time.time()  # wall-clock, comparable to file mtimes below
    while True:
        if not out_dir.exists():
            print(f"[{time.strftime('%H:%M:%S')}] waiting for {out_dir} to exist...")
        else:
            snap = snapshot(out_dir)
            # Watcher-uptime elapsed grows accurately once we've been polling a while; for a
            # single --once snapshot (or a watcher just started against an already-running job)
            # it's near zero, so fall back to time-since-earliest-file-seen, whichever is larger.
            now = time.time()
            since_files = (now - snap["earliest_mtime"]) if snap["earliest_mtime"] else 0.0
            elapsed_s = max(now - t0, since_files)
            print(render(snap, expected=args.expected, elapsed_s=elapsed_s))
            if (
                args.expected is not None
                and snap["done_total"] >= args.expected
                and not snap["inflight"]
            ):
                print("All expected episodes done, no in-flight traces remaining -- stopping.")
                break
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
