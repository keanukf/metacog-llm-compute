#!/usr/bin/env python3
"""Gate F resume-under-concurrency smoke (HART).

Two parts:

1. Batch smoke: repeatedly SIGKILL a real ``scripts/run_phase1.py`` subprocess (mock backend,
   small config) mid-run and resume with ``--resume`` until it completes on its own. Verifies
   the final checkpoint directory has exactly the expected episode ids, each a valid JSON file
   (no missing work item, no leftover corrupt file silently accepted as "done").

2. Write-race probe: targets the specific mechanism behind (1) directly. ``log_episode`` writes
   straight to the final ``ep_*.json`` path (no temp-file+rename), and ``list_completed_episodes``
   only checks file existence, not validity. A large synthetic episode widens the ``json.dump``
   window enough to reliably land a SIGKILL mid-write, so this reproduces (or rules out) a
   truncated-file-treated-as-done race independent of batch-timing luck.

Mock backend only, no GPU/pod required — see docs/consistency_log.md, 2026-07-20 entry.
"""

from __future__ import annotations

import json
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpointing import list_completed_episodes

CONFIG = "configs/dev/gate_f_resume_smoke.yaml"
BATCH_DIR = REPO_ROOT / "data/results/gate_f_resume_smoke/batch"
PROBE_DIR = REPO_ROOT / "data/results/gate_f_resume_smoke/write_race_probe"
MAX_TRIALS = 12


def _expected_episode_ids() -> set[str]:
    from scripts.run_phase1 import load_config
    from src.execution.worklist import build_phase1_worklist

    config = load_config(REPO_ROOT / CONFIG)
    jobs = build_phase1_worklist(config, completed=set(), quarantined=set())
    return {j.episode_id for j in jobs}


def _validate_checkpoint_dir(checkpoint_dir: Path) -> tuple[set[str], list[str]]:
    """Return (valid episode ids, filenames that failed to parse as JSON)."""
    valid: set[str] = set()
    corrupt: list[str] = []
    if not checkpoint_dir.exists():
        return valid, corrupt
    for p in sorted(checkpoint_dir.glob("ep_*.json")):
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            corrupt.append(p.name)
            continue
        if "task_success" not in obj or "schema_version" not in obj:
            corrupt.append(p.name)
            continue
        valid.add(p.stem)
    return valid, corrupt


def _calibrate_full_run_seconds(cmd_base: list[str]) -> float:
    """Time an uninterrupted run against a scratch dir to know where the write-active window is
    (subprocess startup/import overhead dominates a small config's actual episode-writing time,
    so a fixed short kill-delay would only ever land during Python import, never mid-batch)."""
    scratch = BATCH_DIR.parent / "calibration_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    args = [a if a != str(BATCH_DIR) else str(scratch) for a in cmd_base]
    t0 = time.perf_counter()
    subprocess.run(
        args, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    elapsed = time.perf_counter() - t0
    shutil.rmtree(scratch)
    return elapsed


N_CYCLES = 8
MIN_REAL_INTERRUPTIONS = 3  # cycles that must show a genuine kill mid-progress (0 < done < total)


def _run_one_cycle(cmd_base: list[str], full_run_s: float, cycle: int) -> dict[str, Any]:
    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)
    BATCH_DIR.mkdir(parents=True)

    trial_log: list[dict[str, Any]] = []
    corrupt_ever_seen: list[str] = []
    real_interruption = False
    natural_completion = False

    for trial in range(MAX_TRIALS):
        args = cmd_base + (["--resume"] if trial > 0 else [])
        proc = subprocess.Popen(
            args, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Kill within the write-active tail of the run, not during Python/import startup.
        kill_after = random.uniform(0.45, 0.9) * full_run_s
        time.sleep(kill_after)
        killed = False
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
            killed = True
        valid_ids, corrupt = _validate_checkpoint_dir(BATCH_DIR)
        corrupt_ever_seen.extend(corrupt)
        n_valid = len(valid_ids)
        if killed and 0 < n_valid < 48:
            real_interruption = True
        trial_log.append(
            {
                "trial": trial,
                "killed": killed,
                "kill_after_s": round(kill_after, 3),
                "valid_episodes_so_far": n_valid,
                "corrupt_files_this_snapshot": corrupt,
            }
        )
        print(
            f"[batch cycle {cycle}] trial {trial}: killed={killed} after {kill_after:.2f}s | "
            f"valid={n_valid} corrupt={corrupt}"
        )
        if not killed:
            natural_completion = True
            break

    if not natural_completion:
        print(f"[batch cycle {cycle}] running final uninterrupted --resume to completion")
        subprocess.run(cmd_base + ["--resume"], cwd=REPO_ROOT, check=True)

    final_valid, final_corrupt = _validate_checkpoint_dir(BATCH_DIR)
    return {
        "cycle": cycle,
        "real_interruption": real_interruption,
        "trials": trial_log,
        "final_valid_ids": final_valid,
        "final_corrupt": final_corrupt,
        "corrupt_ever_seen": sorted(set(corrupt_ever_seen)),
    }


def run_batch_smoke() -> dict[str, Any]:
    expected_ids = _expected_episode_ids()
    print(f"[batch] expecting {len(expected_ids)} episodes per cycle")

    cmd_base = [
        sys.executable,
        "scripts/run_phase1.py",
        "--config",
        CONFIG,
        "--checkpoint-dir",
        str(BATCH_DIR),
        "--no-timestamp-run",
    ]

    full_run_s = _calibrate_full_run_seconds(cmd_base)
    print(f"[batch] calibrated uninterrupted run time: {full_run_s:.2f}s")

    cycles = [_run_one_cycle(cmd_base, full_run_s, c) for c in range(N_CYCLES)]

    n_real_interruptions = sum(1 for c in cycles if c["real_interruption"])
    all_missing: dict[int, list[str]] = {}
    all_unexpected: dict[int, list[str]] = {}
    all_corrupt_stuck: dict[int, list[str]] = {}
    for c in cycles:
        missing = expected_ids - c["final_valid_ids"]
        unexpected = c["final_valid_ids"] - expected_ids
        if missing:
            all_missing[c["cycle"]] = sorted(missing)
        if unexpected:
            all_unexpected[c["cycle"]] = sorted(unexpected)
        if c["final_corrupt"]:
            all_corrupt_stuck[c["cycle"]] = c["final_corrupt"]

    result = {
        "expected_count": len(expected_ids),
        "n_cycles": N_CYCLES,
        "n_real_interruptions": n_real_interruptions,
        "min_real_interruptions_required": MIN_REAL_INTERRUPTIONS,
        "missing_episode_ids_by_cycle": all_missing,
        "unexpected_episode_ids_by_cycle": all_unexpected,
        "corrupt_files_at_end_by_cycle": all_corrupt_stuck,
        "corrupt_files_seen_transiently_by_cycle": {
            c["cycle"]: c["corrupt_ever_seen"] for c in cycles if c["corrupt_ever_seen"]
        },
        "cycles": [{k: v for k, v in c.items() if k != "final_valid_ids"} for c in cycles],
        "pass": (
            n_real_interruptions >= MIN_REAL_INTERRUPTIONS
            and not all_missing
            and not all_unexpected
            and not all_corrupt_stuck
        ),
    }
    return result


def _write_probe_helper_source() -> str:
    # log_episode()/compact_episode_for_storage() drops per-step fields it doesn't recognize
    # (steps_detail rows without "step_index" are dropped entirely) and strips
    # vc_detail_per_step/logprob_raw_per_step outright -- so bulk has to go in a plain
    # top-level key, which compaction passes through untouched, to actually widen the
    # json.dump() window instead of being silently compacted away before the write check runs.
    return """
import sys, time
sys.path.insert(0, {repo_root!r})
from src.utils.checkpointing import save_episode_checkpoint

PAD_LEN = 20_000_000
data = {{
    "task_success": True,
    "steps": 1,
    "wall_clock_time": 0.0,
    "schema_version": "episode.v1",
    "_test_padding": "x" * PAD_LEN,
}}
t0 = time.perf_counter()
save_episode_checkpoint({target_dir!r}, "ep_probe_big", data)
print(f"WRITE_DONE {{time.perf_counter() - t0:.4f}}")
""".format(repo_root=str(REPO_ROOT), target_dir=str(PROBE_DIR))


def _run_probe_once(delay_s: float | None) -> dict[str, Any]:
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)
    PROBE_DIR.mkdir(parents=True)
    helper = PROBE_DIR / "_helper.py"
    helper.write_text(_write_probe_helper_source())

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, str(helper)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    killed = False
    if delay_s is not None:
        time.sleep(delay_s)
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            killed = True
    out, _ = proc.communicate(timeout=10)
    total_wall_s = time.perf_counter() - t0

    target = PROBE_DIR / "ep_probe_big.json"
    exists = target.exists()
    valid = False
    if exists:
        try:
            obj = json.loads(target.read_text())
            valid = "task_success" in obj and len(obj.get("_test_padding", "")) == 20_000_000
        except (json.JSONDecodeError, OSError):
            valid = False
    completed_set = list_completed_episodes(PROBE_DIR)
    return {
        "delay_s": delay_s,
        "killed": killed,
        "total_wall_s": round(total_wall_s, 4),
        "file_exists": exists,
        "file_valid": valid,
        "counted_as_completed_by_resume": "ep_probe_big" in completed_set,
        "truncated_but_counted_as_done": exists and not valid and "ep_probe_big" in completed_set,
    }


def run_write_race_probe() -> dict[str, Any]:
    baseline = _run_probe_once(delay_s=None)
    total_s = float(baseline["total_wall_s"])
    print(f"[probe] baseline uninterrupted subprocess wall time: {total_s:.4f}s")

    trials = []
    # json.dump of the large payload happens in roughly the back half of the subprocess's
    # wall time (import/module-load overhead dominates the front half) -- sample repeatedly
    # across that window rather than trusting a single self-reported write duration, since
    # OS scheduling jitter is comparable in size to the write itself.
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.5, 0.7]
    for frac in fractions:
        delay = max(0.0002, total_s * frac)
        r = _run_probe_once(delay_s=delay)
        trials.append(r)
        print(
            f"[probe] delay={delay:.4f}s killed={r['killed']} exists={r['file_exists']} "
            f"valid={r['file_valid']} truncated_but_counted_as_done={r['truncated_but_counted_as_done']}"
        )

    race_reproduced = any(t["truncated_but_counted_as_done"] for t in trials)
    return {
        "baseline": baseline,
        "trials": trials,
        "race_reproduced": race_reproduced,
    }


def main() -> None:
    print("=== Gate F resume smoke: batch hard-kill/resume ===")
    batch_result = run_batch_smoke()
    print("\n=== Gate F resume smoke: targeted write-race probe ===")
    probe_result = run_write_race_probe()

    out_dir = REPO_ROOT / "data/results/gate_f_resume_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"batch": batch_result, "write_race_probe": probe_result}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"Batch pass: {batch_result['pass']}")
    print(f"Write-race reproduced: {probe_result['race_reproduced']}")
    print(f"Report: {out_dir / 'report.json'}")

    if not batch_result["pass"] or probe_result["race_reproduced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
