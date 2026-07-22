"""
Lightweight timestamped stdout logging for pilot and long experiment runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_ts() -> str:
    """UTC wall-clock string for log prefixes."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    """Print one timestamped line, flushed immediately.

    Without flush=True, stdout is fully (not line-) buffered once it's a pipe rather than
    a tty -- exactly what `nohup ... > logfile 2>&1 &` gives it. On a multi-hour run that
    can mean the log file stays empty until process exit, making it useless for
    mid-run troubleshooting.
    """
    print(f"[{utc_ts()}] {message}", flush=True)


def log_episode_line(
    phase: str,
    ep_id: str,
    *,
    domain: str,
    label: str,
    instance: int,
    run: int,
    steps: int,
    total_lm_calls: int,
    wall_s: float,
    success: bool,
) -> None:
    """One line after an episode completes (phase1/phase2 verbose mode)."""
    log(
        f"{phase} episode done | {ep_id} | {domain} | {label} inst={instance} run={run} | "
        f"steps={steps} lm_calls={total_lm_calls} wall={wall_s:.2f}s success={success}"
    )


def log_step_line(prefix: str, info: dict[str, Any]) -> None:
    """Format a single env step (pilot / --verbose-steps)."""
    idx = int(info.get("step_index", 0))
    ep_steps = int(info.get("episode_steps", 0))
    mx = int(info.get("max_steps", 0))
    stage = info.get("compute_stage", "?")
    done = info.get("env_done", False)
    lm_step = int(info.get("lm_calls_this_step", 0))
    lm_tot = int(info.get("total_lm_calls", 0))
    log(
        f"{prefix} step {idx + 1}/{mx} (completed {ep_steps}/{mx}) | stage={stage} | "
        f"lm_step={lm_step} lm_total={lm_tot} | env_done={done}"
    )


def format_run_elapsed(perf_elapsed_s: float) -> str:
    """Human-readable elapsed from time.perf_counter() delta."""
    s = float(perf_elapsed_s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m {s % 60:.0f}s"
    return f"{s / 3600:.2f}h"


def print_batch_progress(
    *,
    phase: str,
    total_done: int,
    total: int,
    new_in_run: int,
    elapsed_s: float,
    eta_s: float | None,
    rolling: list[dict],
    domain: str,
    stage_or_strategy: str,
    label_key: str,
) -> None:
    """
    Shared batch summary for Phase 1 (compute_stage) and Phase 2 (strategy).
    `label_key` is 'compute_stage' or 'strategy' for rolling window filtering.
    """
    pct = (total_done / total * 100.0) if total else 0.0
    eta_h = (eta_s / 3600.0) if isinstance(eta_s, (int, float)) and eta_s is not None else None
    eps_per_h = (new_in_run / (elapsed_s / 3600.0)) if elapsed_s > 0 else 0.0
    elapsed_fmt = format_run_elapsed(elapsed_s)
    line = (
        f"{phase} | {total_done}/{total} episodes ({pct:.1f}%) | +{new_in_run} new | "
        f"elapsed {elapsed_fmt} | rate ~{eps_per_h:.1f} ep/h"
    )
    if eta_h is not None:
        line += f" | ETA ~{eta_h:.1f}h"
    log(line)
    if rolling:
        succ = sum(1 for r in rolling if r.get("task_success"))
        avg_steps = sum(r.get("steps", 0) for r in rolling) / len(rolling)
        avg_ep_s = sum(r.get("ep_wall_time_s", 0.0) for r in rolling) / len(rolling)
        tle_vals = [r["tle_mean"] for r in rolling if isinstance(r.get("tle_mean"), (int, float))]
        vc_vals = [r["vc_mean"] for r in rolling if isinstance(r.get("vc_mean"), (int, float))]
        avg_tle = (sum(tle_vals) / len(tle_vals)) if tle_vals else None
        avg_vc = (sum(vc_vals) / len(vc_vals)) if vc_vals else None
        msg = (
            f"  last {len(rolling)} eps: {succ}/{len(rolling)} ok | avg {avg_steps:.1f} steps | "
            f"{avg_ep_s:.1f}s/ep"
        )
        if avg_tle is not None:
            msg += f" | TLE {avg_tle:.2f}"
        if avg_vc is not None:
            msg += f" | VC {avg_vc:.1f}"
        print(msg, flush=True)
    insts = [
        r.get("instance")
        for r in rolling
        if r.get("domain") == domain and r.get(label_key) == stage_or_strategy
    ]
    inst_nums = [int(x) for x in insts if isinstance(x, (int, float, str)) and str(x).strip() != ""]
    if inst_nums:
        print(
            f"  cursor: domain={domain} | {label_key}={stage_or_strategy} | inst {min(inst_nums)}–{max(inst_nums)}",
            flush=True,
        )
    else:
        print(f"  cursor: domain={domain} | {label_key}={stage_or_strategy}", flush=True)
