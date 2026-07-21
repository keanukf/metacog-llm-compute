#!/usr/bin/env python3
"""
Phase 2 — Adaptive Allocation: run domains x instances x strategies x runs.
Supports --resume via checkpoint_dir.
Progress: timestamped batch lines (elapsed, ep/h, ETA); optional --verbose-episodes / --verbose-steps.
Usage: python scripts/run_phase2.py --config configs/experiment_core.yaml [--resume] [--real]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.dotenv_loader import load_dotenv_if_present

_DOTENV_INFO = load_dotenv_if_present(REPO_ROOT)


def load_config(config_path: str | Path) -> dict:
    """Load a run config, merging ``extends: <relative-or-repo-relative path>`` if present.

    Dev/overlay configs (e.g. ``configs/dev/*.yaml``) commonly extend
    ``experiment_core.yaml`` via an ``extends`` key (see ``configs/dev/gate_d_calibration.yaml``).
    A plain ``yaml.safe_load`` silently drops every key not restated in the overlay (model,
    episode, domain_prompts, paths, ...), which only surfaces as a downstream KeyError/behavior
    change, not a load error. Reuse the same recursive merge Gate D's diagnostic scripts already
    rely on (``scripts.sweep_textworld_difficulty._load_merged_config``) so overlay configs behave
    identically here.
    """
    from scripts.difficulty_calibration.sweep_textworld_difficulty import _load_merged_config

    return _load_merged_config(Path(config_path))


def _episode_mean_tle(ep: dict) -> float | None:
    vals: list[float] = []
    for sd in ep.get("steps_detail") or []:
        tle = sd.get("tle") if isinstance(sd, dict) else None
        if isinstance(tle, dict):
            v = tle.get("mean_entropy")
            if isinstance(v, (int, float)):
                vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _episode_mean_vc(ep: dict) -> float | None:
    vals: list[float] = []
    for sd in ep.get("steps_detail") or []:
        if not isinstance(sd, dict):
            continue
        v = sd.get("vc")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pearsonr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _build_run_summary(
    *,
    checkpoint_dir: Path,
    total_wall_time_s: float,
    episodes_attempted: int,
    episodes_completed: int,
    episodes_failed: int,
    execution_metrics: dict[str, Any] | None = None,
) -> dict:
    episodes: list[dict] = []
    for p in sorted(checkpoint_dir.glob("ep_*.json")):
        try:
            with open(p) as f:
                episodes.append(json.load(f))
        except Exception:
            continue

    by_domain: dict[str, dict[str, float]] = {}
    domain_acc: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        d = str(ep.get("domain", "unknown"))
        domain_acc[d].append(ep)
    for d, eps in domain_acc.items():
        n = len(eps)
        if n == 0:
            continue
        succ = sum(1 for e in eps if e.get("task_success"))
        avg_steps = sum(int(e.get("steps") or 0) for e in eps) / n
        avg_calls = sum(int(e.get("total_lm_calls") or 0) for e in eps) / n
        by_domain[d] = {
            "episodes": n,
            "success_rate": succ / n,
            "avg_steps": avg_steps,
            "avg_lm_calls": avg_calls,
        }

    by_strategy: dict[str, dict[str, float]] = {}
    strat_acc: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        s = str(ep.get("strategy", "unknown"))
        strat_acc[s].append(ep)
    for s, eps in strat_acc.items():
        n = len(eps)
        if n == 0:
            continue
        succ = sum(1 for e in eps if e.get("task_success"))
        avg_tokens = (
            sum(int(e.get("total_tokens_generated") or e.get("tokens") or 0) for e in eps) / n
        )
        by_strategy[s] = {"episodes": n, "success_rate": succ / n, "avg_tokens": avg_tokens}

    tle_means: list[float] = []
    vc_means: list[float] = []
    paired_tle: list[float] = []
    paired_vc: list[float] = []
    for ep in episodes:
        mt = _episode_mean_tle(ep)
        mv = _episode_mean_vc(ep)
        if isinstance(mt, (int, float)):
            tle_means.append(float(mt))
        if isinstance(mv, (int, float)):
            vc_means.append(float(mv))
        if isinstance(mt, (int, float)) and isinstance(mv, (int, float)):
            paired_tle.append(float(mt))
            paired_vc.append(float(mv))
    signal_summary = {
        "tle_mean_across_episodes": (sum(tle_means) / len(tle_means)) if tle_means else 0.0,
        "vc_mean_across_episodes": (sum(vc_means) / len(vc_means)) if vc_means else 0.0,
        "tle_vc_correlation": _pearsonr(paired_tle, paired_vc),
    }
    total_episodes = len(episodes)
    avg_episode_time_s = (total_wall_time_s / episodes_completed) if episodes_completed > 0 else 0.0
    summary = {
        "total_episodes": total_episodes,
        "new_episodes_this_run": int(episodes_completed),
        "episodes_attempted": int(episodes_attempted),
        "episodes_completed": int(episodes_completed),
        "episodes_failed": int(episodes_failed),
        "errors": int(episodes_failed),
        "total_wall_time_s": float(total_wall_time_s),
        "avg_episode_time_s": float(avg_episode_time_s),
        "by_domain": by_domain,
        "by_stage_or_strategy": by_strategy,
        "signal_summary": signal_summary,
        "timestamp_end_utc": datetime.now(timezone.utc).isoformat(),
    }
    if execution_metrics is not None:
        summary["execution_metrics"] = execution_metrics
    return summary


def _rng_for_episode(ep_id: str) -> random.Random:
    """Deterministic RNG per episode id (stable across processes for resume)."""
    digest = hashlib.md5(ep_id.encode(), usedforsecurity=False).hexdigest()
    seed = int(digest[:16], 16) % (2**32 - 1) + 1
    return random.Random(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--checkpoint-dir", default="data/results/phase2")
    parser.add_argument(
        "--no-timestamp-run",
        action="store_true",
        help="Write checkpoints directly under --checkpoint-dir instead of a new phase2_*_UTC folder. "
        "Ignored when --resume.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--real", action="store_true", help="Use real model (vLLM/HF) when available"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N new episodes (0=use config/default)",
    )
    parser.add_argument(
        "--verbose-episodes",
        action="store_true",
        help="Log each episode when it completes (one line)",
    )
    parser.add_argument(
        "--verbose-steps", action="store_true", help="Log each environment step (very noisy)"
    )
    parser.add_argument(
        "--allow-history-truncation",
        action="store_true",
        help="Allow history truncation params in config (not valid for confirmatory H3).",
    )
    args = parser.parse_args()
    config_path = (
        REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    )
    checkpoint_base = Path(args.checkpoint_dir)
    if not checkpoint_base.is_absolute():
        checkpoint_base = REPO_ROOT / checkpoint_base
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    if args.resume:
        checkpoint_dir = checkpoint_base
    elif args.no_timestamp_run:
        checkpoint_dir = checkpoint_base
    else:
        from src.utils.run_output_layout import make_run_subdirectory

        checkpoint_dir = make_run_subdirectory(checkpoint_base, prefix="phase2")
    config = load_config(config_path)
    lg = config.get("logging") or {}
    from src.utils.logprob_sidecar import LogprobSidecarConfig

    logprob_sidecar = LogprobSidecarConfig.from_logging_config(lg)
    save_vc_distributions = bool(lg.get("save_vc_distributions", False))
    vc_export_format = str(lg.get("vc_export_format", "json")).lower()
    vc_subdir = str(lg.get("vc_subdir", "vc"))
    from src.utils.trace_debug_view import resolve_step_trace_flags

    save_step_traces, _, _, _ = resolve_step_trace_flags(config)

    from src.execution.backend.factory import create_execution_backend
    from src.execution.config import ExecutionConfig, write_frozen_execution_params
    from src.execution.episode_runner import Phase2RunContext, run_phase2_job
    from src.execution.metrics import build_execution_metrics
    from src.execution.scheduler import EpisodeScheduler
    from src.execution.worklist import build_phase2_worklist
    from src.utils.checkpointing import list_completed_episodes
    from src.utils.logging_utils import write_run_metadata
    from src.utils.run_output_layout import write_short_run_info
    from src.utils.run_progress import (
        format_run_elapsed,
        log,
        log_episode_line,
        log_step_line,
        print_batch_progress,
    )
    from src.utils.run_resilience import load_quarantined_episode_ids

    exec_cfg = ExecutionConfig.from_config(config, real=bool(args.real))
    exec_cfg.enforce_frozen_or_exit()
    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
    quarantined = load_quarantined_episode_ids(checkpoint_dir)
    log(f"Checkpoint directory: {checkpoint_dir.resolve()}")
    phase2 = config.get("phase2", {})
    domains = phase2.get("domains", ["textworld", "tower_of_hanoi"])
    instances_per_domain = phase2.get("instances_per_domain", 50)
    strategies = phase2.get(
        "strategies",
        ["adaptive_tle", "always_c0", "always_c2", "random", "eager_style", "adaptive_vc"],
    )
    runs = phase2.get("runs_per_condition", 5)
    max_steps = config.get("episode", {}).get("max_steps_per_episode", 20)

    from src.agent.allocation_policy import load_policy, policy_signal_for_strategy
    from src.agent.allocator import POLICY_REQUIRED_STRATEGIES, allocate

    policy_required = POLICY_REQUIRED_STRATEGIES & {str(s) for s in strategies}
    policies_by_key: dict[tuple[str, str], Any] = {}
    policy_artifact_path: str | None = None
    policy_artifact_sha256: str | None = None
    if policy_required:
        artifact_rel = phase2.get("policy_artifact")
        if not artifact_rel:
            raise SystemExit(
                "phase2.policy_artifact is required for strategies: "
                + ", ".join(sorted(policy_required))
            )
        artifact_path = Path(str(artifact_rel))
        if not artifact_path.is_absolute():
            artifact_path = REPO_ROOT / artifact_path
        if not artifact_path.is_file():
            raise SystemExit(f"policy artifact not found: {artifact_path}")
        policy_artifact_path = str(artifact_path)
        policy_artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        for dom in domains:
            for strat in policy_required:
                sig = policy_signal_for_strategy(str(strat))
                if sig is None:
                    continue
                policies_by_key[(str(dom), str(strat))] = load_policy(
                    artifact_path,
                    domain=str(dom),
                    signal=sig,
                )

    progress_every = (
        int(args.progress_every)
        if int(args.progress_every) > 0
        else int(phase2.get("progress_every_episodes", 10))
    )
    total = len(domains) * instances_per_domain * len(strategies) * runs
    log(
        f"Phase 2 start — {len(domains)} domains × {instances_per_domain} inst × {len(strategies)} strategies × {runs} runs = {total} episodes "
        f"| resume={args.resume} real={args.real} | already_done={len(completed)}"
    )

    model = create_execution_backend(config, use_real=bool(args.real))
    pilot_mode = "cuda" if args.real else "mock"
    model_cfg = config.get("model", {})
    write_run_metadata(
        checkpoint_dir,
        config,
        script="run_phase2.py",
        config_path=str(config_path),
        pilot_mode=pilot_mode,
        model_name=str(model_cfg.get("name", "unknown")),
        model_dtype=str(model_cfg.get("dtype", "unknown")),
        domains=list(domains),
        total_episodes_planned=int(total),
        resumed_from=int(len(completed)),
        repo_root=REPO_ROOT,
    )
    frozen = config.get("execution") or {}
    if (
        frozen.get("frozen_max_concurrent_episodes") is not None
        and frozen.get("frozen_tle_invariance_eps") is not None
    ):
        from src.execution.config import frozen_execution_params_dict

        write_frozen_execution_params(
            checkpoint_dir,
            frozen_execution_params_dict(
                max_concurrent_episodes=int(frozen["frozen_max_concurrent_episodes"]),
                tle_invariance_eps=float(frozen["frozen_tle_invariance_eps"]),
                eps_derived_under_load=bool(frozen.get("eps_derived_under_load", False)),
            ),
        )
    if policy_artifact_path is not None or args.allow_history_truncation:
        meta_path = checkpoint_dir / "run_metadata.json"
        with open(meta_path) as f:
            meta_obj = json.load(f)
        if policy_artifact_path is not None:
            meta_obj["policy_artifact_path"] = policy_artifact_path
            meta_obj["policy_artifact_sha256"] = policy_artifact_sha256
        if args.allow_history_truncation:
            meta_obj["history_truncation_allowed"] = True
        with open(meta_path, "w") as f:
            json.dump(meta_obj, f, indent=2)
    write_short_run_info(
        checkpoint_dir,
        script="run_phase2.py",
        config_path=config_path,
        extra={
            "checkpoint_dir_resolved": str(checkpoint_dir.resolve()),
            "resume": args.resume,
            "real": args.real,
            "domains": list(domains),
            "total_episodes_planned": int(total),
            "already_completed": int(len(completed)),
            "model_name": str(model_cfg.get("name", "unknown")),
        },
    )
    errors_path = checkpoint_dir / "errors.jsonl"
    run_summary_path = checkpoint_dir / "run_summary.json"
    t_run_start = time.perf_counter()
    last_report_t = time.time()
    rolling: list[dict] = []
    done_count = 0

    run_ctx = Phase2RunContext(
        config=config,
        checkpoint_dir=checkpoint_dir,
        repo_root=REPO_ROOT,
        max_steps=max_steps,
        model_cfg=model_cfg,
        logprob_sidecar=logprob_sidecar,
        save_vc_distributions=save_vc_distributions,
        vc_export_format=vc_export_format,
        vc_subdir=vc_subdir,
        save_step_traces=save_step_traces,
        allow_history_truncation=bool(args.allow_history_truncation),
        verbose_steps=bool(args.verbose_steps),
        policies_by_key=policies_by_key,
        allocate_fn=allocate,
        rng_for_episode=_rng_for_episode,
        tracing_cfg=config.get("tracing"),
        log_fn=log,
        log_step_fn=log_step_line if args.verbose_steps else None,
    )

    jobs = build_phase2_worklist(config, completed=completed, quarantined=quarantined)
    scheduler = EpisodeScheduler(exec_cfg.max_concurrent_episodes)

    def _on_complete(outcome: dict, stats) -> None:
        nonlocal last_report_t, done_count, rolling
        if outcome.get("status") != "completed":
            return
        done_count = stats.done_count
        rolling = list(stats.rolling)
        data = outcome.get("data") or {}
        if args.verbose_episodes:
            log_episode_line(
                "Phase 2",
                str(outcome.get("episode_id")),
                domain=str(outcome.get("domain")),
                label=str(outcome.get("strategy")),
                instance=int(outcome.get("instance") or 0),
                run=int(outcome.get("run") or 0),
                steps=int(data.get("steps") or 0),
                total_lm_calls=int(data.get("total_lm_calls") or 0),
                wall_s=float(outcome.get("ep_wall_time_s") or 0.0),
                success=bool(data.get("task_success")),
            )
        now = time.time()
        elapsed_run = time.perf_counter() - t_run_start
        if (
            done_count == 1
            or (progress_every and done_count > 0 and done_count % progress_every == 0)
            or (now - last_report_t) >= 300
        ):
            last_report_t = now
            total_done = len(completed) + done_count
            rate = (done_count / elapsed_run) if elapsed_run > 0 else 0.0
            remaining = total - total_done
            eta_s = (remaining / rate) if rate > 0 else None
            print_batch_progress(
                phase="Phase 2",
                total_done=total_done,
                total=total,
                new_in_run=done_count,
                elapsed_s=elapsed_run,
                eta_s=eta_s,
                rolling=rolling,
                domain=str(outcome.get("domain")),
                stage_or_strategy=str(outcome.get("strategy")),
                label_key="strategy",
            )

    stats = scheduler.run(
        jobs,
        run_fn=lambda job: run_phase2_job(job, model, run_ctx),
        on_complete=_on_complete,
        errors_path=errors_path,
        checkpoint_dir=checkpoint_dir,
        quarantined=quarantined,
        log_fn=log,
    )
    wall_total = time.perf_counter() - t_run_start
    log(
        f"Phase 2 finished — new episodes: {stats.done_count}; checkpoints: {len(list_completed_episodes(checkpoint_dir))}; "
        f"wall {format_run_elapsed(wall_total)}; max_in_flight={stats.max_in_flight_observed}"
    )
    exec_metrics = build_execution_metrics(
        checkpoint_dir=checkpoint_dir,
        total_wall_time_s=wall_total,
        total_tokens_generated=stats.total_tokens_generated,
        max_in_flight_observed=stats.max_in_flight_observed,
    )
    summary = _build_run_summary(
        checkpoint_dir=checkpoint_dir,
        total_wall_time_s=time.perf_counter() - t_run_start,
        episodes_attempted=stats.episodes_attempted,
        episodes_completed=stats.episodes_completed,
        episodes_failed=stats.episodes_failed,
        execution_metrics=exec_metrics,
    )
    with open(run_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    from src.utils.run_output_layout import finalize_run_debug_views

    dbg = finalize_run_debug_views(checkpoint_dir, config)
    if dbg is not None:
        log(f"Wrote debug views under {dbg}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
