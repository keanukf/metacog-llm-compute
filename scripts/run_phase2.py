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
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.dotenv_loader import load_dotenv_if_present

_DOTENV_INFO = load_dotenv_if_present(REPO_ROOT)


def load_config(config_path: str | Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


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
    return {
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
    save_logprob_distributions = bool(lg.get("save_logprob_distributions", False))
    save_vc_distributions = bool(lg.get("save_vc_distributions", False))
    logprob_export_format = str(lg.get("logprob_export_format", "json")).lower()
    vc_export_format = str(lg.get("vc_export_format", "json")).lower()
    logprob_subdir = str(lg.get("logprob_subdir", "logprobs"))
    vc_subdir = str(lg.get("vc_subdir", "vc"))
    save_step_traces = bool(lg.get("save_step_traces", False))

    from src.agent.base_agent import run_adaptive_episode
    from src.utils.checkpointing import list_completed_episodes, save_episode_checkpoint
    from src.utils.experiment_env import create_experiment_model, make_experiment_env
    from src.utils.logging_utils import (
        write_logprob_distribution_artifacts,
        write_run_metadata,
        write_vc_distribution_artifacts,
    )
    from src.utils.run_output_layout import write_short_run_info
    from src.utils.run_progress import (
        format_run_elapsed,
        log,
        log_episode_line,
        log_step_line,
        print_batch_progress,
    )
    from src.utils.step_config import resolve_step_fn_kwargs
    from src.utils.tracing import optional_trace_hook_from_config

    trace_hook = optional_trace_hook_from_config(config, dotenv_info=_DOTENV_INFO)
    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
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

    model = create_experiment_model(config, args.real)
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
    attempted = 0
    completed_ok = 0
    failed = 0
    rolling: list[dict] = []
    done_count = 0
    for domain in domains:
        log(
            f"Phase 2: domain block — {domain} ({instances_per_domain} instances × {len(strategies)} strategies × {runs} runs)"
        )
        for inst in range(instances_per_domain):
            for strategy in strategies:
                for run in range(runs):
                    ep_id = f"ep_{domain}_{inst}_{strategy}_{run}"
                    if ep_id in completed:
                        continue
                    attempted += 1
                    t_ep0 = time.perf_counter()
                    try:
                        env = make_experiment_env(domain, inst, config, max_steps, REPO_ROOT)
                        rng = _rng_for_episode(ep_id)
                        on_step = None
                        if args.verbose_steps:

                            def _make_on_step(eid: str):
                                def _inner(info: dict) -> None:
                                    log_step_line(f"Phase 2 {eid}", info)

                                return _inner

                            on_step = _make_on_step(ep_id)
                        # For adaptive runs we pass a per-episode seed into C2 so tie-breaking is reproducible.
                        step_cfg = resolve_step_fn_kwargs(config, domain)
                        step_cfg["c2_tie_break_seed"] = ep_id
                        result = run_adaptive_episode(
                            env,
                            model,
                            strategy,
                            max_steps=max_steps,
                            rng=rng,
                            on_step=on_step,
                            save_logprob_distributions=save_logprob_distributions,
                            save_vc_distributions=save_vc_distributions,
                            save_step_traces=save_step_traces,
                            episode_id=ep_id,
                            trace_output_dir=str(checkpoint_dir),
                            trace_model_name=str(model_cfg.get("name", "")) or None,
                            trace_hook=trace_hook,
                            trace_session_id=str(checkpoint_dir.name),
                            trace_tags=[
                                "phase2",
                                str(domain),
                                str(strategy),
                                str(model_cfg.get("name", ""))
                                if str(model_cfg.get("name", ""))
                                else "",
                            ],
                            trace_name=ep_id,
                            **step_cfg,
                        )
                        data = {
                            "episode_id": ep_id,
                            "domain": domain,
                            "instance": inst,
                            "strategy": strategy,
                            "run": run,
                            "task_success": result["task_success"],
                            "steps": result["steps"],
                            # Legacy fields kept for backward compatibility
                            "lm_calls": result.get("lm_calls", result["steps"]),
                            "tokens": result.get("tokens", result.get("total_tokens_generated", 0)),
                            # New explicit fields
                            "episode_length_steps": result.get(
                                "episode_length_steps", result["steps"]
                            ),
                            "total_lm_calls": result.get("total_lm_calls", 0),
                            "total_tokens_generated": result.get(
                                "total_tokens_generated", result.get("tokens", 0)
                            ),
                            "normalized_compute_cost": result.get("normalized_compute_cost", 0.0),
                            "efficiency_score": result.get("efficiency_score"),
                            "timestamp_utc": result.get("timestamp_utc"),
                            "wall_clock_time": result["wall_clock_time"],
                            "tle_per_step": result.get("tle_per_step"),
                            "vc_per_step": result.get("vc_per_step"),
                            "stage_per_step": result.get("stage_per_step"),
                            "steps_detail": result.get("steps_detail"),
                        }
                        if result.get("step_correctness") is not None:
                            data["step_correctness"] = result["step_correctness"]
                        if result.get("vc_detail_per_step") is not None:
                            data["vc_detail_per_step"] = result["vc_detail_per_step"]
                        save_episode_checkpoint(checkpoint_dir, ep_id, data)
                        if save_logprob_distributions and result.get("logprob_raw_per_step"):
                            for p in write_logprob_distribution_artifacts(
                                ep_id,
                                result["logprob_raw_per_step"],
                                checkpoint_dir,
                                export_format=logprob_export_format,
                                logprob_subdir=logprob_subdir,
                            ):
                                log(f"Wrote {p}")
                        if save_vc_distributions and result.get("vc_detail_per_step"):
                            for p in write_vc_distribution_artifacts(
                                ep_id,
                                result["vc_detail_per_step"],
                                checkpoint_dir,
                                export_format=vc_export_format,
                                vc_subdir=vc_subdir,
                            ):
                                log(f"Wrote {p}")
                        completed_ok += 1
                        done_count += 1
                        ep_wall = time.perf_counter() - t_ep0
                        rolling.append(
                            {
                                "task_success": bool(data.get("task_success")),
                                "steps": int(data.get("steps") or 0),
                                "ep_wall_time_s": float(ep_wall),
                                "tle_mean": _episode_mean_tle(data),
                                "vc_mean": _episode_mean_vc(data),
                                "domain": domain,
                                "strategy": strategy,
                                "instance": inst,
                            }
                        )
                        if len(rolling) > 10:
                            rolling = rolling[-10:]
                        if args.verbose_episodes:
                            log_episode_line(
                                "Phase 2",
                                ep_id,
                                domain=domain,
                                label=strategy,
                                instance=inst,
                                run=run,
                                steps=int(data.get("steps") or 0),
                                total_lm_calls=int(data.get("total_lm_calls") or 0),
                                wall_s=float(ep_wall),
                                success=bool(data.get("task_success")),
                            )
                    except Exception:
                        failed += 1
                        err = {
                            "episode_id": ep_id,
                            "domain": domain,
                            "instance": inst,
                            "stage_or_strategy": strategy,
                            "run": run,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "traceback": traceback.format_exc(),
                        }
                        with open(errors_path, "a") as f:
                            f.write(json.dumps(err) + "\n")
                        log(f"Warning: episode failed {ep_id} (continuing)")

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
                            domain=domain,
                            stage_or_strategy=strategy,
                            label_key="strategy",
                        )
    wall_total = time.perf_counter() - t_run_start
    log(
        f"Phase 2 finished — new episodes: {done_count}; checkpoints: {len(list_completed_episodes(checkpoint_dir))}; "
        f"wall {format_run_elapsed(wall_total)}"
    )
    summary = _build_run_summary(
        checkpoint_dir=checkpoint_dir,
        total_wall_time_s=time.perf_counter() - t_run_start,
        episodes_attempted=attempted,
        episodes_completed=completed_ok,
        episodes_failed=failed,
    )
    with open(run_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
