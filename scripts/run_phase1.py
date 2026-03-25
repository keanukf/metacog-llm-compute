#!/usr/bin/env python3
"""
Phase 1 — Calibration: run domains x instances x compute_stages x runs.
Supports --resume via checkpoint_dir; skips already completed episodes.
Usage: python scripts/run_phase1.py --config configs/experiment_core.yaml [--resume] [--real]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _print_progress(
    *,
    phase: str,
    total_done: int,
    total: int,
    new_in_run: int,
    pct: float,
    eta_s: float | None,
    rolling: list[dict],
    domain: str,
    stage_or_strategy: str,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    eta_h = (eta_s / 3600.0) if isinstance(eta_s, (int, float)) else None
    print(
        f"[{ts}] {phase} | {total_done}/{total} episodes ({pct:.1f}%) | {new_in_run} new this batch"
        + (f" | ETA: ~{eta_h:.1f}h" if eta_h is not None else "")
    )
    if rolling:
        succ = sum(1 for r in rolling if r.get("task_success"))
        avg_steps = sum(r.get("steps", 0) for r in rolling) / len(rolling)
        avg_ep_s = sum(r.get("ep_wall_time_s", 0.0) for r in rolling) / len(rolling)
        tle_vals = [r["tle_mean"] for r in rolling if isinstance(r.get("tle_mean"), (int, float))]
        vc_vals = [r["vc_mean"] for r in rolling if isinstance(r.get("vc_mean"), (int, float))]
        avg_tle = (sum(tle_vals) / len(tle_vals)) if tle_vals else None
        avg_vc = (sum(vc_vals) / len(vc_vals)) if vc_vals else None
        msg = f"  Last {len(rolling)} episodes: {succ}/{len(rolling)} success | avg {avg_steps:.1f} steps | avg {avg_ep_s:.1f}s/ep"
        if avg_tle is not None:
            msg += f" | avg TLE {avg_tle:.2f}"
        if avg_vc is not None:
            msg += f" | avg VC {avg_vc:.1f}"
        print(msg)
    insts = [r.get("instance") for r in rolling if r.get("domain") == domain and r.get("compute_stage") == stage_or_strategy]
    if insts:
        print(f"  Domain: {domain} | Stage: {stage_or_strategy} | Instance range: {min(insts)}–{max(insts)}")
    else:
        print(f"  Domain: {domain} | Stage: {stage_or_strategy}")


def _build_run_summary(
    *,
    checkpoint_dir: Path,
    total_wall_time_s: float,
    episodes_attempted: int,
    episodes_completed: int,
    episodes_failed: int,
) -> dict:
    # Aggregate over all checkpoints currently present
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
        by_domain[d] = {"episodes": n, "success_rate": succ / n, "avg_steps": avg_steps, "avg_lm_calls": avg_calls}
    by_stage: dict[str, dict[str, float]] = {}
    stage_acc: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        s = str(ep.get("compute_stage", "unknown"))
        stage_acc[s].append(ep)
    for s, eps in stage_acc.items():
        n = len(eps)
        if n == 0:
            continue
        succ = sum(1 for e in eps if e.get("task_success"))
        avg_tokens = sum(int(e.get("total_tokens_generated") or e.get("tokens") or 0) for e in eps) / n
        by_stage[s] = {"episodes": n, "success_rate": succ / n, "avg_tokens": avg_tokens}

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
        "by_stage_or_strategy": by_stage,
        "signal_summary": signal_summary,
        "timestamp_end_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--checkpoint-dir", default="data/results/phase1")
    parser.add_argument("--resume", action="store_true", help="Skip completed episodes")
    parser.add_argument("--real", action="store_true", help="Use real model (vLLM/HF) when available")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N new episodes (0=use config/default)")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from src.utils.checkpointing import list_completed_episodes, save_episode_checkpoint
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.utils.experiment_env import create_experiment_model, make_experiment_env
    from src.utils.logging_utils import write_run_metadata

    completed = list_completed_episodes(checkpoint_dir) if args.resume else set()
    phase1 = config.get("phase1", {})
    domains = phase1.get("domains", ["textworld", "tower_of_hanoi"])
    instances_per_domain = phase1.get("instances_per_domain", 50)
    stages = ["C0", "C1", "C2"]
    runs = phase1.get("runs_per_condition", 5)
    max_steps = config.get("episode", {}).get("max_steps_per_episode", 20)
    progress_every = int(args.progress_every) if int(args.progress_every) > 0 else int(phase1.get("progress_every_episodes", 10))
    total = len(domains) * instances_per_domain * len(stages) * runs
    print(f"Phase 1: {len(domains)} domains x {instances_per_domain} instances x {len(stages)} stages x {runs} runs = {total} episodes")
    print(f"Completed so far: {len(completed)}. Resume={args.resume}. Real model={args.real}.")

    model = create_experiment_model(config, args.real)
    pilot_mode = "cuda" if args.real else "mock"
    model_cfg = config.get("model", {})
    write_run_metadata(
        checkpoint_dir,
        config,
        script="run_phase1.py",
        config_path=str(config_path),
        pilot_mode=pilot_mode,
        model_name=str(model_cfg.get("name", "unknown")),
        model_dtype=str(model_cfg.get("dtype", "unknown")),
        domains=list(domains),
        total_episodes_planned=int(total),
        resumed_from=int(len(completed)),
        repo_root=REPO_ROOT,
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
        for inst in range(instances_per_domain):
            for stage in stages:
                for run in range(runs):
                    ep_id = f"ep_{domain}_{inst}_{stage}_{run}"
                    if ep_id in completed:
                        continue
                    attempted += 1
                    t_ep0 = time.perf_counter()
                    try:
                        env = make_experiment_env(domain, inst, config, max_steps, REPO_ROOT)
                        step_fn = get_step_fn(stage)
                        result = run_episode(env, model, stage, step_fn=step_fn, max_steps=max_steps)
                        data = {
                            "episode_id": ep_id,
                            "domain": domain,
                            "instance": inst,
                            "compute_stage": stage,
                            "run": run,
                            "task_success": result["task_success"],
                            "steps": result["steps"],
                            # Legacy fields kept for backward compatibility
                            "lm_calls": result.get("lm_calls", result["steps"]),
                            "tokens": result.get("tokens", result.get("total_tokens_generated", 0)),
                            # New explicit fields
                            "episode_length_steps": result.get("episode_length_steps", result["steps"]),
                            "total_lm_calls": result.get("total_lm_calls", 0),
                            "total_tokens_generated": result.get("total_tokens_generated", result.get("tokens", 0)),
                            "normalized_compute_cost": result.get("normalized_compute_cost", 0.0),
                            "efficiency_score": result.get("efficiency_score"),
                            "timestamp_utc": result.get("timestamp_utc"),
                            "wall_clock_time": result["wall_clock_time"],
                            "tle_per_step": result.get("tle_per_step"),
                            "vc_per_step": result.get("vc_per_step"),
                            "steps_detail": result.get("steps_detail"),
                        }
                        if result.get("step_correctness") is not None:
                            data["step_correctness"] = result["step_correctness"]
                        save_episode_checkpoint(checkpoint_dir, ep_id, data)
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
                                "compute_stage": stage,
                                "instance": inst,
                            }
                        )
                        if len(rolling) > 10:
                            rolling = rolling[-10:]
                    except Exception:
                        failed += 1
                        err = {
                            "episode_id": ep_id,
                            "domain": domain,
                            "instance": inst,
                            "stage_or_strategy": stage,
                            "run": run,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "traceback": traceback.format_exc(),
                        }
                        with open(errors_path, "a") as f:
                            f.write(json.dumps(err) + "\n")
                        print(f"Warning: episode failed: {ep_id} (continuing)")

                    now = time.time()
                    if (progress_every and done_count % progress_every == 0) or (now - last_report_t) >= 300:
                        last_report_t = now
                        total_done = len(completed) + done_count
                        pct = (total_done / total * 100.0) if total else 0.0
                        elapsed = time.perf_counter() - t_run_start
                        rate = (done_count / elapsed) if elapsed > 0 else 0.0
                        remaining = total - total_done
                        eta_s = (remaining / rate) if rate > 0 else None
                        _print_progress(
                            phase="Phase 1",
                            total_done=total_done,
                            total=total,
                            new_in_run=done_count,
                            pct=pct,
                            eta_s=eta_s,
                            rolling=rolling,
                            domain=domain,
                            stage_or_strategy=stage,
                        )
    print(f"Phase 1 done. New episodes: {done_count}. Total checkpoints: {len(list_completed_episodes(checkpoint_dir))}.")
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
