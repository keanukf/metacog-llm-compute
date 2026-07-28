"""Single-episode execution body extracted from phase runners."""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.base_agent import run_adaptive_episode, run_episode
from src.agent.compute_stages import get_step_fn
from src.execution.worklist import EpisodeJob
from src.utils.checkpointing import save_episode_checkpoint
from src.utils.experiment_env import make_experiment_env
from src.utils.history_guard import enforce_full_history_or_exit
from src.utils.logging_utils import (
    write_logprob_distribution_artifacts,
    write_vc_distribution_artifacts,
)
from src.utils.logprob_sidecar import (
    LogprobSidecarConfig,
    filter_logprob_raw_for_sidecar,
)
from src.utils.manifest import manifest_entry_for_instance
from src.utils.step_config import resolve_step_fn_kwargs
from src.utils.tracing import TraceHook, build_trace_hook


@dataclass
class Phase1RunContext:
    config: dict[str, Any]
    checkpoint_dir: Path
    repo_root: Path
    max_steps: int
    model_cfg: dict[str, Any]
    logprob_sidecar: LogprobSidecarConfig
    save_vc_distributions: bool
    vc_export_format: str
    vc_subdir: str
    save_step_traces: bool
    allow_history_truncation: bool
    verbose_steps: bool
    tracing_cfg: dict[str, Any] | None = None
    log_fn: Callable[[str], None] = field(default=lambda msg: None)
    log_step_fn: Callable[[str, dict], None] | None = None


@dataclass
class Phase2RunContext:
    config: dict[str, Any]
    checkpoint_dir: Path
    repo_root: Path
    max_steps: int
    model_cfg: dict[str, Any]
    logprob_sidecar: LogprobSidecarConfig
    save_vc_distributions: bool
    vc_export_format: str
    vc_subdir: str
    save_step_traces: bool
    allow_history_truncation: bool
    verbose_steps: bool
    policies_by_key: dict[tuple[str, str], Any]
    allocate_fn: Any
    rng_for_episode: Callable[[str], random.Random]
    tracing_cfg: dict[str, Any] | None = None
    log_fn: Callable[[str], None] = field(default=lambda msg: None)
    log_step_fn: Callable[[str, dict], None] | None = None


def _make_trace_hook(ctx_tracing: dict[str, Any] | None) -> TraceHook:
    return build_trace_hook(ctx_tracing or {})


def _flush_trace_hook(hook: TraceHook) -> None:
    client = getattr(hook, "_client", None)
    if client is not None:
        flush = getattr(client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass


def _write_logprob_sidecar_for_episode(
    *,
    sidecar_cfg: LogprobSidecarConfig,
    domain: str,
    instance: int,
    episode_id: str,
    logprob_raw_per_step: list[Any] | None,
    checkpoint_dir: Path,
) -> None:
    mode = sidecar_cfg.mode_for(domain, instance)
    if mode == "off" or not logprob_raw_per_step:
        return
    filtered = filter_logprob_raw_for_sidecar(logprob_raw_per_step, mode)
    if not filtered:
        return
    write_logprob_distribution_artifacts(
        episode_id,
        filtered,
        checkpoint_dir,
        export_format=sidecar_cfg.export_format,
        logprob_subdir=sidecar_cfg.subdir,
        sidecar_scope=mode,
    )


def run_phase1_job(
    job: EpisodeJob,
    model: Any,
    ctx: Phase1RunContext,
) -> dict[str, Any]:
    """Run one Phase-1 episode; returns outcome dict for scheduler."""
    assert job.compute_stage is not None
    ep_id = job.episode_id
    domain = job.domain
    inst = job.instance
    stage = job.compute_stage
    run = job.run
    trace_hook = _make_trace_hook(ctx.tracing_cfg)
    t_ep0 = time.perf_counter()
    try:
        step_cfg_probe = resolve_step_fn_kwargs(ctx.config, domain)
        enforce_full_history_or_exit(
            step_cfg_probe,
            allow_history_truncation=bool(ctx.allow_history_truncation),
            script_name="run_phase1.py",
        )
        env = make_experiment_env(domain, inst, ctx.config, ctx.max_steps, ctx.repo_root)
        # ToH's env may use a per-instance cap (3x optimal_steps) instead of ctx.max_steps
        # (see src/utils/experiment_env.py) -- read it back so the step loop bound matches what
        # the env itself will actually allow, instead of silently truncating early.
        effective_max_steps = int(getattr(env, "max_steps", ctx.max_steps))
        step_cfg = resolve_step_fn_kwargs(ctx.config, domain)
        step_cfg["c2_tie_break_seed"] = ep_id
        hist_keys = {
            "history_keep_last_pairs",
            "history_max_obs_chars",
            "history_current_obs_max_chars",
            "history_obs_head_ratio",
            "pin_recipe",
        }
        history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in hist_keys}
        capture_logprobs = ctx.logprob_sidecar.capture_enabled(domain, inst)
        step_fn = get_step_fn(
            stage,
            save_logprob_distributions=capture_logprobs,
            save_vc_distributions=ctx.save_vc_distributions,
            **step_cfg,
        )
        on_step = None
        if ctx.verbose_steps and ctx.log_step_fn is not None:

            def _on_step(info: dict) -> None:
                ctx.log_step_fn(f"Phase 1 {ep_id}", info)  # type: ignore[misc]

            on_step = _on_step
        result = run_episode(
            env,
            model,
            stage,
            step_fn=step_fn,
            max_steps=effective_max_steps,
            on_step=on_step,
            save_logprob_distributions=capture_logprobs,
            save_vc_distributions=ctx.save_vc_distributions,
            save_step_traces=ctx.save_step_traces,
            episode_id=ep_id,
            trace_output_dir=str(ctx.checkpoint_dir),
            trace_model_name=str(ctx.model_cfg.get("name", "")) or None,
            trace_hook=trace_hook,
            trace_session_id=str(ctx.checkpoint_dir.name),
            trace_tags=[
                "phase1",
                str(domain),
                str(stage),
                str(ctx.model_cfg.get("name", "")) if str(ctx.model_cfg.get("name", "")) else "",
            ],
            trace_name=ep_id,
            **history_cfg,
        )
        mentry = manifest_entry_for_instance(domain, inst, ctx.config, ctx.repo_root)
        data = {
            "episode_id": ep_id,
            "domain": domain,
            "instance": inst,
            "compute_stage": stage,
            "run": run,
            "holdout": bool(mentry.get("holdout", False)),
            "difficulty_tier": mentry.get("difficulty_tier"),
            "task_success": result["task_success"],
            "steps": result["steps"],
            "lm_calls": result.get("lm_calls", result["steps"]),
            "tokens": result.get("tokens", result.get("total_tokens_generated", 0)),
            "episode_length_steps": result.get("episode_length_steps", result["steps"]),
            "total_lm_calls": result.get("total_lm_calls", 0),
            "total_tokens_generated": result.get("total_tokens_generated", result.get("tokens", 0)),
            "total_prompt_tokens": result.get("total_prompt_tokens", 0),
            "normalized_compute_cost": result.get("normalized_compute_cost", 0.0),
            "efficiency_score": result.get("efficiency_score"),
            "timestamp_utc": result.get("timestamp_utc"),
            "wall_clock_time": result["wall_clock_time"],
            "tle_per_step": result.get("tle_per_step"),
            "vc_per_step": result.get("vc_per_step"),
            "steps_detail": result.get("steps_detail"),
            "schema_version": "episode.v1",
        }
        if result.get("step_correctness") is not None:
            data["step_correctness"] = result["step_correctness"]
        if result.get("vc_detail_per_step") is not None:
            data["vc_detail_per_step"] = result["vc_detail_per_step"]
        save_episode_checkpoint(ctx.checkpoint_dir, ep_id, data)
        _write_logprob_sidecar_for_episode(
            sidecar_cfg=ctx.logprob_sidecar,
            domain=str(domain),
            instance=int(inst),
            episode_id=ep_id,
            logprob_raw_per_step=result.get("logprob_raw_per_step"),
            checkpoint_dir=ctx.checkpoint_dir,
        )
        if ctx.save_vc_distributions and result.get("vc_detail_per_step"):
            write_vc_distribution_artifacts(
                ep_id,
                result["vc_detail_per_step"],
                ctx.checkpoint_dir,
                export_format=ctx.vc_export_format,
                vc_subdir=ctx.vc_subdir,
            )
        ep_wall = time.perf_counter() - t_ep0
        return {
            "status": "completed",
            "episode_id": ep_id,
            "data": data,
            "ep_wall_time_s": ep_wall,
            "domain": domain,
            "compute_stage": stage,
            "instance": inst,
            "run": run,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "episode_id": ep_id,
            "domain": domain,
            "instance": inst,
            "stage_or_strategy": stage,
            "run": run,
            "exc": exc,
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            trace_hook.episode_end()
        except Exception:
            pass
        _flush_trace_hook(trace_hook)


def run_phase2_job(
    job: EpisodeJob,
    model: Any,
    ctx: Phase2RunContext,
) -> dict[str, Any]:
    """Run one Phase-2 episode; returns outcome dict for scheduler."""
    assert job.strategy is not None
    ep_id = job.episode_id
    domain = job.domain
    inst = job.instance
    strategy = job.strategy
    run = job.run
    trace_hook = _make_trace_hook(ctx.tracing_cfg)
    t_ep0 = time.perf_counter()
    try:
        step_cfg_probe = resolve_step_fn_kwargs(ctx.config, domain)
        enforce_full_history_or_exit(
            step_cfg_probe,
            allow_history_truncation=bool(ctx.allow_history_truncation),
            script_name="run_phase2.py",
        )
        env = make_experiment_env(domain, inst, ctx.config, ctx.max_steps, ctx.repo_root)
        # ToH's env may use a per-instance cap (3x optimal_steps) instead of ctx.max_steps
        # (see src/utils/experiment_env.py) -- read it back so the step loop bound matches what
        # the env itself will actually allow, instead of silently truncating early.
        effective_max_steps = int(getattr(env, "max_steps", ctx.max_steps))
        rng = ctx.rng_for_episode(ep_id)
        on_step = None
        if ctx.verbose_steps and ctx.log_step_fn is not None:

            def _on_step(info: dict) -> None:
                ctx.log_step_fn(f"Phase 2 {ep_id}", info)  # type: ignore[misc]

            on_step = _on_step
        step_cfg = resolve_step_fn_kwargs(ctx.config, domain)
        step_cfg["c2_tie_break_seed"] = ep_id
        ep_policy = ctx.policies_by_key.get((str(domain), str(strategy)))
        capture_logprobs = ctx.logprob_sidecar.capture_enabled(domain, inst)
        result = run_adaptive_episode(
            env,
            model,
            strategy,
            max_steps=effective_max_steps,
            rng=rng,
            policy=ep_policy,
            allocate_fn=ctx.allocate_fn,
            on_step=on_step,
            save_logprob_distributions=capture_logprobs,
            save_vc_distributions=ctx.save_vc_distributions,
            save_step_traces=ctx.save_step_traces,
            episode_id=ep_id,
            trace_output_dir=str(ctx.checkpoint_dir),
            trace_model_name=str(ctx.model_cfg.get("name", "")) or None,
            trace_hook=trace_hook,
            trace_session_id=str(ctx.checkpoint_dir.name),
            trace_tags=[
                "phase2",
                str(domain),
                str(strategy),
                str(ctx.model_cfg.get("name", "")) if str(ctx.model_cfg.get("name", "")) else "",
            ],
            trace_name=ep_id,
            **step_cfg,
        )
        mentry = manifest_entry_for_instance(domain, inst, ctx.config, ctx.repo_root)
        data = {
            "episode_id": ep_id,
            "domain": domain,
            "instance": inst,
            "strategy": strategy,
            "run": run,
            "holdout": bool(mentry.get("holdout", False)),
            "difficulty_tier": mentry.get("difficulty_tier"),
            "task_success": result["task_success"],
            "steps": result["steps"],
            "lm_calls": result.get("lm_calls", result["steps"]),
            "tokens": result.get("tokens", result.get("total_tokens_generated", 0)),
            "episode_length_steps": result.get("episode_length_steps", result["steps"]),
            "total_lm_calls": result.get("total_lm_calls", 0),
            "total_tokens_generated": result.get("total_tokens_generated", result.get("tokens", 0)),
            "total_prompt_tokens": result.get("total_prompt_tokens", 0),
            "normalized_compute_cost": result.get("normalized_compute_cost", 0.0),
            "efficiency_score": result.get("efficiency_score"),
            "timestamp_utc": result.get("timestamp_utc"),
            "wall_clock_time": result["wall_clock_time"],
            "tle_per_step": result.get("tle_per_step"),
            "vc_per_step": result.get("vc_per_step"),
            "stage_per_step": result.get("stage_per_step"),
            "steps_detail": result.get("steps_detail"),
            "schema_version": "episode.v1",
        }
        if result.get("step_correctness") is not None:
            data["step_correctness"] = result["step_correctness"]
        if result.get("vc_detail_per_step") is not None:
            data["vc_detail_per_step"] = result["vc_detail_per_step"]
        save_episode_checkpoint(ctx.checkpoint_dir, ep_id, data)
        _write_logprob_sidecar_for_episode(
            sidecar_cfg=ctx.logprob_sidecar,
            domain=str(domain),
            instance=int(inst),
            episode_id=ep_id,
            logprob_raw_per_step=result.get("logprob_raw_per_step"),
            checkpoint_dir=ctx.checkpoint_dir,
        )
        if ctx.save_vc_distributions and result.get("vc_detail_per_step"):
            write_vc_distribution_artifacts(
                ep_id,
                result["vc_detail_per_step"],
                ctx.checkpoint_dir,
                export_format=ctx.vc_export_format,
                vc_subdir=ctx.vc_subdir,
            )
        ep_wall = time.perf_counter() - t_ep0
        return {
            "status": "completed",
            "episode_id": ep_id,
            "data": data,
            "ep_wall_time_s": ep_wall,
            "domain": domain,
            "strategy": strategy,
            "instance": inst,
            "run": run,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "episode_id": ep_id,
            "domain": domain,
            "instance": inst,
            "stage_or_strategy": strategy,
            "run": run,
            "exc": exc,
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            trace_hook.episode_end()
        except Exception:
            pass
        _flush_trace_hook(trace_hook)


def append_episode_error(
    errors_path: Path,
    *,
    episode_id: str,
    domain: str,
    instance: int,
    stage_or_strategy: str,
    run: int,
    traceback_text: str,
) -> None:
    err = {
        "episode_id": episode_id,
        "domain": domain,
        "instance": instance,
        "stage_or_strategy": stage_or_strategy,
        "run": run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "traceback": traceback_text,
    }
    with open(errors_path, "a", encoding="utf-8") as f:
        f.write(__import__("json").dumps(err) + "\n")
