"""
Minimal agent loop: observation -> LM call -> action -> next observation.
No framework (no LangChain/LlamaIndex). Each compute stage is a clear function.
"""
from __future__ import annotations

import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.allocator import allocate
from src.agent.compute_stages import get_step_fn
from src.utils.logging_utils import write_step_trace_line

# Compute stage step: (observation, history, model) -> (action, tle_or_none, vc_or_none, tokens_used, lm_calls_this_step)
# Backward compat: step may return 3- or 4-tuple; defaults fill in missing values.
StepFn = Callable[
    [str, list[str], Any],
    tuple[str, dict | None, float | None]
    | tuple[str, dict | None, float | None, int]
    | tuple[str, dict | None, float | None, int, int],
]


def _normalize_step_result(
    result: tuple,
) -> tuple[
    str,
    dict | None,
    float | None,
    int,
    int,
    list[dict[str, Any]] | None,
    dict[str, Any] | None,
    str | None,
    str | None,
]:
    """
    Unpack step result as (action, tle, vc, tokens_used, lm_calls_this_step, logprobs_raw,
    vc_detail, prompt_full, response_full).

    Backward compatibility:
    - 3-tuple -> tokens_used=0, lm_calls_this_step=1, logprobs_raw=None, vc_detail=None,
      prompt_full=None, response_full=None
    - 4-tuple -> lm_calls_this_step=1
    - 6-tuple -> legacy: last element is raw per-token logprob list (optional save)
    - 7-tuple -> (..., action_logprobs, vc_detail dict from follow-up)
    - 9-tuple -> (..., prompt_full, response_full) for step tracing / observability
    """
    raw_lp: list[dict[str, Any]] | None = None
    vc_detail: dict[str, Any] | None = None
    prompt_full: str | None = None
    response_full: str | None = None
    n = len(result)
    if n >= 9:
        p7, p8 = result[7], result[8]
        prompt_full = p7 if isinstance(p7, str) else None
        response_full = p8 if isinstance(p8, str) else None
    if n >= 7:
        raw_lp = result[5]  # type: ignore[assignment]
        vc_detail = result[6]  # type: ignore[assignment]
    elif n == 6:
        sixth = result[5]
        if isinstance(sixth, dict) and (
            "vc_prompt" in sixth or "vc_raw_text" in sixth or "vc_value" in sixth
        ):
            vc_detail = sixth
        else:
            raw_lp = sixth  # type: ignore[assignment]
    if n >= 5:
        return (
            result[0],
            result[1],
            result[2],
            int(result[3]),
            int(result[4]),
            raw_lp,
            vc_detail,
            prompt_full,
            response_full,
        )
    if len(result) >= 4:
        return (
            result[0],
            result[1],
            result[2],
            int(result[3]),
            1,
            raw_lp,
            vc_detail,
            prompt_full,
            response_full,
        )
    return result[0], result[1], result[2], 0, 1, raw_lp, vc_detail, prompt_full, response_full


def _copy_step_results(env: Any) -> list[dict[str, Any]] | None:
    """Shallow copy env.step_results so callers cannot mutate episode records via the return dict."""
    raw = getattr(env, "step_results", None)
    if raw is None:
        return None
    return [dict(d) for d in raw]


def _signal_for_next_step(
    tle: dict[str, float] | None,
    vc: float | None,
) -> dict[str, Any] | None:
    """Build allocator signal from the last step's TLE / VC for the *next* step."""
    sig: dict[str, Any] = {}
    if tle is not None and "mean_entropy" in tle:
        sig["mean_entropy"] = tle["mean_entropy"]
    if vc is not None:
        sig["vc"] = vc
    return sig if sig else None


def run_episode(
    env: Any,
    model: Any,
    compute_stage: str,
    step_fn: StepFn | None = None,
    max_steps: int = 20,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    *,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    save_step_traces: bool = False,
    episode_id: str | None = None,
    trace_output_dir: str | Path | None = None,
    trace_model_name: str | None = None,
    trace_hook: Any | None = None,
) -> dict[str, Any]:
    """
    Run one episode: reset env, then loop observation -> step_fn -> env.step until done.

    Args:
        env: Environment with reset() and step(action); has .observation and .done.
        model: Model wrapper with generate(prompt, logprobs=...).
        compute_stage: "C0" | "C1" | "C2" (used to select step_fn if not provided).
        step_fn: If None, a default stub step is used (returns dummy action).
        max_steps: Cap on steps per episode.
        on_step: Optional callback after each env step; dict keys include step_index,
            episode_steps, max_steps, env_done, compute_stage, lm_calls_this_step, total_lm_calls.
        save_logprob_distributions: If True, ``step_fn`` should return 7- or 9-tuples with raw action
            logprob lists; result includes ``logprob_raw_per_step`` for sidecar export.
        save_vc_distributions: If True, include full VC follow-up records in ``vc_detail_per_step``
            (and sidecar JSON when the runner writes it).
        save_step_traces: If True, append one JSON line per env step to
            ``trace_{episode_id}.jsonl`` under ``trace_output_dir`` (full prompt/response).
        episode_id: Used for trace filename when ``save_step_traces`` is True.
        trace_output_dir: Directory for JSONL trace files.
        trace_model_name: Optional model label for Langfuse metadata.
        trace_hook: Optional observability hook (e.g. Langfuse) with ``episode_start`` /
            ``log_action_generation`` / ``episode_end``.

    Returns:
        Dict with keys: steps, task_success, lm_calls, tokens, wall_clock_time,
        tle_per_step (optional), vc_per_step (optional),
        step_correctness (optional): shallow copy of env.step_results when present.
        logprob_raw_per_step (optional): list of per-token logprob dicts per env step when enabled.
        vc_detail_per_step (optional): rich VC metadata per step when follow-up VC is used.
    """
    obs = env.reset()
    # Keep reset() output in history: many envs (e.g. TextWorld) do not repeat the full scene
    # on every step, only the latest feedback + state. Without this, step ≥1 prompts lose the
    # opening game text.
    history: list[str] = [f"OBSERVATION: {obs}"]
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    logprob_raw_per_step: list[list[dict[str, Any]] | None] = []
    vc_detail_per_step: list[dict[str, Any] | None] = []
    steps_detail: list[dict[str, Any]] = []
    if step_fn is None:
        def _stub_step(o: str, h: list[str], m: Any) -> tuple[str, dict | None, float | None, int, int]:
            return "go north", None, None, 0, 0
        step_fn = _stub_step

    trace_path: Path | None = None
    if save_step_traces:
        if not episode_id:
            warnings.warn("save_step_traces is True but episode_id is missing; step traces not written.")
        elif not trace_output_dir:
            warnings.warn("save_step_traces is True but trace_output_dir is missing; step traces not written.")
        else:
            trace_path = Path(trace_output_dir) / f"trace_{episode_id}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

    ep_id_for_trace = episode_id or "ep_unknown"
    if trace_hook is not None:
        trace_hook.episode_start(
            ep_id_for_trace,
            metadata={
                "compute_stage": compute_stage,
                "model": trace_model_name or "",
            },
        )

    t_start = time.perf_counter()
    try:
        while not getattr(env, "done", False) and steps < max_steps:
            step_obs = obs
            history_snapshot = list(history)
            t0 = time.perf_counter()
            raw = step_fn(step_obs, history, model)
            step_wall_time_s = time.perf_counter() - t0
            action, tle, vc, tokens_used, lm_calls_this_step, log_raw, vc_det, prompt_full, response_full = (
                _normalize_step_result(raw)
            )
            tle_per_step.append(tle)
            vc_per_step.append(vc)
            if save_logprob_distributions:
                logprob_raw_per_step.append(log_raw)
            vc_detail_per_step.append(vc_det)
            total_tokens_generated += tokens_used
            total_lm_calls += lm_calls_this_step
            row: dict[str, Any] = {
                "step_index": steps,
                "compute_stage": compute_stage,
                "action": action,
                "tokens_generated": int(tokens_used),
                "lm_calls_this_step": int(lm_calls_this_step),
                "step_wall_time_s": float(step_wall_time_s),
                "tle": tle,
                "vc": vc,
                "correctness": None,
                "observation_length_chars": len(step_obs or ""),
            }
            if vc_det:
                row["vc_prompt"] = vc_det.get("vc_prompt")
                row["vc_raw_text"] = vc_det.get("vc_raw_text")
                row["vc_pattern_matched"] = vc_det.get("vc_pattern_matched")
                row["vc_tokens_used"] = vc_det.get("vc_tokens_used")
                if save_vc_distributions and vc_det.get("vc_logprobs") is not None:
                    row["vc_logprobs"] = vc_det.get("vc_logprobs")
            steps_detail.append(row)
            obs = env.step(action)
            correctness_after: str | None = None
            sr = getattr(env, "step_results", None)
            if isinstance(sr, list) and sr:
                correctness_after = sr[-1].get("correctness")  # type: ignore[union-attr]
            if trace_path is not None:
                trace_rec: dict[str, Any] = {
                    "step_index": steps,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "compute_stage": compute_stage,
                    "prompt_full": prompt_full or "",
                    "response_full": response_full or "",
                    "action_parsed": action,
                    "observation_before": step_obs,
                    "observation_after": obs,
                    "history_snapshot": history_snapshot,
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_after,
                    "step_wall_time_s": float(step_wall_time_s),
                    "lm_calls": int(lm_calls_this_step),
                    "tokens_generated": int(tokens_used),
                }
                if vc_det:
                    trace_rec["vc_followup_prompt"] = vc_det.get("vc_prompt")
                    trace_rec["vc_followup_response"] = vc_det.get("vc_raw_text")
                write_step_trace_line(trace_path, trace_rec)
            if trace_hook is not None:
                lf_meta: dict[str, Any] = {
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_after,
                    "tokens_generated": int(tokens_used),
                    "lm_calls": int(lm_calls_this_step),
                }
                trace_hook.log_action_generation(
                    step_index=steps,
                    compute_stage=compute_stage,
                    prompt=prompt_full or "",
                    output=response_full or "",
                    model_name=trace_model_name,
                    metadata=lf_meta,
                )
            # Maintain growing context as ACTION/OBSERVATION pairs.
            # This allows later prompts/traces to reconstruct the full interaction history.
            history.append(f"ACTION: {action}")
            history.append(f"OBSERVATION: {obs}")
            steps += 1
            if on_step is not None:
                on_step(
                    {
                        "step_index": steps - 1,
                        "episode_steps": steps,
                        "max_steps": max_steps,
                        "env_done": bool(getattr(env, "done", False)),
                        "compute_stage": compute_stage,
                        "lm_calls_this_step": int(lm_calls_this_step),
                        "total_lm_calls": int(total_lm_calls),
                    }
                )
    finally:
        if trace_hook is not None:
            trace_hook.episode_end()

    wall_clock_time = time.perf_counter() - t_start
    if hasattr(env, "task_success"):
        task_success = bool(getattr(env, "task_success"))
    else:
        task_success = bool(getattr(env, "done", False))
    step_correctness = _copy_step_results(env)
    if step_correctness is not None:
        by_idx: dict[int, dict[str, Any]] = {}
        for d in step_correctness:
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            by_idx[idx] = d
        for sd in steps_detail:
            idx = int(sd["step_index"])
            corr = by_idx.get(idx, {}).get("correctness")
            sd["correctness"] = corr if corr is not None else None
    normalized_compute_cost = (
        (total_lm_calls / (max_steps * 3)) if max_steps and (max_steps * 3) > 0 else 0.0
    )
    efficiency_score = (
        (float(task_success) / normalized_compute_cost) if normalized_compute_cost > 0 else None
    )
    out: dict[str, Any] = {
        "steps": steps,  # legacy
        "episode_length_steps": steps,
        "task_success": task_success,
        "lm_calls": steps,  # legacy (was step-count historically)
        "total_lm_calls": int(total_lm_calls),
        "tokens": int(total_tokens_generated),  # legacy
        "total_tokens_generated": int(total_tokens_generated),
        "normalized_compute_cost": float(normalized_compute_cost),
        "efficiency_score": efficiency_score,
        "wall_clock_time": wall_clock_time,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
        "steps_detail": steps_detail,
        "step_correctness": step_correctness,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if save_logprob_distributions:
        out["logprob_raw_per_step"] = logprob_raw_per_step
    if any(x is not None for x in vc_detail_per_step):
        out["vc_detail_per_step"] = vc_detail_per_step
    return out


def run_adaptive_episode(
    env: Any,
    model: Any,
    strategy: str,
    *,
    max_steps: int = 20,
    rng: random.Random | None = None,
    allocate_fn: Callable[..., str] | None = None,
    step_fn_for_stage: Callable[[str], StepFn] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    save_step_traces: bool = False,
    episode_id: str | None = None,
    trace_output_dir: str | Path | None = None,
    trace_model_name: str | None = None,
    trace_hook: Any | None = None,
    vc_mode: str = "inline",
    prompt_prefix: str = "",
    action_max_tokens: int | None = None,
    action_temperature: float | None = None,
    followup_max_tokens: int = 4,
    followup_temperature: float = 0.0,
    action_stop: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run an episode where each step's compute stage comes from ``allocate_fn`` (default: allocator.allocate).

    The signal passed into allocate is built from the *previous* step's TLE / VC; the first step uses
    ``signal=None`` so adaptive strategies default to C0 until a signal exists.
    on_step: Optional callback after each env step (same keys as ``run_episode``, plus ``strategy``).

    Returns:
        Same keys as ``run_episode`` plus ``stage_per_step``: list of ``"C0"`` | ``"C1"`` | ``"C2"`` per step.
    """
    rng = rng or random.Random()
    alloc = allocate_fn or allocate
    if step_fn_for_stage is not None:
        resolve = step_fn_for_stage
    else:

        def resolve(stage: str) -> StepFn:
            return get_step_fn(
                stage,
                save_logprob_distributions=save_logprob_distributions,
                save_vc_distributions=save_vc_distributions,
                vc_mode=vc_mode,
                prompt_prefix=prompt_prefix,
                action_max_tokens=action_max_tokens,
                action_temperature=action_temperature,
                action_stop=action_stop,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
            )

    obs = env.reset()
    # Same as run_episode: retain full opening observation for growing prompts.
    history: list[str] = [f"OBSERVATION: {obs}"]
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    logprob_raw_per_step: list[list[dict[str, Any]] | None] = []
    vc_detail_per_step: list[dict[str, Any] | None] = []
    stage_per_step: list[str] = []
    steps_detail: list[dict[str, Any]] = []
    signal: dict[str, Any] | None = None

    trace_path: Path | None = None
    if save_step_traces:
        if not episode_id:
            warnings.warn("save_step_traces is True but episode_id is missing; step traces not written.")
        elif not trace_output_dir:
            warnings.warn("save_step_traces is True but trace_output_dir is missing; step traces not written.")
        else:
            trace_path = Path(trace_output_dir) / f"trace_{episode_id}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

    ep_id_adapt = episode_id or "ep_unknown"
    if trace_hook is not None:
        trace_hook.episode_start(
            ep_id_adapt,
            metadata={
                "strategy": strategy,
                "model": trace_model_name or "",
            },
        )

    t_start = time.perf_counter()
    try:
        while not getattr(env, "done", False) and steps < max_steps:
            step_obs = obs
            history_snapshot = list(history)
            stage = alloc(signal, strategy, steps, rng)
            stage_per_step.append(stage)
            step_fn = resolve(stage)
            t0 = time.perf_counter()
            raw = step_fn(step_obs, history, model)
            step_wall_time_s = time.perf_counter() - t0
            action, tle, vc, tokens_used, lm_calls_this_step, log_raw, vc_det, prompt_full, response_full = (
                _normalize_step_result(raw)
            )
            tle_per_step.append(tle)
            vc_per_step.append(vc)
            if save_logprob_distributions:
                logprob_raw_per_step.append(log_raw)
            vc_detail_per_step.append(vc_det)
            total_tokens_generated += tokens_used
            total_lm_calls += lm_calls_this_step
            row_a: dict[str, Any] = {
                "step_index": steps,
                "compute_stage": stage,
                "action": action,
                "tokens_generated": int(tokens_used),
                "lm_calls_this_step": int(lm_calls_this_step),
                "step_wall_time_s": float(step_wall_time_s),
                "tle": tle,
                "vc": vc,
                "correctness": None,
                "observation_length_chars": len(step_obs or ""),
            }
            if vc_det:
                row_a["vc_prompt"] = vc_det.get("vc_prompt")
                row_a["vc_raw_text"] = vc_det.get("vc_raw_text")
                row_a["vc_pattern_matched"] = vc_det.get("vc_pattern_matched")
                row_a["vc_tokens_used"] = vc_det.get("vc_tokens_used")
                if save_vc_distributions and vc_det.get("vc_logprobs") is not None:
                    row_a["vc_logprobs"] = vc_det.get("vc_logprobs")
            steps_detail.append(row_a)
            obs = env.step(action)
            correctness_ad: str | None = None
            sr_a = getattr(env, "step_results", None)
            if isinstance(sr_a, list) and sr_a:
                correctness_ad = sr_a[-1].get("correctness")  # type: ignore[union-attr]
            if trace_path is not None:
                trace_rec_a: dict[str, Any] = {
                    "step_index": steps,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "compute_stage": stage,
                    "strategy": strategy,
                    "prompt_full": prompt_full or "",
                    "response_full": response_full or "",
                    "action_parsed": action,
                    "observation_before": step_obs,
                    "observation_after": obs,
                    "history_snapshot": history_snapshot,
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_ad,
                    "step_wall_time_s": float(step_wall_time_s),
                    "lm_calls": int(lm_calls_this_step),
                    "tokens_generated": int(tokens_used),
                }
                if vc_det:
                    trace_rec_a["vc_followup_prompt"] = vc_det.get("vc_prompt")
                    trace_rec_a["vc_followup_response"] = vc_det.get("vc_raw_text")
                write_step_trace_line(trace_path, trace_rec_a)
            if trace_hook is not None:
                lf_meta_a: dict[str, Any] = {
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_ad,
                    "strategy": strategy,
                    "tokens_generated": int(tokens_used),
                    "lm_calls": int(lm_calls_this_step),
                }
                trace_hook.log_action_generation(
                    step_index=steps,
                    compute_stage=stage,
                    prompt=prompt_full or "",
                    output=response_full or "",
                    model_name=trace_model_name,
                    metadata=lf_meta_a,
                )
            # Maintain growing context as ACTION/OBSERVATION pairs.
            history.append(f"ACTION: {action}")
            history.append(f"OBSERVATION: {obs}")
            steps += 1
            signal = _signal_for_next_step(tle, vc)
            if on_step is not None:
                on_step(
                    {
                        "step_index": steps - 1,
                        "episode_steps": steps,
                        "max_steps": max_steps,
                        "env_done": bool(getattr(env, "done", False)),
                        "compute_stage": stage,
                        "strategy": strategy,
                        "lm_calls_this_step": int(lm_calls_this_step),
                        "total_lm_calls": int(total_lm_calls),
                    }
                )
    finally:
        if trace_hook is not None:
            trace_hook.episode_end()

    wall_clock_time = time.perf_counter() - t_start
    if hasattr(env, "task_success"):
        task_success = bool(getattr(env, "task_success"))
    else:
        task_success = bool(getattr(env, "done", False))
    step_correctness = _copy_step_results(env)
    if step_correctness is not None:
        by_idx: dict[int, dict[str, Any]] = {}
        for d in step_correctness:
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            by_idx[idx] = d
        for sd in steps_detail:
            idx = int(sd["step_index"])
            corr = by_idx.get(idx, {}).get("correctness")
            sd["correctness"] = corr if corr is not None else None
    normalized_compute_cost = (
        (total_lm_calls / (max_steps * 3)) if max_steps and (max_steps * 3) > 0 else 0.0
    )
    efficiency_score = (
        (float(task_success) / normalized_compute_cost) if normalized_compute_cost > 0 else None
    )
    adaptive_out: dict[str, Any] = {
        "steps": steps,  # legacy
        "episode_length_steps": steps,
        "task_success": task_success,
        "lm_calls": steps,  # legacy
        "total_lm_calls": int(total_lm_calls),
        "tokens": int(total_tokens_generated),  # legacy
        "total_tokens_generated": int(total_tokens_generated),
        "normalized_compute_cost": float(normalized_compute_cost),
        "efficiency_score": efficiency_score,
        "wall_clock_time": wall_clock_time,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
        "step_correctness": step_correctness,
        "stage_per_step": stage_per_step,
        "steps_detail": steps_detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if save_logprob_distributions:
        adaptive_out["logprob_raw_per_step"] = logprob_raw_per_step
    if any(x is not None for x in vc_detail_per_step):
        adaptive_out["vc_detail_per_step"] = vc_detail_per_step
    return adaptive_out
