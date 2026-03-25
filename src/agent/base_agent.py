"""
Minimal agent loop: observation -> LM call -> action -> next observation.
No framework (no LangChain/LlamaIndex). Each compute stage is a clear function.
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.agent.allocator import allocate
from src.agent.compute_stages import get_step_fn

# Compute stage step: (observation, history, model) -> (action, tle_or_none, vc_or_none, tokens_used, lm_calls_this_step)
# Backward compat: step may return 3- or 4-tuple; defaults fill in missing values.
StepFn = Callable[
    [str, list[str], Any],
    tuple[str, dict | None, float | None]
    | tuple[str, dict | None, float | None, int]
    | tuple[str, dict | None, float | None, int, int],
]


def _normalize_step_result(result: tuple) -> tuple[str, dict | None, float | None, int, int]:
    """
    Unpack step result as (action, tle, vc, tokens_used, lm_calls_this_step).

    Backward compatibility:
    - 3-tuple -> tokens_used=0, lm_calls_this_step=1
    - 4-tuple -> lm_calls_this_step=1
    """
    if len(result) >= 5:
        return result[0], result[1], result[2], int(result[3]), int(result[4])
    if len(result) >= 4:
        return result[0], result[1], result[2], int(result[3]), 1
    return result[0], result[1], result[2], 0, 1


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
) -> dict[str, Any]:
    """
    Run one episode: reset env, then loop observation -> step_fn -> env.step until done.

    Args:
        env: Environment with reset() and step(action); has .observation and .done.
        model: Model wrapper with generate(prompt, logprobs=...).
        compute_stage: "C0" | "C1" | "C2" (used to select step_fn if not provided).
        step_fn: If None, a default stub step is used (returns dummy action).
        max_steps: Cap on steps per episode.

    Returns:
        Dict with keys: steps, task_success, lm_calls, tokens, wall_clock_time,
        tle_per_step (optional), vc_per_step (optional),
        step_correctness (optional): shallow copy of env.step_results when present.
    """
    obs = env.reset()
    history: list[str] = []
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    steps_detail: list[dict[str, Any]] = []
    if step_fn is None:
        def _stub_step(o: str, h: list[str], m: Any) -> tuple[str, dict | None, float | None, int, int]:
            return "go north", None, None, 0, 0
        step_fn = _stub_step
    t_start = time.perf_counter()
    while not getattr(env, "done", False) and steps < max_steps:
        step_obs = obs
        t0 = time.perf_counter()
        raw = step_fn(step_obs, history, model)
        step_wall_time_s = time.perf_counter() - t0
        action, tle, vc, tokens_used, lm_calls_this_step = _normalize_step_result(raw)
        tle_per_step.append(tle)
        vc_per_step.append(vc)
        total_tokens_generated += tokens_used
        total_lm_calls += lm_calls_this_step
        steps_detail.append(
            {
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
        )
        obs = env.step(action)
        history.append(obs)
        steps += 1
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
) -> dict[str, Any]:
    """
    Run an episode where each step's compute stage comes from ``allocate_fn`` (default: allocator.allocate).

    The signal passed into allocate is built from the *previous* step's TLE / VC; the first step uses
    ``signal=None`` so adaptive strategies default to C0 until a signal exists.

    Returns:
        Same keys as ``run_episode`` plus ``stage_per_step``: list of ``"C0"`` | ``"C1"`` | ``"C2"`` per step.
    """
    rng = rng or random.Random()
    alloc = allocate_fn or allocate
    resolve = step_fn_for_stage or get_step_fn

    obs = env.reset()
    history: list[str] = []
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    stage_per_step: list[str] = []
    steps_detail: list[dict[str, Any]] = []
    signal: dict[str, Any] | None = None

    t_start = time.perf_counter()
    while not getattr(env, "done", False) and steps < max_steps:
        step_obs = obs
        stage = alloc(signal, strategy, steps, rng)
        stage_per_step.append(stage)
        step_fn = resolve(stage)
        t0 = time.perf_counter()
        raw = step_fn(step_obs, history, model)
        step_wall_time_s = time.perf_counter() - t0
        action, tle, vc, tokens_used, lm_calls_this_step = _normalize_step_result(raw)
        tle_per_step.append(tle)
        vc_per_step.append(vc)
        total_tokens_generated += tokens_used
        total_lm_calls += lm_calls_this_step
        steps_detail.append(
            {
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
        )
        obs = env.step(action)
        history.append(obs)
        steps += 1
        signal = _signal_for_next_step(tle, vc)

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
    return {
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
