"""
Minimal agent loop: observation -> LM call -> action -> next observation.
No framework (no LangChain/LlamaIndex). Each compute stage is a clear function.
"""
from __future__ import annotations

import time
from typing import Any, Callable

# Compute stage step: (observation, history, model) -> (action, tle_or_none, vc_or_none, tokens_used)
# tokens_used is output token count for this step; step may return 3-tuple for backward compat (tokens_used=0).
StepFn = Callable[[str, list[str], Any], tuple[str, dict | None, float | None] | tuple[str, dict | None, float | None, int]]


def _normalize_step_result(result: tuple) -> tuple[str, dict | None, float | None, int]:
    """Unpack step result as (action, tle, vc, tokens_used); default tokens_used=0 for 3-tuple."""
    if len(result) >= 4:
        return result[0], result[1], result[2], int(result[3])
    return result[0], result[1], result[2], 0


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
        step_correctness (optional): copy of env.step_results when present.
    """
    obs = env.reset()
    history: list[str] = []
    steps = 0
    lm_calls = 0
    tokens = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    if step_fn is None:
        def _stub_step(o: str, h: list[str], m: Any) -> tuple[str, dict | None, float | None, int]:
            return "go north", None, None, 0
        step_fn = _stub_step
    t_start = time.perf_counter()
    while not getattr(env, "done", False) and steps < max_steps:
        raw = step_fn(obs, history, model)
        action, tle, vc, tokens_used = _normalize_step_result(raw)
        tle_per_step.append(tle)
        vc_per_step.append(vc)
        tokens += tokens_used
        obs = env.step(action)
        history.append(obs)
        steps += 1
        lm_calls += 1
    wall_clock_time = time.perf_counter() - t_start
    if hasattr(env, "task_success"):
        task_success = bool(getattr(env, "task_success"))
    else:
        task_success = bool(getattr(env, "done", False))
    out: dict[str, Any] = {
        "steps": steps,
        "task_success": task_success,
        "lm_calls": lm_calls,
        "tokens": tokens,
        "wall_clock_time": wall_clock_time,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
    }
    step_results = getattr(env, "step_results", None)
    if step_results is not None:
        out["step_correctness"] = list(step_results)
    return out
