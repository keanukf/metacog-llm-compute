"""
Minimal agent loop: observation -> LM call -> action -> next observation.
No framework (no LangChain/LlamaIndex). Each compute stage is a clear function.
"""
from __future__ import annotations

from typing import Any, Callable

# Compute stage step: (observation, history) -> (action, tle_or_none, vc_or_none)
StepFn = Callable[[str, list[str], Any], tuple[str, dict | None, float | None]]


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
        tle_per_step (optional), vc_per_step (optional).
    """
    obs = env.reset()
    history: list[str] = []
    steps = 0
    lm_calls = 0
    tokens = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    if step_fn is None:
        def _stub_step(o: str, h: list[str], m: Any) -> tuple[str, dict | None, float | None]:
            return "go north", None, None
        step_fn = _stub_step
    while not getattr(env, "done", False) and steps < max_steps:
        action, tle, vc = step_fn(obs, history, model)
        tle_per_step.append(tle)
        vc_per_step.append(vc)
        obs = env.step(action)
        history.append(obs)
        steps += 1
        lm_calls += 1
        tokens += 200
    return {
        "steps": steps,
        "task_success": getattr(env, "done", False),
        "lm_calls": lm_calls,
        "tokens": tokens,
        "wall_clock_time": 0.0,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
    }
