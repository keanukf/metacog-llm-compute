"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (best-of-N).
"""
from __future__ import annotations

from typing import Any

from src.signals import token_entropy, verbalized_confidence

# Return type: (action, tle, vc, tokens_used, lm_calls) or with optional 6th element logprobs_raw
StepReturn = (
    tuple[str, dict[str, float] | None, float | None, int, int]
    | tuple[str, dict[str, float] | None, float | None, int, int, list[dict[str, Any]] | None]
)


def _c0_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    return_raw_logprobs: bool,
) -> StepReturn:
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    tokens_used = len(logprobs) if logprobs else 0
    base = (text.strip(), tle, vc, tokens_used, 1)
    if return_raw_logprobs:
        return (*base, logprobs)
    return base


def c0_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C0: One action call with logprobs (TLE); optional VC prompt."""
    r = _c0_step_core(observation, history, model, return_raw_logprobs=False)
    return r  # type: ignore[return-value]


def _c1_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    return_raw_logprobs: bool,
) -> StepReturn:
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    tokens_used = len(logprobs) if logprobs else 0
    base = (text.strip(), tle, vc, tokens_used, 1)
    if return_raw_logprobs:
        return (*base, logprobs)
    return base


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C1: CoT + verify stub — single call."""
    r = _c1_step_core(observation, history, model, return_raw_logprobs=False)
    return r  # type: ignore[return-value]


def _majority_vote(actions: list[str]) -> str:
    """Return the most frequent action; on tie, return the first that achieves the max count."""
    if not actions:
        return ""
    from collections import Counter

    counts = Counter(actions)
    max_count = max(counts.values())
    for a in actions:
        if counts[a] == max_count:
            return a
    return actions[0]


def _c2_step_core(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
    *,
    return_raw_logprobs: bool,
) -> StepReturn:
    prompt = "\n".join(history + [observation]) if history else observation
    samples: list[tuple[str, Any]] = []
    total_tokens = 0
    for _ in range(n_samples):
        text, logprobs = model.generate(prompt, logprobs=True)
        samples.append((text.strip(), logprobs))
        total_tokens += len(logprobs) if logprobs else 0
    actions = [s[0] for s in samples]
    winner = _majority_vote(actions)
    tle, vc = None, None
    win_logprobs: list[dict[str, Any]] | None = None
    for text, logprobs in samples:
        if text == winner:
            tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
            vc = verbalized_confidence.parse_confidence(text)
            win_logprobs = logprobs
            break
    if tle is None and samples:
        text, logprobs = samples[0]
        tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
        vc = verbalized_confidence.parse_confidence(text)
        win_logprobs = logprobs
    base = (winner, tle, vc, total_tokens, int(n_samples))
    if return_raw_logprobs:
        return (*base, win_logprobs)
    return base


def c2_step(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C2: Best-of-N samples + majority vote."""
    r = _c2_step_core(observation, history, model, n_samples, return_raw_logprobs=False)
    return r  # type: ignore[return-value]


def get_step_fn(stage: str, *, save_logprob_distributions: bool = False):
    """
    Return the step function for stage 'C0', 'C1', or 'C2'.

    If ``save_logprob_distributions`` is True, each call returns a 6-tuple with
    raw per-token logprob rows (last element) for optional persistence.
    """
    core_map = {
        "C0": _c0_step_core,
        "C1": _c1_step_core,
        "C2": _c2_step_core,
    }
    fn = core_map.get(stage, _c0_step_core)

    if not save_logprob_distributions:
        legacy = {"C0": c0_step, "C1": c1_step, "C2": c2_step}.get(stage, c0_step)
        return legacy

    if stage == "C2":

        def _w2(obs: str, hist: list[str], m: Any):
            return fn(obs, hist, m, return_raw_logprobs=True)

        return _w2

    def _w(obs: str, hist: list[str], m: Any):
        return fn(obs, hist, m, return_raw_logprobs=True)

    return _w
