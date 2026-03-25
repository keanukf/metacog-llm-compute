"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (best-of-N).
"""
from __future__ import annotations

from typing import Any

from src.signals import token_entropy, verbalized_confidence


def c0_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """
    C0: One action call with logprobs (TLE); optional VC prompt.
    Returns (action_text, tle_dict, vc_or_none, tokens_used, lm_calls_this_step).
    """
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    tokens_used = len(logprobs) if logprobs else 0
    return text.strip(), tle, vc, tokens_used, 1


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """
    C1: CoT generation + self-verification (two calls conceptually).
    Stub: single call returning action and optional TLE/VC.
    """
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    tokens_used = len(logprobs) if logprobs else 0
    # NOTE: current C1 stub is a single model call; when implemented as CoT + verify, set this to 2.
    return text.strip(), tle, vc, tokens_used, 1


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


def c2_step(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """
    C2: Best-of-N (e.g. 3) samples + majority vote.
    Generates n_samples times, then picks action by majority vote; returns TLE/VC from winning sample.
    """
    prompt = "\n".join(history + [observation]) if history else observation
    samples: list[tuple[str, Any]] = []
    total_tokens = 0
    for _ in range(n_samples):
        text, logprobs = model.generate(prompt, logprobs=True)
        samples.append((text.strip(), logprobs))
        total_tokens += len(logprobs) if logprobs else 0
    actions = [s[0] for s in samples]
    winner = _majority_vote(actions)
    # Use TLE/VC from first sample that produced the winning action
    tle, vc = None, None
    for text, logprobs in samples:
        if text == winner:
            tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
            vc = verbalized_confidence.parse_confidence(text)
            break
    if tle is None and samples:
        text, logprobs = samples[0]
        tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
        vc = verbalized_confidence.parse_confidence(text)
    return winner, tle, vc, total_tokens, int(n_samples)


def get_step_fn(stage: str):
    """Return the step function for stage 'C0', 'C1', or 'C2'."""
    return {"C0": c0_step, "C1": c1_step, "C2": c2_step}.get(stage, c0_step)
