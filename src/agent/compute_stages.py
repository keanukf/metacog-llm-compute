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
) -> tuple[str, dict[str, float] | None, float | None]:
    """
    C0: One action call with logprobs (TLE); optional VC prompt.
    Returns (action_text, tle_dict, vc_or_none).
    """
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    return text.strip(), tle, vc


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None]:
    """
    C1: CoT generation + self-verification (two calls conceptually).
    Stub: single call returning action and optional TLE/VC.
    """
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    return text.strip(), tle, vc


def c2_step(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
) -> tuple[str, dict[str, float] | None, float | None]:
    """
    C2: Best-of-N (e.g. 3) samples + majority vote.
    Stub: single call; full impl would run n_samples and vote.
    """
    prompt = "\n".join(history + [observation]) if history else observation
    text, logprobs = model.generate(prompt, logprobs=True)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    vc = verbalized_confidence.parse_confidence(text)
    return text.strip(), tle, vc


def get_step_fn(stage: str):
    """Return the step function for stage 'C0', 'C1', or 'C2'."""
    return {"C0": c0_step, "C1": c1_step, "C2": c2_step}.get(stage, c0_step)
