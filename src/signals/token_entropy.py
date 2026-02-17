"""
Token-Level Entropy (TLE) from vLLM logprobs.
H(t) = -sum p(x_i) log p(x_i) over output tokens; aggregate via mean/max.
High entropy signals uncertainty (processing fluency analogue).
"""
from __future__ import annotations

import math
from typing import Any


def compute_tle(logprobs: list[dict[str, Any]] | list[float]) -> dict[str, float]:
    """
    Compute token-level entropy from a list of token logprobs.

    If logprobs are dicts (e.g. vLLM format), each item should have 'logprob' or
    a distribution; we derive probs and then H = -sum p log p.

    Args:
        logprobs: Either list of per-token log probabilities (float), or list of
            dicts with 'logprob' key.

    Returns:
        Dict with 'mean_entropy' and 'max_entropy' over the sequence.
    """
    probs: list[float] = []
    for x in logprobs:
        if isinstance(x, dict):
            lp = x.get("logprob", 0.0)
            probs.append(math.exp(lp) if lp <= 0 else 0.0)
        else:
            p = math.exp(float(x)) if float(x) <= 0 else 0.0
            probs.append(p)
    if not probs:
        return {"mean_entropy": 0.0, "max_entropy": 0.0}
    entropies = []
    for p in probs:
        if p <= 0 or p >= 1:
            entropies.append(0.0)
        else:
            entropies.append(-p * math.log2(p) - (1 - p) * math.log2(1 - p))
    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_entropy": max(entropies),
    }


def extract_tle_from_response(
    text: str,
    logprobs: list[dict[str, Any]] | list[float] | None,
) -> dict[str, float] | None:
    """
    Extract TLE from model response. If logprobs are missing, returns None.

    Args:
        text: Generated text (for length checks if needed).
        logprobs: Token logprobs from model_wrapper.generate(..., logprobs=True).

    Returns:
        compute_tle(logprobs) or None.
    """
    if logprobs is None:
        return None
    return compute_tle(logprobs)
