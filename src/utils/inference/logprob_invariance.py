"""Temperature-invariance diagnostics for raw vLLM logprobs (thesis §5.7)."""

from __future__ import annotations

import math
from typing import Any

from src.signals.token_entropy import entropy_shannon_from_top_logprobs

# Preregistered floor (bits); RunPod control run 2026-07-04 on Qwen3-8B / vLLM 0.19.1.
TLE_INVARIANCE_EPS_BITS = 0.05
TLE_INVARIANCE_NOISE_SAFETY_FACTOR = 3.0

_CAPABILITY_PROBE_PROMPT = "Reply with one word: north"


def first_token_tle(logprobs: list[dict[str, Any]] | None) -> float | None:
    """Shannon TLE (bits) on the first token's top-k distribution."""
    if not logprobs or not isinstance(logprobs[0], dict):
        return None
    top = logprobs[0].get("top_logprobs")
    if not isinstance(top, list) or not top:
        return None
    return float(entropy_shannon_from_top_logprobs(top))


def max_top_logprob_delta(
    top_a: list[dict[str, Any]] | None,
    top_b: list[dict[str, Any]] | None,
) -> float | None:
    """Max absolute logprob delta over shared token keys."""
    if not isinstance(top_a, list) or not isinstance(top_b, list):
        return None
    map_a = {
        str(x.get("token", "")): float(x["logprob"])
        for x in top_a
        if isinstance(x, dict) and x.get("logprob") is not None
    }
    map_b = {
        str(x.get("token", "")): float(x["logprob"])
        for x in top_b
        if isinstance(x, dict) and x.get("logprob") is not None
    }
    keys = set(map_a) & set(map_b)
    if not keys:
        return None
    return max(abs(map_a[k] - map_b[k]) for k in keys)


def predicted_temp_scaling_logprob_span(
    top: list[dict[str, Any]] | None,
    *,
    t_low: float = 0.3,
    t_high: float = 1.0,
) -> float | None:
    """
    Diagnostic lower bound on |Δlogprob| if temperature scaling were applied.

    Uses logprob at ``t_high`` as a logit proxy: processed_lp(t_low) - raw_lp(t_high)
    ≈ logit_i * (1/t_low - 1/t_high) before partition-function terms.
    """
    if not isinstance(top, list) or not top or t_low <= 0 or t_high <= 0:
        return None
    inv_diff = (1.0 / t_low) - (1.0 / t_high)
    if abs(inv_diff) < 1e-12:
        return 0.0
    lps = [float(x["logprob"]) for x in top if isinstance(x, dict) and x.get("logprob") is not None]
    if not lps:
        return None
    return max(abs(lp * inv_diff) for lp in lps)


def resolve_tle_invariance_eps(same_t_dtle_values: list[float]) -> float:
    """Combine preregistered floor with Same-T noise floor."""
    noise_floor = max(same_t_dtle_values) if same_t_dtle_values else 0.0
    dynamic = noise_floor * TLE_INVARIANCE_NOISE_SAFETY_FACTOR
    return max(TLE_INVARIANCE_EPS_BITS, dynamic)


def probe_temperature_invariance(
    model: Any,
    prompt: str,
    *,
    t_low: float = 0.3,
    t_high: float = 1.0,
    max_tokens: int = 8,
) -> dict[str, Any]:
    """
    Run T_low, T_high, and duplicate T_high; return diagnostic metrics for one prompt.
    """
    gen_kw = {"logprobs": True, "enable_thinking": False, "max_tokens": max_tokens}
    _, lp_low = model.generate(prompt, temperature=t_low, **gen_kw)
    _, lp_high = model.generate(prompt, temperature=t_high, **gen_kw)
    _, lp_high_b = model.generate(prompt, temperature=t_high, **gen_kw)

    tle_low = first_token_tle(lp_low)
    tle_high = first_token_tle(lp_high)
    tle_high_b = first_token_tle(lp_high_b)

    top_low = lp_low[0].get("top_logprobs") if lp_low and isinstance(lp_low[0], dict) else None
    top_high = lp_high[0].get("top_logprobs") if lp_high and isinstance(lp_high[0], dict) else None
    top_high_b = (
        lp_high_b[0].get("top_logprobs") if lp_high_b and isinstance(lp_high_b[0], dict) else None
    )

    cross_t_dtle = abs(tle_low - tle_high) if tle_low is not None and tle_high is not None else None
    same_t_dtle = (
        abs(tle_high - tle_high_b) if tle_high is not None and tle_high_b is not None else None
    )

    return {
        "logprobs_t_low": lp_low,
        "tle_t_low": tle_low,
        "tle_t_high": tle_high,
        "tle_t_high_repeat": tle_high_b,
        "cross_t_dtle": cross_t_dtle,
        "same_t_dtle": same_t_dtle,
        "max_logprob_delta_cross_t": max_top_logprob_delta(top_low, top_high),
        "max_logprob_delta_same_t": max_top_logprob_delta(top_high, top_high_b),
        "predicted_scaling_span_bits": (
            None
            if top_high is None
            else _logprob_span_to_entropy_upper_bound(
                predicted_temp_scaling_logprob_span(top_high, t_low=t_low, t_high=t_high)
            )
        ),
        "predicted_scaling_logprob_span": predicted_temp_scaling_logprob_span(
            top_high, t_low=t_low, t_high=t_high
        ),
    }


def _logprob_span_to_entropy_upper_bound(logprob_span: float | None) -> float | None:
    """Conservative diagnostic: map max logprob shift to an entropy shift upper bound."""
    if logprob_span is None:
        return None
    # Renormalized entropy is bounded; use min(span / ln(2), 2.0) as loose diagnostic cap.
    return min(float(logprob_span) / math.log(2), 2.0)


def capability_probe_prompt() -> str:
    return _CAPABILITY_PROBE_PROMPT
