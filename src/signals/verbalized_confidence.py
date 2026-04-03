"""
Verbalized Confidence (VC): extract numeric confidence 0-100 from model output.
Prompt: "Answer, then rate your confidence 0-100."
Fallback: few-shot examples in prompt; robust parsing here.
"""
from __future__ import annotations

import re
from typing import Any


def parse_confidence(text: str) -> float | None:
    """
    Parse a numeric confidence value in 0-100 from model output.

    Looks for patterns like "confidence: 85", "0-100: 70", "Confidence: 85",
    or a number in [0, 100] near the end of the string.

    Args:
        text: Raw model output (answer + optional confidence statement).

    Returns:
        Float in [0, 100] or None if unparseable.
    """
    val, _ = parse_confidence_with_meta(text)
    return val


def parse_confidence_with_meta(text: str) -> tuple[float | None, str | None]:
    """
    Like ``parse_confidence`` but also returns which pattern matched (for transparency).

    Returns:
        (value 0-100 or None, pattern name or None).
    """
    if not text or not text.strip():
        return None, None
    text = text.strip()
    patterns: list[tuple[str, str]] = [
        ("confidence_label", r"(?:confidence|conf)\s*[:\s]+\s*(\d+(?:\.\d+)?)\s*%?"),
        ("0_100_label", r"(?:0-100|0–100)\s*[:\s]+\s*(\d+(?:\.\d+)?)"),
        ("percent_suffix", r"\b(\d+(?:\.\d+)?)\s*%\s*(?:confidence)?"),
        ("rate_rating", r"(?:rate|rating)\s*(?:is)?\s*(\d+(?:\.\d+)?)"),
        ("out_of_100", r"\b(\d{1,3})\s*/\s*100"),
    ]
    for name, pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val, name
    tail = text[-100:] if len(text) > 100 else text
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", tail)
    for n in reversed(nums):
        val = float(n)
        if 0 <= val <= 100:
            return val, "tail_fallback"
    return None, None


def extract_vc_from_followup(
    vc_prompt: str,
    raw_text: str,
    logprobs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Build a structured VC record from a dedicated confidence follow-up call.

    Args:
        vc_prompt: Prompt sent for the VC-only generation.
        raw_text: Model completion text.
        logprobs: Per-token logprob rows from the model (optional).

    Returns:
        Dict with vc_value, vc_raw_text, vc_prompt, vc_tokens_used, vc_logprobs, vc_pattern_matched.
    """
    val, pat = parse_confidence_with_meta(raw_text)
    return {
        "vc_value": val,
        "vc_raw_text": (raw_text or "").strip(),
        "vc_prompt": vc_prompt,
        "vc_tokens_used": len(logprobs) if logprobs else 0,
        "vc_logprobs": logprobs,
        "vc_pattern_matched": pat,
    }
