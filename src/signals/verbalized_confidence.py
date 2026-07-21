"""
Verbalized Confidence (VC) parsing -- the study's second metacognitive signal (Factor 1), the
"gray-box" Feeling-of-Knowing proxy that H1 predicts is *less* well calibrated than TLE because
RLHF biases it toward overconfidence.

VC is elicited in a separate follow-up call after the step action is committed (see
``src.agent.stages.shared._run_vc_followup``), asking how likely the chosen action is to be correct
as a single integer 0-100 with explicitly anchored scale ends (0 = certainly wrong, 100 = certainly
correct; Yang et al., 2024). No few-shot number examples are given -- for small models those bias
the distribution -- so the model's format compliance is imperfect. This module therefore does the
robust extraction: normalize a bare ``Confidence:`` echo, then try a cascade of numeric patterns and
report which one matched (``vc_pattern_matched``) for transparency in the calibration analysis.
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


def _normalize_vc_completion_text(text: str) -> str:
    """
    Normalize VC follow-up completions before parsing.

    Long task-context prompts sometimes elicit a bare ``Confidence:`` label echo
    (no integer). Multiline ``Confidence:\\n90`` is collapsed to the numeric tail.
    """
    t = (text or "").strip()
    if not t:
        return t
    if re.fullmatch(r"(?i)confidence\s*:?\s*", t):
        return ""
    multiline = re.match(
        r"(?i)confidence\s*:\s*\n+\s*(\d{1,3}(?:\.\d+)?)\s*$",
        t,
    )
    if multiline:
        return multiline.group(1)
    return t


def parse_confidence_with_meta(text: str) -> tuple[float | None, str | None]:
    """
    Like ``parse_confidence`` but also returns which pattern matched (for transparency).

    Returns:
        (value 0-100 or None, pattern name or None).
    """
    text = _normalize_vc_completion_text(text or "")
    if not text:
        return None, None
    text = text.strip()
    patterns: list[tuple[str, str]] = [
        ("bare_number", r"^\s*(\d{1,3}(?:\.\d+)?)\s*$"),
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
