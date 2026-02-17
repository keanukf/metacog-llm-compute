"""
Verbalized Confidence (VC): extract numeric confidence 0-100 from model output.
Prompt: "Answer, then rate your confidence 0-100."
Fallback: few-shot examples in prompt; robust parsing here.
"""
from __future__ import annotations

import re


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
    if not text or not text.strip():
        return None
    text = text.strip()
    # Common patterns
    patterns = [
        r"(?:confidence|conf)\s*[:\s]+\s*(\d+(?:\.\d+)?)\s*%?",  # confidence: 85 or 85%
        r"(?:0-100|0–100)\s*[:\s]+\s*(\d+(?:\.\d+)?)",
        r"\b(\d+(?:\.\d+)?)\s*%\s*(?:confidence)?",
        r"(?:rate|rating)\s*(?:is)?\s*(\d+(?:\.\d+)?)",
        r"\b(\d{1,3})\s*/\s*100",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val
    # Last resort: any number in [0, 100] in the last 100 chars
    tail = text[-100:] if len(text) > 100 else text
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", tail)
    for n in reversed(nums):
        val = float(n)
        if 0 <= val <= 100:
            return val
    return None
