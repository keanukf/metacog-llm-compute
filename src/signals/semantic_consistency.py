"""
Semantic Consistency (SC) — Extension.
Requires 5x sampling per step; compare consistency across samples.
Stub only.
"""
from __future__ import annotations

from typing import Any


def compute_semantic_consistency(
    samples: list[str],
    **kwargs: Any,
) -> float:
    """
    Compute a scalar consistency score over multiple model samples for the same prompt.
    Extension: not used in core pilot.

    Args:
        samples: List of generated strings (e.g. 5 samples per step).

    Returns:
        Placeholder: 0.0 until implemented.
    """
    return 0.0
