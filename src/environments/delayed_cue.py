"""
Delayed-Cue task generator. Produces task instances (e.g. list of dicts).
"""
from __future__ import annotations

from typing import Any


def generate_tasks(n: int, seed: int | None = None) -> list[dict[str, Any]]:
    """
    Generate n delayed-cue task instances.

    Each instance is a dict with at least: id, prompt or context, expected_answer (optional),
    and any fields needed to run the task in an environment.

    Args:
        n: Number of instances.
        seed: Optional RNG seed.

    Returns:
        List of task instance dicts.
    """
    if seed is not None:
        import random
        random.seed(seed)
    return [
        {"id": f"delayed_cue_{i}", "prompt": f"Task {i} placeholder", "expected": None}
        for i in range(n)
    ]
