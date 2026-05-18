from __future__ import annotations

from typing import Any


def c1_step(observation: str, history: list[str], model: Any, **kwargs: Any):
    """C1 stage wrapper kept in a stage-specific module."""
    from src.agent import compute_stages

    return compute_stages.c1_step(observation, history, model, **kwargs)
