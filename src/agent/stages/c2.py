from __future__ import annotations

from typing import Any


def c2_step(observation: str, history: list[str], model: Any, **kwargs: Any):
    """C2 stage wrapper kept in a stage-specific module."""
    from src.agent import compute_stages

    return compute_stages.c2_step(observation, history, model, **kwargs)
