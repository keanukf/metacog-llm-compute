from __future__ import annotations

from typing import Any


def c0_step(observation: str, history: list[str], model: Any, **kwargs: Any):
    """C0 stage wrapper kept in a stage-specific module."""
    from src.agent import compute_stages

    return compute_stages.c0_step(observation, history, model, **kwargs)
