"""Seeds the per-episode history list with the reset observation.

Split out because getting this right is subtle: many environments (TextWorld especially) print the
full scene only once at reset and then emit just deltas, so if the opening text is not retained the
step >=1 prompts silently lose it.
"""

from __future__ import annotations

from src.agent.history_utils import truncate_for_history


def build_initial_history(
    observation: str,
    *,
    history_max_obs_chars: int,
    history_obs_head_ratio: float,
) -> list[str]:
    """
    Build initial history with reset observation preserved for subsequent prompts.
    """
    return [
        "OBSERVATION: "
        + truncate_for_history(
            observation,
            max_chars=history_max_obs_chars,
            head_ratio=history_obs_head_ratio,
        )
    ]
