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
