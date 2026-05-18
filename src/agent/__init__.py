"""
Agent public API.

The package keeps `base_agent` as the stable facade while internal helpers are
split into focused modules (`history_utils`, `step_results`, `trace_integration`,
`episode_runner`).
"""

from src.agent.allocator import allocate
from src.agent.base_agent import run_adaptive_episode, run_episode
from src.agent.compute_stages import get_step_fn

__all__ = [
    "allocate",
    "get_step_fn",
    "run_adaptive_episode",
    "run_episode",
]
