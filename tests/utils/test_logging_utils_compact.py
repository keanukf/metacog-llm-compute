"""Compact episode storage (``compact_episode_for_storage``) must not silently drop fields.

``_minimal_steps_detail`` reduces each step record to an explicit allowlist before writing --
adding a new per-step field elsewhere (e.g. base_agent.py) is silently discarded here unless the
allowlist is updated too. This caught exactly that gap for ``prompt_tokens`` (P1-stat-7): the field
was computed and put on the step row, but compact storage (the production default) stripped it
back out before it ever reached disk.
"""

from __future__ import annotations

from src.utils.logging_utils import compact_episode_for_storage


def test_compact_storage_retains_prompt_tokens_per_step():
    data = {
        "episode_id": "ep_x",
        "steps_detail": [
            {
                "step_index": 0,
                "compute_stage": "C0",
                "tle": {"mean_entropy": 0.1},
                "vc": 80.0,
                "tokens_generated": 3,
                "prompt_tokens": 120,
                "lm_calls_this_step": 1,
                "step_wall_time_s": 0.5,
                "correctness": "optimal",
            }
        ],
        "total_prompt_tokens": 120,
    }
    out = compact_episode_for_storage(data)
    assert out["steps_detail"][0]["prompt_tokens"] == 120
    # Episode-level total is a top-level key, never touched by the per-step allowlist.
    assert out["total_prompt_tokens"] == 120
