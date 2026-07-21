"""Gate D difficulty metrics: production move-cap derivation and success@Cap.

Verifies success@Cap from observed runs, the production cap as the observed p90 plus a margin, and
that an episode's win-step is read from task success. The move cap is the Gate D difficulty
calibration that freezes how hard each task instance is; deriving it from the observed p90 keeps
the task sets neither trivially easy nor unsolvable, which is what makes correctness a discriminating
DV across domains (RQ4).
"""

from __future__ import annotations

from scripts.difficulty_calibration.difficulty_metrics import (
    derive_production_cap,
    episode_record,
    success_rate_at_cap,
    success_rate_at_obs,
)


def test_success_at_cap_from_observed_runs():
    episodes = [
        {"task_success": True, "win_step": 18, "episode_length_steps": 18, "truncated": False},
        {"task_success": True, "win_step": 22, "episode_length_steps": 22, "truncated": False},
        {"task_success": False, "win_step": None, "episode_length_steps": 25, "truncated": True},
    ]
    assert success_rate_at_obs(episodes) == 2 / 3
    assert success_rate_at_cap(episodes, 20) == 1 / 3
    assert success_rate_at_cap(episodes, 25) == 2 / 3


def test_derive_production_cap_p90_plus_margin():
    win_steps = [10, 12, 14, 16, 18, 20, 22]
    cap = derive_production_cap(win_steps, margin=2)
    assert cap is not None
    assert cap >= 22


def test_episode_record_win_step_from_task_success():
    result = {
        "task_success": True,
        "episode_length_steps": 15,
        "step_correctness": [{"step_index": 14, "won": True}],
    }
    rec = episode_record(result, obs_ceiling=25)
    assert rec["win_step"] == 15
    assert rec["truncated"] is False
