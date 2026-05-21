"""
Shared fixtures for pilot tests: mock_model, mock_env, sample_episode_data, temp_results_dir.
Ensures repo root is on sys.path so 'src' imports work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def mock_model():
    """Model-like object with generate(prompt, logprobs=False) returning (text, logprobs_or_none)."""

    class MockModel:
        def generate(self, prompt, logprobs=False, max_tokens=256, temperature=0.3, **kwargs):
            text = "go north"
            if logprobs:
                lp = [{"logprob": -0.5}] * 5
                return text, lp
            return text, None

    return MockModel()


@pytest.fixture
def mock_env():
    """Env with reset() and step(action); .observation and .done."""

    class MockEnv:
        def __init__(self):
            self.observation = ""
            self.done = False
            self._steps = 0

        def reset(self):
            self.done = False
            self._steps = 0
            self.observation = "You are in a room. Exits: north."
            return self.observation

        def step(self, action):
            self._steps += 1
            self.done = self._steps >= 3
            self.observation = "You moved." if not self.done else "Done."
            return self.observation

    return MockEnv()


@pytest.fixture
def sample_episode_data():
    """One episode dict matching pilot_calibration schema."""
    return {
        "episode_id": "ep_textworld_0_C0_0",
        "domain": "textworld",
        "instance": 0,
        "compute_stage": "C0",
        "run": 0,
        "task_success": True,
        "steps": 5,
        # Legacy fields
        "lm_calls": 5,
        "tokens": 1000,
        # New explicit fields
        "episode_length_steps": 5,
        "total_lm_calls": 5,
        "total_tokens_generated": 1000,
        "normalized_compute_cost": 5 / (20 * 3),
        "efficiency_score": 1.0 / (5 / (20 * 3)),
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "wall_clock_time": 12.5,
        "tle_per_step": [{"mean_entropy": 0.3, "max_entropy": 0.5}],
        "vc_per_step": [85.0, 70.0],
        "steps_detail": [
            {
                "step_index": 0,
                "compute_stage": "C0",
                "action": "go north",
                "tokens_generated": 5,
                "lm_calls_this_step": 1,
                "step_wall_time_s": 0.1,
                "tle": {"mean_entropy": 0.3, "max_entropy": 0.5},
                "vc": 85.0,
                "correctness": "legal",
                "observation_length_chars": 42,
            }
        ],
    }


@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary directory for JSON outputs."""
    return tmp_path
