"""
Pilot Test 5 — End-to-End Mini-Experiment.
Mini pipeline: 2 instances x 3 stages x 1 run = 6 episodes with mock env and mock model.
Assert 6 episode outputs with required keys.
"""
from __future__ import annotations

import json

import pytest

from src.agent.base_agent import run_episode
from src.agent.compute_stages import get_step_fn
from src.environments.textworld_env import TextWorldEnv
from src.utils.logging_utils import EpisodeLogger, log_episode


@pytest.fixture
def mock_model():
    class M:
        def generate(self, prompt, logprobs=False, **kwargs):
            return "go north", [{"logprob": -0.5}] * 5 if logprobs else None
    return M()


def test_e2e_produces_episode_dict_with_required_keys(mock_model, temp_results_dir):
    env = TextWorldEnv(max_steps=5)
    step_fn = get_step_fn("C0")
    result = run_episode(env, mock_model, "C0", step_fn=step_fn, max_steps=5)
    required = {"steps", "task_success", "lm_calls", "tokens", "wall_clock_time"}
    for k in required:
        assert k in result, f"missing key {k}"
    assert "tle_per_step" in result or "vc_per_step" in result


def test_e2e_mini_pipeline_6_episodes(mock_model, temp_results_dir):
    """Run 2 instances x 3 stages x 1 run = 6 episodes; write JSON; assert 6 files."""
    logger = EpisodeLogger(temp_results_dir)
    instances = 2
    stages = ["C0", "C1", "C2"]
    run = 1
    episode_ids = []
    for inst in range(instances):
        for stage in stages:
            env = TextWorldEnv(max_steps=5)
            step_fn = get_step_fn(stage)
            result = run_episode(env, mock_model, stage, step_fn=step_fn, max_steps=5)
            ep_id = f"ep_tw_{inst}_{stage}_0"
            episode_ids.append(ep_id)
            data = {
                "episode_id": ep_id,
                "domain": "textworld",
                "instance": inst,
                "compute_stage": stage,
                "run": 0,
                "task_success": result["task_success"],
                "steps": result["steps"],
                "lm_calls": result["lm_calls"],
                "tokens": result["tokens"],
                "wall_clock_time": result["wall_clock_time"],
                "tle_per_step": result.get("tle_per_step"),
                "vc_per_step": result.get("vc_per_step"),
            }
            logger.log(ep_id, data)
    files = list(temp_results_dir.glob("*.json"))
    assert len(files) == 6
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        assert "episode_id" in d and "steps" in d and "lm_calls" in d and "tokens" in d
