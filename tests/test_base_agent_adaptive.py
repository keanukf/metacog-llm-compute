"""Tests for run_adaptive_episode."""

from __future__ import annotations

from pathlib import Path

from src.agent.allocation_policy import load_policy
from src.agent.base_agent import run_adaptive_episode

POLICY = load_policy(
    Path(__file__).resolve().parent / "fixtures" / "policy_artifact_v1.json",
    domain="textworld",
    signal="tle_mean_entropy",
)


class _LoopEnv:
    """Terminates after ``n`` steps."""

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._i = 0
        self.observation = ""
        self.done = False
        self.task_success = False
        self.step_results: list[dict] = []

    def reset(self) -> str:
        self._i = 0
        self.done = False
        self.observation = "start"
        self.step_results = []
        return self.observation

    def step(self, action: str) -> str:
        self._i += 1
        self.observation = f"obs_{self._i}"
        if self._i >= self._n:
            self.done = True
        return self.observation


class _StubModel:
    def generate(self, prompt, logprobs=False, **kwargs):
        text = "noop"
        lp = [{"logprob": -0.5, "top_logprobs": [{"logprob": -0.5}] * 20}] * 2 if logprobs else None
        return text, lp


def test_run_adaptive_always_c0_stages():
    env = _LoopEnv(n=3)
    model = _StubModel()
    r = run_adaptive_episode(env, model, "always_c0", max_steps=10)
    assert r["steps"] == 3
    assert r["stage_per_step"] == ["C0", "C0", "C0"]


def test_run_adaptive_eager_style_episode_fixed_via_policy():
    env = _LoopEnv(n=5)
    model = _StubModel()
    r = run_adaptive_episode(env, model, "eager_style", max_steps=10, policy=POLICY, vc_mode="none")
    assert r["stage_per_step"][0] == "C0"
    fixed = r["stage_per_step"][1]
    assert r["stage_per_step"][1:] == [fixed] * (len(r["stage_per_step"]) - 1)


def test_run_adaptive_step_correctness_is_detached_copy():
    from src.environments.textworld_env import TextWorldEnv

    tw = TextWorldEnv(max_steps=2)
    model = _StubModel()
    r = run_adaptive_episode(tw, model, "always_c0", max_steps=2)
    assert r["step_correctness"] is not None
    assert len(r["step_correctness"]) >= 1
    orig = tw.step_results[0]["correctness"]
    r["step_correctness"][0]["correctness"] = "mutated"
    assert tw.step_results[0]["correctness"] == orig
