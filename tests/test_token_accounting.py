"""Token accounting consistency (thesis §5.3)."""

from __future__ import annotations

from src.agent.base_agent import run_episode
from src.agent.compute_stages import get_step_fn


class _OneStepEnv:
    def reset(self):
        self.done = False
        self.task_success = False
        self.observation = "start"
        return self.observation

    def step(self, action):
        self.done = True
        self.observation = "end"
        return self.observation


class _TokenModel:
    def __init__(self, n_tokens: int = 3):
        self.n = n_tokens

    def generate(self, prompt, logprobs=False, **kwargs):
        lp = (
            [{"logprob": -0.5, "top_logprobs": [{"logprob": -0.5}] * 20}] * self.n
            if logprobs
            else None
        )
        return "go north", lp


def test_c0_episode_tokens_match_subcall_sum():
    env = _OneStepEnv()
    model = _TokenModel(n_tokens=4)
    step_fn = get_step_fn("C0", vc_mode="none")
    r = run_episode(env, model, "C0", step_fn=step_fn, max_steps=5)
    assert r["total_tokens_generated"] >= r["steps_detail"][0].get("tokens_generated", 0)
