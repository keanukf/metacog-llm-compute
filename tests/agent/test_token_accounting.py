"""Token-accounting consistency across compute stages (thesis §5.3).

Verifies the per-episode token total equals the sum of its sub-call tokens for C0, C1 and C2,
that the VC follow-up call's tokens are counted, and that a VC retry doubles the follow-up calls.
Token count is the compute cost that denominates every RQ2 performance/compute trade-off, so any
under- or double-counting here would bias the central adaptive-vs-fixed comparison.
"""

from __future__ import annotations

from src.agent.base_agent import run_episode
from src.agent.compute_stages import VC_FOLLOWUP_PROMPT_MARKER, get_step_fn
from src.utils.inference.generate_result import GenerateResult


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
        text = "<think>\nthink\n</think>\ngo north" if kwargs.get("enable_thinking") else "go north"
        return text, lp


class _C2ManyModel:
    def generate_many(self, prompt, n=3, logprobs=False, **kwargs):
        return [("go north", [{"logprob": -0.5}] * 4) for _ in range(n)]


class _C1VcTokenModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, logprobs=False, **kwargs):
        self.calls += 1
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            return "70", [{"logprob": -0.1}] * 3 if logprobs else None
        return "<think>\nthink\n</think>\ngo north", [{"logprob": -0.5}] * 7 if logprobs else None


class _VcRetryEpisodeModel:
    def __init__(self) -> None:
        self.followup_calls = 0

    def generate(self, prompt, logprobs=False, **kwargs):
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            self.followup_calls += 1
            if self.followup_calls == 1:
                return "unparseable garbage", None
            return "80", [{"logprob": -0.1}] * 2 if logprobs else None
        return "go north", [{"logprob": -0.5}] * 4 if logprobs else None


class _PromptTokenModel:
    """Backend that reports a real prompt-token count via GenerateResult, matching what
    ServerBackend now does against a real vLLM usage block (P1-stat-7)."""

    def __init__(self, prompt_tokens: int = 100):
        self.prompt_tokens = prompt_tokens

    def generate(self, prompt, logprobs=False, **kwargs):
        lp = [{"logprob": -0.5}] * 3 if logprobs else None
        text = "<think>\nthink\n</think>\ngo north" if kwargs.get("enable_thinking") else "go north"
        return GenerateResult(text, lp, prompt_tokens=self.prompt_tokens)

    def generate_many(self, prompt, n=3, logprobs=False, **kwargs):
        return [
            GenerateResult("go north", [{"logprob": -0.5}] * 4, prompt_tokens=self.prompt_tokens)
            for _ in range(n)
        ]


def _assert_episode_token_sum(r: dict) -> None:
    step_sum = sum(int(sd.get("tokens_generated") or 0) for sd in r["steps_detail"])
    assert r["total_tokens_generated"] == step_sum


def _assert_episode_prompt_token_sum(r: dict) -> None:
    step_sum = sum(int(sd.get("prompt_tokens") or 0) for sd in r["steps_detail"])
    assert r["total_prompt_tokens"] == step_sum


def test_c0_episode_tokens_match_subcall_sum():
    env = _OneStepEnv()
    model = _TokenModel(n_tokens=4)
    step_fn = get_step_fn("C0", vc_mode="none")
    r = run_episode(env, model, "C0", step_fn=step_fn, max_steps=5)
    _assert_episode_token_sum(r)


def test_c1_episode_tokens_match_subcall_sum():
    env = _OneStepEnv()
    model = _TokenModel(n_tokens=5)
    step_fn = get_step_fn("C1", vc_mode="none")
    r = run_episode(env, model, "C1", step_fn=step_fn, max_steps=5)
    _assert_episode_token_sum(r)
    assert sum(int(sd.get("tokens_generated") or 0) for sd in r["steps_detail"]) >= 5


def test_c2_episode_tokens_match_subcall_sum():
    env = _OneStepEnv()
    model = _C2ManyModel()
    step_fn = get_step_fn("C2", vc_mode="none", c2_n_samples=3)
    r = run_episode(env, model, "C2", step_fn=step_fn, max_steps=5)
    _assert_episode_token_sum(r)
    assert sum(int(sd.get("tokens_generated") or 0) for sd in r["steps_detail"]) == 12


def test_c1_vc_followup_tokens_included():
    env = _OneStepEnv()
    model = _C1VcTokenModel()
    step_fn = get_step_fn(
        "C1", vc_mode="followup", prompt_prefix="Prefix.", save_vc_distributions=True
    )
    r = run_episode(env, model, "C1", step_fn=step_fn, max_steps=5)
    _assert_episode_token_sum(r)
    assert model.calls == 2
    assert r["steps_detail"][0]["tokens_generated"] == 10


def test_vc_retry_doubles_followup_calls():
    env = _OneStepEnv()
    model = _VcRetryEpisodeModel()
    step_fn = get_step_fn("C0", vc_mode="followup", vc_retry_on_parse_failure=True)
    r = run_episode(env, model, "C0", step_fn=step_fn, max_steps=5)
    _assert_episode_token_sum(r)
    lm_calls = r["steps_detail"][0].get("lm_calls") or r["steps_detail"][0].get(
        "lm_calls_this_step"
    )
    assert lm_calls == 3
    assert model.followup_calls == 2


def test_c0_episode_prompt_tokens_match_subcall_sum():
    """P1-stat-7 regression: total_prompt_tokens must equal the sum of per-step prompt_tokens,
    the same invariant already enforced for output tokens above."""
    env = _OneStepEnv()
    model = _PromptTokenModel(prompt_tokens=100)
    step_fn = get_step_fn("C0", vc_mode="none")
    r = run_episode(env, model, "C0", step_fn=step_fn, max_steps=5)
    _assert_episode_prompt_token_sum(r)
    assert r["total_prompt_tokens"] > 0


def test_c2_episode_prompt_tokens_booked_per_candidate_not_shared():
    """C2 draws 3 candidates per step; a real batched vLLM request reports one shared
    prompt_tokens value, attached to each candidate (test_generate_many_batched_attaches_shared_
    prompt_tokens_to_each_candidate in test_execution_server.py) -- reasoning_step_core must then
    sum it per-candidate, symmetric with how it already sums output tokens per-candidate."""
    env = _OneStepEnv()
    model = _PromptTokenModel(prompt_tokens=50)
    step_fn = get_step_fn("C2", vc_mode="none", c2_n_samples=3)
    r = run_episode(env, model, "C2", step_fn=step_fn, max_steps=5)
    _assert_episode_prompt_token_sum(r)
    # One step, 3 candidates, 50 prompt tokens booked per candidate -> 150 for that step.
    assert r["steps_detail"][0]["prompt_tokens"] == 150


def test_prompt_tokens_default_to_zero_when_backend_cannot_report_them():
    """Every other test in this file uses a plain-tuple-returning mock (no GenerateResult) --
    confirms that path (still the overwhelming majority of the test suite) yields exactly 0, not
    an error, keeping the whole feature purely additive."""
    env = _OneStepEnv()
    model = _TokenModel(n_tokens=4)
    step_fn = get_step_fn("C0", vc_mode="none")
    r = run_episode(env, model, "C0", step_fn=step_fn, max_steps=5)
    assert r["total_prompt_tokens"] == 0
    assert all(int(sd.get("prompt_tokens") or 0) == 0 for sd in r["steps_detail"])
