"""Adaptive-allocation episode loop (``run_adaptive_episode``), the Phase 2 (RQ2/RQ4) path.

Verifies the loop honours the frozen allocation policy: always-C0 when the policy says so, a
per-episode fixed stage under eager-style policies, and a C2 call-index that increments across
steps so token accounting and traces stay attributable. The detached-copy check guards against
step-correctness aliasing that would silently corrupt the DV recorded per step.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.allocation_policy import load_policy
from src.agent.base_agent import run_adaptive_episode

POLICY = load_policy(
    Path(__file__).resolve().parent.parent / "fixtures" / "policy_artifact_v1.json",
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


def test_run_adaptive_c2_call_index_increments_across_episode_steps(monkeypatch):
    """Regression: the step function used to be rebuilt fresh every step, so the C2
    tie-break RNG's call_index counter (nonlocal to get_step_fn) reset to 0 on every
    step instead of advancing across the episode. Fixed by caching the step function
    per stage for the lifetime of the episode.
    """
    import src.agent.compute_stages as compute_stages

    captured_indices: list[int] = []

    def fake_c2_step_core(obs, hist, m, *, call_index, **kwargs):
        captured_indices.append(call_index)
        return ("noop", None, None, 1, 1, None, None, "", "", None)

    monkeypatch.setattr(compute_stages, "c2_step_core", fake_c2_step_core)

    env = _LoopEnv(n=3)
    model = _StubModel()
    r = run_adaptive_episode(env, model, "always_c2", max_steps=10)

    assert r["stage_per_step"] == ["C2", "C2", "C2"]
    assert captured_indices == [0, 1, 2]


def test_run_adaptive_tle_logs_allocator_uncertainty_score():
    """2026-08-04: the policy's continuous uncertainty_score (computed and previously discarded
    inside AllocationPolicy.stage()) must now be retained per step -- Ch.7.4 ("Allocation
    Patterns") needs the score, not just the discrete stage it collapsed to."""
    env = _LoopEnv(n=4)
    model = _StubModel()
    r = run_adaptive_episode(
        env, model, "adaptive_tle", max_steps=10, policy=POLICY, vc_mode="none"
    )
    steps_detail = r["steps_detail"]

    # Step 0: signal is None (first-step rule), so no policy lookup happens -- no score logged.
    assert "allocator_uncertainty_score" not in steps_detail[0]

    # _StubModel's C2 path (unlike C0) doesn't return a real tle dict, so a step whose *previous*
    # step ran C2 gets signal=None and no score logged for that step -- a quirk of this minimal
    # test double, not of the production entropy computation (a real LM call always populates
    # tle). At least one step with a real incoming signal must still show a well-formed score.
    scored_rows = [row for row in steps_detail[1:] if "allocator_uncertainty_score" in row]
    assert scored_rows, "expected at least one step with a logged allocator_uncertainty_score"
    for row in scored_rows:
        assert 0.0 <= row["allocator_uncertainty_score"] <= 1.0
        assert row["allocator_theta1"] == POLICY.theta1
        assert row["allocator_theta2"] == POLICY.theta2


def test_run_adaptive_always_c0_never_logs_allocator_score():
    """Baselines that never touch the policy (always_c0/always_c2/random) must not log a score --
    there was no policy-driven decision to explain."""
    env = _LoopEnv(n=3)
    model = _StubModel()
    r = run_adaptive_episode(env, model, "always_c0", max_steps=10, policy=POLICY, vc_mode="none")
    for row in r["steps_detail"]:
        assert "allocator_uncertainty_score" not in row


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
