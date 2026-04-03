"""Compute stages: VC follow-up and prompt prefix."""
from __future__ import annotations

from src.agent.compute_stages import get_step_fn


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, logprobs: bool = False, **kwargs):
        self.calls += 1
        if "You just chose the action:" in (prompt or ""):
            return "65", [{"logprob": -0.1}] * 2 if logprobs else None
        return "A->C", [{"logprob": -0.5}] * 3 if logprobs else None


def test_followup_triggers_two_lm_calls_per_step():
    m = _CountingModel()
    step = get_step_fn("C0", vc_mode="followup", prompt_prefix="Test prefix.")
    step("obs", [], m)
    assert m.calls == 2


def test_inline_single_call():
    m = _CountingModel()
    step = get_step_fn("C0", vc_mode="inline")
    step("Confidence: 50\n", [], m)
    assert m.calls == 1


def test_c2_followup_adds_call_after_samples():
    m = _CountingModel()
    step = get_step_fn("C2", vc_mode="followup")
    step("obs", [], m)
    # 3 samples + 1 VC follow-up
    assert m.calls == 4
