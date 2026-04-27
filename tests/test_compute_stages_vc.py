"""Compute stages: VC follow-up and prompt prefix."""
from __future__ import annotations

from src.agent.compute_stages import VC_FOLLOWUP_PROMPT_MARKER, _extract_first_line, get_step_fn


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, logprobs: bool = False, **kwargs):
        self.calls += 1
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            return "65", [{"logprob": -0.1}] * 2 if logprobs else None
        return "A->C", [{"logprob": -0.5}] * 3 if logprobs else None


def test_followup_triggers_two_lm_calls_per_step():
    m = _CountingModel()
    step = get_step_fn("C0", vc_mode="followup", prompt_prefix="Test prefix.")
    step("obs", [], m)
    assert m.calls == 2


def test_followup_vc_prompt_contains_task_context():
    class _Cap:
        vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "60", [{"logprob": -0.1}] * 2 if logprobs else None
            return "north", [{"logprob": -0.5}] * 3 if logprobs else None

    m = _Cap()
    step = get_step_fn("C0", vc_mode="followup", prompt_prefix="Scoped prefix line.")
    step("current_obs", [], m)
    assert m.vc_prompt is not None
    assert "Scoped prefix line." in m.vc_prompt
    assert "=== TASK CONTEXT ===" in m.vc_prompt
    assert "current_obs" in m.vc_prompt


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


def test_extract_first_line_returns_first_non_empty():
    assert _extract_first_line("go north\nextra") == "go north"
    assert _extract_first_line("\n\n  take key  \n") == "take key"
    assert _extract_first_line("") == ""


class _C1TwoCallModel:
    """CoT call then verify (both use logprobs in real C1); optional VC follow-up."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, logprobs: bool = False, **kwargs):
        self.calls += 1
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            return "70", [{"logprob": -0.1}] * 2 if logprobs else None
        if "Self-check" in (prompt or ""):
            return "go north", [{"logprob": -0.3}] * 4 if logprobs else None
        return "Consider the map.\nACTION: go north", [{"logprob": -0.5}] * 8 if logprobs else None


def test_c1_two_lm_calls_with_followup_vc():
    m = _C1TwoCallModel()
    step = get_step_fn("C1", vc_mode="followup", prompt_prefix="Prefix.")
    action, _tle, vc, _tok, lm_calls, *_rest = step("obs", [], m)
    assert action == "go north"
    assert lm_calls == 3
    assert m.calls == 3
    assert vc == 70.0


def test_c1_vc_followup_includes_chain_of_thought():
    """C1 VC must judge CoT + final action, not action alone (cross-stage comparability)."""

    class _M:
        vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "55", [{"logprob": -0.1}] * 2 if logprobs else None
            if "Self-check" in (prompt or ""):
                return "go east", [{"logprob": -0.2}] * 3 if logprobs else None
            return "MarkerReasoningUnique42.\nACTION: go east", [{"logprob": -0.15}] * 10 if logprobs else None

    m = _M()
    step = get_step_fn("C1", vc_mode="followup")
    step("obs", [], m)
    assert m.vc_prompt is not None
    assert "Chain-of-thought (your reasoning for this turn)" in m.vc_prompt
    assert "MarkerReasoningUnique42." in m.vc_prompt


def test_c1_two_lm_calls_vc_none():
    class _NoVc:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            if "Self-check" in (prompt or ""):
                return "take key", [{"logprob": -0.2}] * 3 if logprobs else None
            return "Plan.\nACTION: take key", [{"logprob": -0.4}] * 8 if logprobs else None

    m = _NoVc()
    step = get_step_fn("C1", vc_mode="none", prompt_prefix="P.")
    action, _tle, vc, _tok, lm_calls, *_ = step("obs", [], m)
    assert action == "take key"
    assert lm_calls == 2
    assert m.calls == 2
    assert vc is None


def test_c0_returns_first_line_as_action():
    class _MultiLineModel:
        calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                return "65", [{"logprob": -0.1}] * 2 if logprobs else None
            return "go south\nI think this is right.", [{"logprob": -0.5}] * 5 if logprobs else None

    m = _MultiLineModel()
    step = get_step_fn("C0", vc_mode="followup", prompt_prefix="Prefix.")
    action, *_rest = step("obs", [], m)
    assert action == "go south"
    assert m.calls == 2
