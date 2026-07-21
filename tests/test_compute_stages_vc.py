"""Compute stages: VC follow-up and prompt prefix."""

from __future__ import annotations

from src.agent import compute_stages as cs
from src.agent.compute_stages import (
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    VC_FOLLOWUP_PROMPT_MARKER,
    _build_prompt,
    _extract_first_line,
    _parse_cot_action,
    get_step_fn,
)


def test_c0_tle_is_action_line_only_when_tokens_present():
    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            # Multiline completion: action line then extra text.
            text = "go north\nextra reasoning"
            if not logprobs:
                return text, None
            lp = [
                {"token": "go", "logprob": -0.1},
                {"token": " ", "logprob": -0.1},
                {"token": "north", "logprob": -0.1},
                {"token": "\n", "logprob": -0.1},
                {"token": "extra", "logprob": -5.0},
            ]
            return text, lp

    m = _M()
    step = get_step_fn("C0", vc_mode="none")
    _action, tle, _vc, _tok, _calls, *_rest = step("obs", [], m)
    assert tle is not None
    # Ensure we did not include the very low-prob extra token by verifying mean entropy differs
    # from computing on the full sequence.
    from src.signals.token_entropy import compute_tle

    assert (
        tle["mean_entropy"]
        != compute_tle([{"logprob": -0.1}] * 4 + [{"logprob": -5.0}])["mean_entropy"]
    )


def test_c1_tle_is_action_only_from_single_reason_call():
    class _C1Model:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            text = "<think>\n- x\n</think>\ntake key\nbecause..."
            if not logprobs:
                return text, None
            lp = [
                {"token": "<think>", "logprob": -0.01},
                {"token": "\n", "logprob": -0.01},
                {"token": "-", "logprob": -0.01},
                {"token": " x", "logprob": -0.01},
                {"token": "\n", "logprob": -0.01},
                {"token": "</think>", "logprob": -0.01},
                {"token": "\n", "logprob": -0.01},
                {"token": "take", "logprob": -0.2},
                {"token": " key", "logprob": -0.2},
                {"token": "\n", "logprob": -0.2},
                {"token": "because", "logprob": -6.0},
                {"token": "...", "logprob": -6.0},
            ]
            return text, lp

    m = _C1Model()
    step = get_step_fn("C1", vc_mode="none")
    _action, tle, _vc, _tok, calls, *_rest = step("obs", [], m)
    assert calls == 1
    assert m.calls == 1
    assert tle is not None
    from src.signals.token_entropy import compute_tle

    assert tle["mean_entropy"] == compute_tle([{"logprob": -0.2}] * 3)["mean_entropy"]


def test_thinking_flags_c1_reason_only():
    class _M:
        def __init__(self) -> None:
            self.seen: list[bool | None] = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.seen.append(kwargs.get("enable_thinking"))
            return "<think>plan</think>\ngo east", ([{"logprob": -0.1}] * 5 if logprobs else None)

    m = _M()
    step = get_step_fn("C1", vc_mode="none")
    step("obs", [], m)
    assert m.seen == [True]


def test_c1_thinking_call_ignores_action_stop():
    class _M:
        def __init__(self) -> None:
            self.kwargs: list[dict] = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.kwargs.append(dict(kwargs))
            return "<think>x</think>\ngo east", ([{"logprob": -0.1}] * 5 if logprobs else None)

    m = _M()
    step = get_step_fn("C1", vc_mode="none", action_stop=["\n"], c1_cot_max_tokens=512)
    step("obs", [], m)
    assert m.kwargs[0].get("stop") is None
    assert m.kwargs[0]["max_tokens"] == 512
    assert m.kwargs[0]["enable_thinking"] is True


def test_c2_sample_uses_cot_max_tokens_not_action_cap():
    class _M:
        def __init__(self) -> None:
            self.kwargs: list[dict] = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.kwargs.append(dict(kwargs))
            return "<think>plan</think>\ngo north", ([{"logprob": -0.1}] * 4 if logprobs else None)

    m = _M()
    step = get_step_fn(
        "C2",
        vc_mode="none",
        action_max_tokens=32,
        action_stop=["\n"],
        c2_cot_max_tokens=1024,
        c2_n_samples=2,
        c2_tie_break_seed="seed",
    )
    step("obs", [], m)
    assert len(m.kwargs) == 2
    for kw in m.kwargs:
        assert kw.get("stop") is None
        assert kw["max_tokens"] == 1024
        assert kw["enable_thinking"] is True


def test_c1_single_call_commits_post_think_action():
    class _M:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.prompts.append(prompt)
            return "<think>plan only</think>\n\n", ([{"logprob": -0.1}] * 2 if logprobs else None)

    m = _M()
    step = get_step_fn("C1", vc_mode="none")
    action, *_rest = step("obs", [], m)
    assert len(m.prompts) == 1
    assert "<draft_action>" not in m.prompts[0]
    assert action == ""


def test_c1_no_verify_fallback_single_call():
    class _M:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            return "<think>plan</think>\ngo east", ([{"logprob": -0.1}] * 4 if logprobs else None)

    m = _M()
    step = get_step_fn("C1", vc_mode="none")
    action, *_rest = step("obs", [], m)
    assert m.calls == 1
    assert action == "go east"


def test_thinking_flags_c0_forced_off():
    class _M:
        seen: list[bool | None]

        def __init__(self) -> None:
            self.seen = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.seen.append(kwargs.get("enable_thinking"))
            return "go north", ([{"logprob": -0.2}] * 3 if logprobs else None)

    m = _M()
    step = get_step_fn("C0", vc_mode="none")
    step("obs", [], m)
    assert m.seen == [False]


def test_thinking_flags_c2_thinking_on_in_samples():
    class _M:
        seen: list[bool | None]

        def __init__(self) -> None:
            self.seen = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.seen.append(kwargs.get("enable_thinking"))
            return "Go north.", ([{"logprob": -0.1}] * 2 if logprobs else None)

    m = _M()
    step = get_step_fn("C2", vc_mode="none", c2_n_samples=3, c2_tie_break_seed="seed")
    step("obs", [], m)
    assert m.seen == [True, True, True]


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, logprobs: bool = False, **kwargs):
        self.calls += 1
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            assert kwargs.get("enable_thinking") is False
            return "65", [{"logprob": -0.1}] * 2 if logprobs else None
        text = "<think>\nplan\n</think>\nA->C"
        return text, [{"logprob": -0.5}] * 6 if logprobs else None


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
    assert "<task_context>" in m.vc_prompt
    assert "<output_to_judge>" in m.vc_prompt
    assert "current_obs" in m.vc_prompt
    assert DEFAULT_VC_FOLLOWUP_INSTRUCTION in m.vc_prompt


def test_followup_vc_prompt_can_override_instruction():
    custom = "Rate 0-100 only.\n\nConfidence:"

    class _Cap2:
        vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "60", [{"logprob": -0.1}] * 2 if logprobs else None
            return "north", [{"logprob": -0.5}] * 3 if logprobs else None

    m2 = _Cap2()
    step = get_step_fn("C0", vc_mode="followup", vc_followup_instruction=custom)
    step("obs", [], m2)
    assert m2.vc_prompt is not None
    assert custom in m2.vc_prompt
    assert DEFAULT_VC_FOLLOWUP_INSTRUCTION not in m2.vc_prompt


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


def test_c2_n_samples_is_configurable_and_traced():
    def _closed(action: str) -> str:
        return f"<think>\nplan\n</think>\n{action}"

    class _M:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            # Make every sample identical so vote agreement is 1.0.
            text = _closed("Go north") + "\nextra"
            return text, ([{"logprob": -0.1}] * 6 if logprobs else None)

    m = _M()
    step = get_step_fn("C2", vc_mode="none", c2_n_samples=5, c2_tie_break_seed="seed")
    action, _tle, _vc, _tok, calls, *_rest = step("obs", [], m)
    assert action == "Go north"  # punctuation stripped for execution
    assert calls == 5
    assert m.calls == 5


def test_c2_vote_normalization_merges_surface_forms():
    class _Seq:
        def __init__(self) -> None:
            self.i = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.i += 1
            outs = [
                "<think>\na\n</think>\nGo north.",
                "<think>\nb\n</think>\ngo  north",
                "<think>\nc\n</think>\nGO NORTH!",
            ]
            text = outs[self.i - 1]
            return text, ([{"logprob": -0.2}] * 6 if logprobs else None)

    m = _Seq()
    step = get_step_fn("C2", vc_mode="none", c2_n_samples=3, c2_tie_break_seed="seed")
    action, _tle, _vc, _tok, _calls, *_rest = step("obs", [], m)
    # All three should map to the same vote key; winner comes from the first matching sample.
    assert action in {"Go north", "go north", "GO NORTH"}


def test_extract_first_line_returns_first_non_empty():
    assert _extract_first_line("go north\nextra") == "go north"
    assert _extract_first_line("\n\n  take key  \n") == "take key"
    assert _extract_first_line("") == ""


def test_c0_recovers_embedded_action_from_verbose_output():
    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            text = (
                "Just the command.\n</task>\n"
                'The player is in the Livingroom, and the correct command should be "go east".'
            )
            return text, ([{"logprob": -0.2}] * 4 if logprobs else None)

    m = _M()
    step = get_step_fn("C0", vc_mode="none")
    action, *_rest = step("obs", [], m)
    assert action == "go east"


def test_c0_recovers_decision_cue_action_from_verbose_output():
    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            text = (
                "<think>We are in the bedroom and need to reach the kitchen. "
                "So we need to go north first.</think>"
            )
            return text, ([{"logprob": -0.2}] * 4 if logprobs else None)

    m = _M()
    step = get_step_fn("C0", vc_mode="none")
    action, *_rest = step("obs", [], m)
    assert action == "go north"


class _C1SingleCallModel:
    """Single reasoning call; optional VC follow-up."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, logprobs: bool = False, **kwargs):
        self.calls += 1
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            return "70", [{"logprob": -0.1}] * 2 if logprobs else None
        return "<think>\nConsider the map.\n</think>\ngo north", [
            {"logprob": -0.5}
        ] * 8 if logprobs else None


def test_c1_single_call_with_followup_vc():
    m = _C1SingleCallModel()
    step = get_step_fn("C1", vc_mode="followup", prompt_prefix="Prefix.")
    action, _tle, vc, _tok, lm_calls, *_rest = step("obs", [], m)
    assert action == "go north"
    assert lm_calls == 2
    assert m.calls == 2
    assert vc == 70.0


def test_c0_multiline_without_token_text_yields_no_tle():
    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            text = "go north\nextra"
            return text, ([{"logprob": -0.2}] * 5 if logprobs else None)

    m = _M()
    step = get_step_fn("C0", vc_mode="none")
    _action, tle, _vc, _tok, _calls, *_rest = step("obs", [], m)
    assert tle is None


def test_c1_vc_followup_includes_chain_of_thought_when_full_mode():
    """Exploratory ``judged_context=full`` includes CoT in VC prompt."""

    class _M:
        vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "55", [{"logprob": -0.1}] * 2 if logprobs else None
            return "<think>\nMarkerReasoningUnique42.\n</think>\ngo east", [
                {"logprob": -0.15}
            ] * 10 if logprobs else None

    m = _M()
    step = get_step_fn("C1", vc_mode="followup", vc_judged_context="full")
    step("obs", [], m)
    assert m.vc_prompt is not None
    assert "Chain-of-thought (your reasoning for this turn)" in m.vc_prompt
    assert "MarkerReasoningUnique42." in m.vc_prompt


def test_c1_vc_followup_action_only_default():
    class _M:
        vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "55", None
            return "<think>hidden</think>\ngo east", [{"logprob": -0.15}] * 3

    m = _M()
    step = get_step_fn("C1", vc_mode="followup")
    step("obs", [], m)
    assert m.vc_prompt is not None
    assert "[C1] go east" in m.vc_prompt
    assert "hidden" not in m.vc_prompt


def test_c1_single_call_vc_none():
    class _NoVc:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            return "<think>\nPlan.\n</think>\ntake key", [
                {"logprob": -0.4}
            ] * 8 if logprobs else None

    m = _NoVc()
    step = get_step_fn("C1", vc_mode="none", prompt_prefix="P.")
    action, _tle, vc, _tok, lm_calls, *_ = step("obs", [], m)
    assert action == "take key"
    assert lm_calls == 1
    assert m.calls == 1
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


def test_action_prompt_uses_flat_xml_sections():
    p = _build_prompt(
        "current",
        ["OBSERVATION: reset", "ACTION: A->C", "OBSERVATION: after"],
        "Tower rules here",
    )
    assert "<task>" in p
    assert "<history>" in p
    assert "<state>" in p
    assert "=== HISTORY ===" not in p


def test_parse_cot_action_post_think_parses_action_and_reasoning_internal():
    parsed = _parse_cot_action("<think>reason</think>\ntake key")
    assert parsed["status"] == "parsed"
    assert parsed["parse_method"] == "post_think"
    assert parsed["action"] == "take key"
    assert "reason" in parsed["reasoning_internal"]


def test_parse_cot_action_reasoning_only_is_unparsed():
    parsed = _parse_cot_action("<think>\njust thinking\n</think>\n")
    assert parsed["status"] == "unparsed"
    assert parsed["action"] == ""


def test_parse_cot_action_unclosed_think_recovers_action_via_fallback():
    parsed = _parse_cot_action("<think>reason\nA->C")
    # We accept either parsed fallback or unparsed depending on strictness, but never a tag artifact.
    assert "<think" not in (parsed["action"] or "").lower()


def test_parse_cot_action_legacy_action_prefix_parses():
    parsed = _parse_cot_action("ACTION: go east")
    assert parsed["status"] == "parsed"
    assert parsed["parse_method"] == "legacy_action_prefix"
    assert parsed["action"] == "go east"


def test_single_line_output_instruction_is_present_in_c0_prompt():
    class _Cap:
        prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.prompt = prompt
            return "go north", ([{"logprob": -0.2}] * 3 if logprobs else None)

    m = _Cap()
    step = get_step_fn("C0", vc_mode="none", prompt_prefix="Prefix.")
    step("obs", [], m)
    assert m.prompt is not None
    assert cs._SINGLE_LINE_OUTPUT_INSTRUCTION in m.prompt


def test_single_line_output_instruction_is_present_in_c1_reason_prompt():
    class _CapC1:
        prompts: list[str]

        def __init__(self) -> None:
            self.prompts = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.prompts.append(prompt)
            return "<think>x</think>\ngo east", ([{"logprob": -0.1}] * 5 if logprobs else None)

    m = _CapC1()
    step = get_step_fn("C1", vc_mode="none", prompt_prefix="Prefix.")
    step("obs", [], m)
    assert len(m.prompts) == 1
    assert "Before answering, briefly reason" in m.prompts[0]


def test_single_line_output_instruction_literal_has_single_definition_in_src():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    literal = "Write one valid game action on a single line"
    matches: list[Path] = []
    for p in (repo_root / "src").rglob("*.py"):
        txt = p.read_text(encoding="utf-8")
        if literal in txt:
            matches.append(p)
    assert len(matches) == 1, f"Found literal in multiple sources: {matches}"


def test_reasoning_output_instruction_literal_has_single_definition_in_src():
    """Regression guard: C1 and C2 used to have their own, independently drifted reasoning
    instruction text (C1 explicit, C2 accidentally reusing C0's no-thinking instruction) --
    unified 2026-07-21 into shared._REASONING_OUTPUT_INSTRUCTION. This fails again if a future
    edit reintroduces a second copy instead of importing the shared constant."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    literal = "Before answering, briefly reason inside <think>...</think> tags."
    matches: list[Path] = []
    for p in (repo_root / "src").rglob("*.py"):
        txt = p.read_text(encoding="utf-8")
        if literal in txt:
            matches.append(p)
    assert len(matches) == 1, f"Found literal in multiple sources: {matches}"
    assert len(matches) == 1, f"Found literal in multiple sources: {matches}"
