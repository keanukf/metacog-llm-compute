"""Reasoning-candidate admissibility: closed thinking + post_think action required.

Shared by C1 (n_samples=1) and C2 (n_samples>1 + majority vote) since 2026-07-21 -- both stages
call the same src.agent.stages.shared.reasoning_step_core engine, so this check (and the bug it
guards against: an unclosed <think> block silently getting parsed as the action) now applies to
both, not just C2."""

from __future__ import annotations

from src.agent.compute_stages import get_step_fn
from src.agent.stages.shared import assess_candidate_admissibility
from tests.agent.test_c1_c2_umbau import _action_logprobs


def _closed(action: str, *, reasoning: str = "plan") -> str:
    return f"<think>\n{reasoning}\n</think>\n{action}"


def test_assess_rejects_unclosed_thinking():
    meta = assess_candidate_admissibility("<think>\nstill thinking")
    assert meta["admissible"] is False
    assert meta["reject_reason"] == "thinking_unclosed"


def test_assess_accepts_post_think_action():
    meta = assess_candidate_admissibility(_closed("go north"))
    assert meta["admissible"] is True
    assert meta["parse_method"] == "post_think"
    assert meta["action_exec"] == "go north"


def test_c1_rejects_unclosed_thinking_instead_of_parsing_the_tag_as_the_action():
    """Regression for the 2026-07-20 QC-probe finding: C1 used to fall back to a naive
    first-line extraction that returned the literal '<think>' string as the action when
    reasoning never closed. It must now reject the step (empty action, no TLE) like C2 already
    did for its own samples."""

    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            text = "<think>\nendless loop, never closes"
            lp = [{"logprob": -0.1, "token": t} for t in text.split()]
            return text, lp if logprobs else None

    step = get_step_fn("C1", vc_mode="none")
    action, tle, _vc, _tok, _calls, *_mid, call_detail, _ptok = step("obs", [], _M())
    assert action == ""
    assert action != "<think>"
    assert tle is None
    assert call_detail["truncation_reason"] == "no_admissible_samples"


def test_c2_rejects_unclosed_sample_from_majority_vote():
    class _M:
        i = 0

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.i += 1
            if self.i == 1:
                text = _closed("A->C")
            elif self.i == 2:
                text = "<think>\ntruncated without close"
            else:
                text = _closed("A->B")
            lp = _action_logprobs(text, action_lp=-0.1)
            return text, lp if logprobs else None

    step = get_step_fn("C2", vc_mode="none", c2_n_samples=3, c2_tie_break_seed="seed")
    action, tle, _vc, tok, calls, *_mid, call_detail, _ptok = step("obs", [], _M())
    assert action in {"A->C", "A->B"}
    assert tle is not None
    assert calls == 3
    assert tok > 0
    assert call_detail["n_samples_admissible"] == 2
    assert call_detail["n_samples_rejected"] == 1
    assert call_detail["step_outcome"] == "vote"
    rejected = [sc for sc in call_detail["subcalls"] if not sc["admissible"]]
    assert len(rejected) == 1
    assert rejected[0]["reject_reason"] == "thinking_unclosed"


def test_c2_truncation_no_action_when_all_samples_inadmissible():
    class _M:
        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            text = "<think>\nno close tag"
            lp = [{"logprob": -0.1}] * 8
            return text, lp if logprobs else None

    step = get_step_fn("C2", vc_mode="followup", c2_n_samples=2, c2_tie_break_seed="seed")
    action, tle, vc, tok, calls, *_mid, call_detail, _ptok = step("obs", [], _M())
    assert action == ""
    assert tle is None
    assert vc is None
    assert calls == 2
    assert tok > 0
    assert call_detail["step_outcome"] == "truncation_no_action"
    assert call_detail["truncation_reason"] == "no_admissible_samples"
    assert call_detail["n_samples_admissible"] == 0
