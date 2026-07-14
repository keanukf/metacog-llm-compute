"""C2 vote admissibility: closed thinking + post_think action required."""

from __future__ import annotations

from src.agent.compute_stages import get_step_fn
from src.agent.stages.c2 import assess_c2_sample_admissibility
from tests.test_c1_c2_umbau import _action_logprobs


def _closed(action: str, *, reasoning: str = "plan") -> str:
    return f"<think>\n{reasoning}\n</think>\n{action}"


def test_assess_rejects_unclosed_thinking():
    meta = assess_c2_sample_admissibility("<think>\nstill thinking")
    assert meta["admissible"] is False
    assert meta["reject_reason"] == "thinking_unclosed"


def test_assess_accepts_post_think_action():
    meta = assess_c2_sample_admissibility(_closed("go north"))
    assert meta["admissible"] is True
    assert meta["parse_method"] == "post_think"
    assert meta["action_exec"] == "go north"


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
    action, tle, _vc, tok, calls, *_mid, call_detail = step("obs", [], _M())
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
    action, tle, vc, tok, calls, *_mid, call_detail = step("obs", [], _M())
    assert action == ""
    assert tle is None
    assert vc is None
    assert calls == 2
    assert tok > 0
    assert call_detail["step_outcome"] == "truncation_no_action"
    assert call_detail["truncation_reason"] == "no_admissible_samples"
    assert call_detail["n_samples_admissible"] == 0
