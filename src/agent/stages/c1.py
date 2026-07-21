"""C1: single reasoning call with native thinking (one LM call).

Thin wrapper over the shared reasoning engine (shared.reasoning_step_core) that C2 also uses --
C1 is that engine called with n_samples=1 (one candidate, no vote). See shared.py's module
docstring for the admissibility/parsing bug this unification fixed.
"""

from __future__ import annotations

from typing import Any

from src.agent.stages.shared import (
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    reasoning_step_core,
)


def c1_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    save_action_logprobs: bool,
    vc_mode: str,
    prompt_prefix: str,
    vc_followup_instruction: str,
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None,
    followup_max_tokens: int,
    followup_temperature: float,
    vc_followup_logprobs: bool,
    followup_max_context_chars: int | None,
    followup_cot_max_chars: int,
    vc_raw_completion_max_chars: int,
    vc_judged_context: str = "action_only",
    vc_retry_on_parse_failure: bool = True,
    c1_cot_temperature: float | None,
    c1_cot_max_tokens: int | None,
) -> StepReturn:
    """
    C1: one LM call — reason inside <think>...</think>, then commit
    a final single-line action. TLE is measured on the action tokens only (post-think).
    """
    reason_temp = c1_cot_temperature
    if reason_temp is None:
        reason_temp = float(action_temperature) if action_temperature is not None else 0.5

    (
        action,
        tle,
        vc,
        tokens_used,
        lm_calls,
        lp_saved,
        vc_detail,
        prompt,
        response_full,
        call_detail,
    ) = reasoning_step_core(
        observation,
        history,
        model,
        n_samples=1,
        sample_temperature=float(reason_temp),
        cot_max_tokens=c1_cot_max_tokens,
        stage_tag="C1",
        save_action_logprobs=save_action_logprobs,
        vc_mode=vc_mode,
        prompt_prefix=prompt_prefix,
        vc_followup_instruction=vc_followup_instruction,
        action_max_tokens=action_max_tokens,
        action_temperature=action_temperature,
        action_stop=action_stop,
        followup_max_tokens=followup_max_tokens,
        followup_temperature=followup_temperature,
        vc_followup_logprobs=vc_followup_logprobs,
        followup_max_context_chars=followup_max_context_chars,
        followup_cot_max_chars=followup_cot_max_chars,
        vc_raw_completion_max_chars=vc_raw_completion_max_chars,
        vc_judged_context=vc_judged_context,
        vc_retry_on_parse_failure=vc_retry_on_parse_failure,
    )
    # reasoning_step_core always returns action_logprobs_raw as one list per candidate
    # ([cand0_logprobs]); C1's external contract (sidecar writer, K-sensitivity sweep) expects a
    # single flat list, matching C0. Unwrap here rather than changing that contract.
    lp_out = lp_saved[0] if lp_saved else None
    return (
        action,
        tle,
        vc,
        tokens_used,
        lm_calls,
        lp_out,
        vc_detail,
        prompt,
        response_full,
        call_detail,
    )


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C1: single reasoning call; VC disabled (legacy helper)."""
    r = c1_step_core(
        observation,
        history,
        model,
        save_action_logprobs=False,
        vc_mode="none",
        prompt_prefix="",
        vc_followup_instruction=DEFAULT_VC_FOLLOWUP_INSTRUCTION,
        action_max_tokens=None,
        action_temperature=None,
        action_stop=None,
        followup_max_tokens=4,
        followup_temperature=0.0,
        vc_followup_logprobs=False,
        followup_max_context_chars=None,
        followup_cot_max_chars=12000,
        vc_raw_completion_max_chars=8000,
        c1_cot_temperature=None,
        c1_cot_max_tokens=None,
    )
    return r[0], r[1], r[2], r[3], r[4]
