"""
Compute stages facade: C0 (direct + logprobs), C1 (CoT + verify), C2 (self-consistency).

Implementation in ``src.agent.stages``; this module re-exports the public API for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Callable, cast

from src.agent.stages.c0 import c0_step, c0_step_core
from src.agent.stages.c1 import c1_step, c1_step_core
from src.agent.stages.c2 import c2_step, c2_step_core, majority_vote
from src.agent.stages.shared import (
    DEFAULT_C1_VERIFY_INSTRUCTION,
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    VC_FOLLOWUP_PROMPT_MARKER,
    StepReturn,
    _build_prompt,
    _extract_first_line,
    _parse_cot_action,
)

_c0_step_core = c0_step_core
_c1_step_core = c1_step_core
_c2_step_core = c2_step_core
_majority_vote = majority_vote

__all__ = [
    "DEFAULT_C1_VERIFY_INSTRUCTION",
    "DEFAULT_VC_FOLLOWUP_INSTRUCTION",
    "VC_FOLLOWUP_PROMPT_MARKER",
    "StepReturn",
    "c0_step",
    "c1_step",
    "c2_step",
    "get_step_fn",
    "_build_prompt",
    "_extract_first_line",
    "_parse_cot_action",
]


def get_step_fn(
    stage: str,
    *,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    c2_n_samples: int = 3,
    c2_tie_break_seed: str | int | None = None,
    vc_mode: str = "inline",
    prompt_prefix: str = "",
    vc_followup_instruction: str | None = None,
    action_max_tokens: int | None = None,
    action_temperature: float | None = None,
    action_stop: list[str] | None = None,
    followup_max_tokens: int = 4,
    followup_temperature: float = 0.0,
    followup_max_context_chars: int | None = None,
    followup_cot_max_chars: int = 12000,
    vc_raw_completion_max_chars: int = 8000,
    c1_cot_temperature: float | None = None,
    c1_cot_max_tokens: int | None = None,
    c1_verify_temperature: float = 0.0,
    c1_verify_max_tokens: int | None = None,
    c1_verify_stop: list[str] | None = None,
    c1_verify_instruction: str | None = None,
):
    """
    Return the step function for stage 'C0', 'C1', or 'C2'.

    Returns a 9-tuple:
    (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None,
     prompt_full, response_full).

    ``save_logprob_distributions``: persist raw per-token rows for the *action* completion.

    ``save_vc_distributions``: request logprobs on the VC follow-up call (when ``vc_mode`` is followup).

    ``vc_mode``: ``followup`` | ``inline`` | ``none``.

    ``followup_max_context_chars``: optional hard cap on the full VC follow-up prompt length (can make VC
    asymmetric vs the action prompt; prefer tightening ``history_*`` / observation limits instead).

    ``followup_cot_max_chars``: max characters for the C1 CoT block inside the VC follow-up (head/tail truncation).

    ``vc_raw_completion_max_chars``: max characters for the C0 full completion snippet in VC follow-up.

    ``vc_followup_instruction``: text under ``=== INSTRUCTION ===`` for VC follow-up (from YAML ``vc.followup_instruction``).
    """
    vc_instr = (vc_followup_instruction or "").strip() or DEFAULT_VC_FOLLOWUP_INSTRUCTION
    core_map = {
        "C0": c0_step_core,
        "C1": c1_step_core,
        "C2": c2_step_core,
    }
    fn = cast(Callable[..., StepReturn], core_map.get(stage, c0_step_core))
    vc_followup_logprobs = (
        bool(save_vc_distributions) and (vc_mode or "").strip().lower() == "followup"
    )

    if stage == "C2":
        c2_call_index = 0

        def _w2(obs: str, hist: list[str], m: Any):
            nonlocal c2_call_index
            idx = c2_call_index
            c2_call_index += 1
            return fn(
                obs,
                hist,
                m,
                n_samples=int(c2_n_samples),
                save_action_logprobs=save_logprob_distributions,
                tie_break_seed=c2_tie_break_seed,
                call_index=int(idx),
                vc_mode=vc_mode,
                prompt_prefix=prompt_prefix,
                vc_followup_instruction=vc_instr,
                action_max_tokens=action_max_tokens,
                action_temperature=action_temperature,
                action_stop=action_stop,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
                vc_followup_logprobs=vc_followup_logprobs,
                followup_max_context_chars=followup_max_context_chars,
                followup_cot_max_chars=followup_cot_max_chars,
                vc_raw_completion_max_chars=vc_raw_completion_max_chars,
            )

        return _w2

    def _w(obs: str, hist: list[str], m: Any):
        if stage == "C1":
            return fn(
                obs,
                hist,
                m,
                save_action_logprobs=save_logprob_distributions,
                vc_mode=vc_mode,
                prompt_prefix=prompt_prefix,
                vc_followup_instruction=vc_instr,
                action_max_tokens=action_max_tokens,
                action_temperature=action_temperature,
                action_stop=action_stop,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
                vc_followup_logprobs=vc_followup_logprobs,
                followup_max_context_chars=followup_max_context_chars,
                followup_cot_max_chars=followup_cot_max_chars,
                vc_raw_completion_max_chars=vc_raw_completion_max_chars,
                c1_cot_temperature=c1_cot_temperature,
                c1_cot_max_tokens=c1_cot_max_tokens,
                c1_verify_temperature=c1_verify_temperature,
                c1_verify_max_tokens=c1_verify_max_tokens,
                c1_verify_stop=c1_verify_stop,
                c1_verify_instruction=c1_verify_instruction,
            )
        return fn(
            obs,
            hist,
            m,
            save_action_logprobs=save_logprob_distributions,
            vc_mode=vc_mode,
            prompt_prefix=prompt_prefix,
            vc_followup_instruction=vc_instr,
            action_max_tokens=action_max_tokens,
            action_temperature=action_temperature,
            action_stop=action_stop,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            vc_followup_logprobs=vc_followup_logprobs,
            followup_max_context_chars=followup_max_context_chars,
            followup_cot_max_chars=followup_cot_max_chars,
            vc_raw_completion_max_chars=vc_raw_completion_max_chars,
        )

    return _w
