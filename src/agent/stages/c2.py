"""C2: self-consistency sampling and majority vote.

Thin wrapper over the shared reasoning engine (shared.reasoning_step_core) that C1 also uses --
C2 is that engine called with n_samples>1 plus a real majority vote (C1's n_samples=1 vote is
trivial: the one admissible candidate, or none).
"""

from __future__ import annotations

from typing import Any

from src.agent.stages.shared import (
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    assess_candidate_admissibility,
    majority_vote,
    reasoning_step_core,
)

# Re-exported for backward compatibility (src.agent.compute_stages imports these from here).
__all__ = ["c2_step", "c2_step_core", "majority_vote", "assess_candidate_admissibility"]


def c2_step_core(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
    *,
    tie_break_seed: str | int | None = None,
    call_index: int = 0,
    sample_temperature: float = 0.7,
    save_action_logprobs: bool,
    vc_mode: str,
    prompt_prefix: str,
    vc_followup_instruction: str,
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None,
    c2_cot_max_tokens: int | None,
    followup_max_tokens: int,
    followup_temperature: float,
    vc_followup_logprobs: bool,
    followup_max_context_chars: int | None,
    followup_cot_max_chars: int,
    vc_raw_completion_max_chars: int,
    vc_judged_context: str = "action_only",
    vc_retry_on_parse_failure: bool = True,
) -> StepReturn:
    """C2: self-consistency sampling (N samples + majority vote)."""
    return reasoning_step_core(
        observation,
        history,
        model,
        n_samples=n_samples,
        sample_temperature=float(sample_temperature),
        cot_max_tokens=c2_cot_max_tokens,
        stage_tag="C2",
        tie_break_seed=tie_break_seed,
        call_index=call_index,
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


def c2_step(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C2: Self-consistency sampling (N samples + majority vote)."""
    r = c2_step_core(
        observation,
        history,
        model,
        n_samples,
        save_action_logprobs=False,
        vc_mode="inline",
        prompt_prefix="",
        vc_followup_instruction=DEFAULT_VC_FOLLOWUP_INSTRUCTION,
        action_max_tokens=None,
        action_temperature=None,
        action_stop=None,
        c2_cot_max_tokens=None,
        followup_max_tokens=4,
        followup_temperature=0.0,
        vc_followup_logprobs=False,
        followup_max_context_chars=None,
        followup_cot_max_chars=12000,
        vc_raw_completion_max_chars=8000,
    )
    return r[0], r[1], r[2], r[3], r[4]
