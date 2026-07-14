"""C1: single reasoning call with native thinking (one LM call)."""

from __future__ import annotations

from typing import Any

from src.agent.stages.shared import (
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    _action_generate_kwargs,
    _build_prompt,
    _normalize_action_line,
    _resolve_vc,
)
from src.signals import token_entropy
from src.utils.inference.lmstudio.wrapper import attach_lmstudio_diagnostics_to_subcalls


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
    base_prompt = _build_prompt(observation, history, prompt_prefix)
    reason_instruction = (
        "\n\n"
        "Before answering, briefly reason inside <think>...</think> tags.\n"
        "After </think>, write one valid game action on its own line (e.g. go north)."
    )
    reason_prompt = f"{base_prompt}{reason_instruction}"
    act_tok = int(action_max_tokens) if action_max_tokens is not None else 32
    reason_max_tokens = (
        int(c1_cot_max_tokens) if c1_cot_max_tokens is not None else max(128, act_tok * 2)
    )
    if reason_max_tokens <= 0:
        reason_max_tokens = max(128, act_tok * 2)
    reason_temp = c1_cot_temperature
    if reason_temp is None:
        reason_temp = float(action_temperature) if action_temperature is not None else 0.5
    # Stop sequences apply to the full completion; they truncate native thinking blocks.
    gen_kw = _action_generate_kwargs(action_max_tokens, float(reason_temp), None)
    gen_kw["max_tokens"] = reason_max_tokens
    gen_kw["enable_thinking"] = True
    text, logprobs = model.generate(reason_prompt, logprobs=True, **gen_kw)
    action = _normalize_action_line(text or "")
    tokens_used = len(logprobs) if logprobs else 0
    lm_calls = 1
    tle = token_entropy.extract_action_tle_from_response(text, logprobs) if logprobs else None

    vc, vc_detail, extra_tok, extra_calls = _resolve_vc(
        model,
        vc_mode=vc_mode,
        inline_text=text or "",
        observation=observation,
        history=history,
        prompt_prefix=prompt_prefix,
        stage_tag="C1",
        action_line=action,
        vc_followup_instruction=vc_followup_instruction,
        judged_context=vc_judged_context,
        retry_on_parse_failure=vc_retry_on_parse_failure,
        raw_action_completion=None,
        cot_text=text or "",
        verify_completion=None,
        c2_n_samples=None,
        c2_sample_first_lines=None,
        c2_winner_completion=None,
        vc_followup_logprobs=vc_followup_logprobs,
        followup_max_tokens=followup_max_tokens,
        followup_temperature=followup_temperature,
        followup_max_context_chars=followup_max_context_chars,
        followup_cot_max_chars=followup_cot_max_chars,
        raw_completion_max_chars=vc_raw_completion_max_chars,
    )
    tokens_used += extra_tok
    lm_calls += extra_calls

    lp_out: list[dict[str, Any]] | None = logprobs if save_action_logprobs else None
    response_full = text or ""
    subcalls: list[dict[str, Any]] = [
        {
            "kind": "reason",
            "prompt": reason_prompt,
            "response": text,
            "tokens_generated": int(len(logprobs) if logprobs else 0),
            "temperature": float(reason_temp),
            "max_tokens": int(reason_max_tokens),
            "enable_thinking": True,
        },
    ]
    attach_lmstudio_diagnostics_to_subcalls(model, subcalls)
    call_detail = {
        "stage": "C1",
        "subcalls": subcalls,
    }
    return (
        action,
        tle,
        vc,
        tokens_used,
        lm_calls,
        lp_out,
        vc_detail,
        reason_prompt,
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
