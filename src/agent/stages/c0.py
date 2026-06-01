"""C0: direct action + logprobs (TLE); optional VC."""

from __future__ import annotations

from typing import Any

from src.agent.stages.shared import (
    _SINGLE_LINE_OUTPUT_INSTRUCTION,
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    _action_generate_kwargs,
    _build_prompt,
    _normalize_action_line,
    _resolve_vc,
)
from src.signals import token_entropy
from src.utils.inference.lmstudio.wrapper import collect_step_inference_diagnostics


def c0_step_core(
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
) -> StepReturn:
    prompt = (
        f"{_build_prompt(observation, history, prompt_prefix)}\n\n{_SINGLE_LINE_OUTPUT_INSTRUCTION}"
    )
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
    # Force thinking OFF for baseline action calls (C0). Only the C1-CoT subcall uses thinking.
    gen_kw["enable_thinking"] = False
    text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
    tle = token_entropy.extract_action_tle_from_response(text, logprobs) if logprobs else None
    tokens_used = len(logprobs) if logprobs else 0
    lm_calls = 1

    action = _normalize_action_line(text)

    vc, vc_detail, extra_tok, extra_calls = _resolve_vc(
        model,
        vc_mode=vc_mode,
        inline_text=text,
        observation=observation,
        history=history,
        prompt_prefix=prompt_prefix,
        stage_tag="C0",
        action_line=action,
        vc_followup_instruction=vc_followup_instruction,
        raw_action_completion=text,
        cot_text=None,
        verify_completion=None,
        c2_n_samples=None,
        c2_sample_first_lines=None,
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
    diag = collect_step_inference_diagnostics(model)
    call_detail = {"stage": "C0", "inference_diagnostics": diag} if diag else None
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, prompt, text, call_detail)


def c0_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C0: One action call with logprobs (TLE); optional VC prompt."""
    r = c0_step_core(
        observation,
        history,
        model,
        save_action_logprobs=False,
        vc_mode="inline",
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
    )
    return r[0], r[1], r[2], r[3], r[4]
