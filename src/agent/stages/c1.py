"""C1: chain-of-thought + verify (two LM calls)."""

from __future__ import annotations

from typing import Any

from src.agent.stages.shared import (
    _SINGLE_LINE_OUTPUT_INSTRUCTION,
    DEFAULT_C1_VERIFY_INSTRUCTION,
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    _action_generate_kwargs,
    _build_prompt,
    _normalize_action_line,
    _parse_cot_action,
    _resolve_vc,
)
from src.signals import token_entropy


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
    c1_cot_temperature: float | None,
    c1_cot_max_tokens: int | None,
    c1_verify_temperature: float,
    c1_verify_max_tokens: int | None,
    c1_verify_stop: list[str] | None,
    c1_verify_instruction: str | None,
) -> StepReturn:
    """
    C1: two LM calls — (1) chain-of-thought inside <think>...</think> followed by a draft command,
    (2) verify pass that outputs the final single-line command with logprobs (TLE).
    """
    base_prompt = _build_prompt(observation, history, prompt_prefix)
    cot_instruction = (
        "\n\n"
        "Before answering, briefly reason inside <think>...</think> tags.\n"
        "After </think>, output exactly one final command on a single line."
    )
    cot_prompt = f"{base_prompt}{cot_instruction}"
    act_tok = int(action_max_tokens) if action_max_tokens is not None else 32
    cot_max_tokens = (
        int(c1_cot_max_tokens) if c1_cot_max_tokens is not None else max(128, act_tok * 2)
    )
    if cot_max_tokens <= 0:
        cot_max_tokens = max(128, act_tok * 2)
    cot_temp = c1_cot_temperature
    if cot_temp is None:
        cot_temp = float(action_temperature) if action_temperature is not None else 0.5
    cot_kw: dict[str, Any] = {
        "max_tokens": cot_max_tokens,
        "temperature": float(cot_temp),
        # C1-CoT is the only place where we enable model-native thinking.
        "enable_thinking": True,
    }
    cot_text, cot_lp = model.generate(cot_prompt, logprobs=True, **cot_kw)
    parsed = _parse_cot_action(cot_text or "")
    draft_action = str(parsed.get("action") or "")
    draft_status = str(parsed.get("status") or "unparsed")
    parse_method = str(parsed.get("parse_method") or "none")
    draft_reasoning_raw = str(parsed.get("reasoning_internal") or "")
    if draft_status == "parsed":
        verify_instr = (c1_verify_instruction or "").strip() or DEFAULT_C1_VERIFY_INSTRUCTION
        verify_instruction = (
            "\n\n"
            f"<draft_action>{draft_action}</draft_action>\n"
            f"<draft_status>{draft_status}</draft_status>\n\n"
            "Verify draft_action against the rules in <task> using <history> and <state> as the source of truth.\n\n"
            f"{verify_instr.strip()}"
        )
        verify_prompt = f"{base_prompt}{verify_instruction}"
    else:
        # Keep the unparsed branch minimal to reduce instruction-echo failures
        # like "Just the command." from long verifier prompts.
        verify_prompt = f"{base_prompt}\n\n{_SINGLE_LINE_OUTPUT_INSTRUCTION}"
    verify_max_tokens = (
        c1_verify_max_tokens if c1_verify_max_tokens is not None else action_max_tokens
    )
    verify_stop = c1_verify_stop if c1_verify_stop is not None else action_stop
    gen_kw = _action_generate_kwargs(verify_max_tokens, float(c1_verify_temperature), verify_stop)
    # Verify must be single-line action; force thinking OFF.
    gen_kw["enable_thinking"] = False
    final_text, logprobs = model.generate(verify_prompt, logprobs=True, **gen_kw)
    tle = token_entropy.extract_action_tle_from_response(final_text, logprobs) if logprobs else None
    tokens_used = len(logprobs) if logprobs else 0
    tokens_used += len(cot_lp) if cot_lp else 0
    lm_calls = 2

    action = _normalize_action_line(final_text or "")

    vc, vc_detail, extra_tok, extra_calls = _resolve_vc(
        model,
        vc_mode=vc_mode,
        inline_text=final_text or "",
        observation=observation,
        history=history,
        prompt_prefix=prompt_prefix,
        stage_tag="C1",
        action_line=action,
        vc_followup_instruction=vc_followup_instruction,
        raw_action_completion=None,
        cot_text=cot_text or "",
        verify_completion=final_text or "",
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
    response_full = f"=== C1 CoT ===\n{cot_text}\n\n=== C1 verify ===\n{final_text}"
    call_detail = {
        "stage": "C1",
        "draft_action": draft_action,
        "draft_status": draft_status,
        "parse_method": parse_method,
        "draft_reasoning_raw": draft_reasoning_raw,
        "subcalls": [
            {
                "kind": "cot",
                "prompt": cot_prompt,
                "response": cot_text,
                "tokens_generated": int(len(cot_lp) if cot_lp else 0),
                "temperature": float(cot_temp),
                "max_tokens": int(cot_max_tokens),
            },
            {
                "kind": "verify",
                "prompt": verify_prompt,
                "response": final_text,
                "tokens_generated": int(len(logprobs) if logprobs else 0),
                "temperature": float(c1_verify_temperature),
                "max_tokens": int(verify_max_tokens) if verify_max_tokens is not None else None,
                "stop": list(verify_stop) if isinstance(verify_stop, list) else None,
            },
        ],
    }
    return (
        action,
        tle,
        vc,
        tokens_used,
        lm_calls,
        lp_out,
        vc_detail,
        verify_prompt,
        response_full,
        call_detail,
    )


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C1: CoT + self-verify (two action LM calls); VC disabled (legacy helper)."""
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
        c1_verify_temperature=0.0,
        c1_verify_max_tokens=None,
        c1_verify_stop=None,
        c1_verify_instruction=None,
    )
    return r[0], r[1], r[2], r[3], r[4]
