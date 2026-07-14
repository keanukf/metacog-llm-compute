"""C2: self-consistency sampling and majority vote."""

from __future__ import annotations

import random
from typing import Any

from src.agent.cot_parser import parse_cot_action
from src.agent.stages.shared import (
    _SINGLE_LINE_OUTPUT_INSTRUCTION,
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    _action_generate_kwargs,
    _build_prompt,
    _normalize_action_for_execution,
    _normalize_vote_key,
    _run_vc_followup,
    _seeded_rng,
)
from src.signals import token_entropy, verbalized_confidence
from src.utils.inference.lmstudio.wrapper import attach_lmstudio_diagnostics_to_subcalls

_THINK_CLOSE_TAG = "</think>"


def _thinking_block_closed(text: str) -> bool:
    return _THINK_CLOSE_TAG.casefold() in (text or "").casefold()


def assess_c2_sample_admissibility(response: str) -> dict[str, Any]:
    """
    A C2 sample is vote-eligible only with a closed thinking block and a post-think action.

    Returns admissible flag, reject_reason, vote fields, and parse_method for tracing.
    """
    text = response or ""
    if not _thinking_block_closed(text):
        return {
            "admissible": False,
            "reject_reason": "thinking_unclosed",
            "action_exec": "",
            "vote_key": "",
            "raw_first_line": "",
            "parse_method": None,
        }
    parsed = parse_cot_action(text)
    parse_method = parsed.get("parse_method")
    action = str(parsed.get("action") or "").strip()
    if parsed.get("status") != "parsed" or parse_method != "post_think" or not action:
        reject_reason = "no_parseable_action"
        if parse_method:
            reject_reason = f"parse_method_{parse_method}"
        return {
            "admissible": False,
            "reject_reason": reject_reason,
            "action_exec": "",
            "vote_key": "",
            "raw_first_line": "",
            "parse_method": parse_method,
        }
    return {
        "admissible": True,
        "reject_reason": None,
        "action_exec": _normalize_action_for_execution(action),
        "vote_key": _normalize_vote_key(action),
        "raw_first_line": action,
        "parse_method": "post_think",
    }


def _annotate_c2_sample(sample: dict[str, Any]) -> None:
    meta = assess_c2_sample_admissibility(str(sample.get("response") or ""))
    sample.update(meta)


def _build_c2_sample_record(
    *,
    sample_index: int,
    prompt: str,
    text: str,
    logprobs: Any,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "kind": "sample",
        "sample_index": int(sample_index),
        "prompt": prompt,
        "response": text,
        "logprobs": logprobs,
        "tokens_generated": int(len(logprobs) if logprobs else 0),
    }
    _annotate_c2_sample(sample)
    return sample


def majority_vote(
    vote_keys_in_order: list[str],
    *,
    rng: random.Random | None,
) -> tuple[str, bool, dict[str, int]]:
    """
    Majority vote over pre-normalized vote keys.

    Returns (winning_key, tie_broken, counts).
    """
    if not vote_keys_in_order:
        return "", False, {}
    from collections import Counter

    counts = Counter(vote_keys_in_order)
    max_count = max(counts.values())
    tied = [k for k, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0], False, dict(counts)
    r = rng or random.Random(0)
    return str(r.choice(tied)), True, dict(counts)


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
    prompt = (
        f"{_build_prompt(observation, history, prompt_prefix)}\n\n{_SINGLE_LINE_OUTPUT_INSTRUCTION}"
    )
    act_tok = int(action_max_tokens) if action_max_tokens is not None else 32
    sample_max_tokens = (
        int(c2_cot_max_tokens) if c2_cot_max_tokens is not None else max(128, act_tok * 2)
    )
    if sample_max_tokens <= 0:
        sample_max_tokens = max(128, act_tok * 2)
    # Stop sequences truncate native thinking; budget must cover think + action line.
    gen_kw = _action_generate_kwargs(action_max_tokens, float(sample_temperature), None)
    gen_kw["max_tokens"] = sample_max_tokens
    # Self-consistency requires thinking + diversity temperature; TLE uses raw_logprobs (T=1.0 scale).
    gen_kw["enable_thinking"] = True
    samples: list[dict[str, Any]] = []
    total_tokens = 0
    n = max(1, int(n_samples))
    if hasattr(model, "generate_many") and callable(getattr(model, "generate_many")):
        outs = model.generate_many(prompt, n=n, logprobs=True, **gen_kw)
        for i, (text, logprobs) in enumerate(outs):
            samples.append(
                _build_c2_sample_record(
                    sample_index=int(i),
                    prompt=prompt,
                    text=text,
                    logprobs=logprobs,
                )
            )
            total_tokens += len(logprobs) if logprobs else 0
    else:
        for i in range(n):
            text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
            samples.append(
                _build_c2_sample_record(
                    sample_index=int(i),
                    prompt=prompt,
                    text=text,
                    logprobs=logprobs,
                )
            )
            total_tokens += len(logprobs) if logprobs else 0

    admissible_samples = [s for s in samples if s.get("admissible")]
    n_admissible = len(admissible_samples)
    n_rejected = int(n) - n_admissible
    vote_keys = [str(s.get("vote_key") or "") for s in admissible_samples if s.get("vote_key")]
    step_outcome = "vote"
    truncation_reason: str | None = None

    if not vote_keys:
        step_outcome = "truncation_no_action"
        truncation_reason = "no_admissible_samples"
        winning_key = ""
        tie_broken = False
        vote_counts: dict[str, int] = {}
        vote_agreement = 0.0
        unique_actions = 0
        winner_index = None
    else:
        rng = _seeded_rng(tie_break_seed, call_index=int(call_index))
        winning_key, tie_broken, vote_counts = majority_vote(vote_keys, rng=rng)
        max_count = max(vote_counts.values()) if vote_counts else 0
        vote_agreement = (float(max_count) / float(n_admissible)) if n_admissible > 0 else 0.0
        unique_actions = len({k for k in vote_keys if k})

        winner_index = None
        for s in admissible_samples:
            if str(s.get("vote_key") or "") == winning_key:
                winner_index = int(s.get("sample_index") or 0)
                break

    # Per-sample TLE (trace only; allocator signal uses winner sample only).
    for s in samples:
        lp = s.get("logprobs")
        raw_text = str(s.get("response") or "")
        s["tle"] = token_entropy.extract_action_tle_from_response(raw_text, lp) if lp else None
        if lp and isinstance(lp, list):
            vals: list[float] = []
            for x in lp:
                if not isinstance(x, dict):
                    continue
                lp_val = x.get("logprob")
                if isinstance(lp_val, (int, float)):
                    vals.append(float(lp_val))
            s["mean_logprob"] = (sum(vals) / len(vals)) if vals else None
        else:
            s["mean_logprob"] = None

    winner_sample = samples[winner_index] if winner_index is not None else {}
    winner_action_exec = str(winner_sample.get("action_exec") or "")
    winner_raw_first = str(winner_sample.get("raw_first_line") or "")
    winner_tle: dict[str, float] | None = None
    winner_mean_logprob: float | None = None
    winner_completion = str(winner_sample.get("response") or "")

    if winner_index is not None:
        winner_tle = winner_sample.get("tle")
        mlp = winner_sample.get("mean_logprob")
        winner_mean_logprob = float(mlp) if isinstance(mlp, (int, float)) else None

    vc: float | None = None
    vc_detail: dict[str, Any] | None = None
    extra_tok = 0
    extra_calls = 0
    mode = (vc_mode or "inline").strip().lower()
    if step_outcome == "vote" and winner_action_exec:
        if mode == "inline":
            vc = verbalized_confidence.parse_confidence(winner_completion)
        elif mode == "none":
            vc = None
        else:
            admissible_first_lines = [
                str(s.get("raw_first_line") or "") for s in admissible_samples
            ]
            vc, vc_detail, extra_tok, extra_calls = _run_vc_followup(
                model,
                observation=observation,
                history=history,
                prompt_prefix=prompt_prefix,
                stage_tag="C2",
                action_line=winner_action_exec,
                vc_followup_instruction=vc_followup_instruction,
                judged_context=vc_judged_context,
                retry_on_parse_failure=vc_retry_on_parse_failure,
                c2_n_samples=n_admissible,
                c2_sample_first_lines=admissible_first_lines,
                c2_winner_completion=winner_completion,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
                request_logprobs=vc_followup_logprobs,
                followup_max_context_chars=followup_max_context_chars,
                followup_cot_max_chars=followup_cot_max_chars,
                raw_completion_max_chars=vc_raw_completion_max_chars,
            )

    total_tokens += extra_tok
    lm_calls = int(n) + extra_calls
    if save_action_logprobs:
        lp_saved = [
            s.get("logprobs") if isinstance(s.get("logprobs"), list) else None for s in samples
        ]
    else:
        lp_saved = None
    sample_blocks = [
        (
            f"=== sample {int(s.get('sample_index', 0)) + 1}/{n} "
            f"(admissible={bool(s.get('admissible'))}, "
            f"first_line={str(s.get('raw_first_line') or '')!r}) ===\n"
            f"{str(s.get('response') or '')}"
        )
        for s in samples
    ]
    response_full = "\n\n".join(sample_blocks)
    subcalls = [
        {
            "kind": "sample",
            "sample_index": int(s.get("sample_index", 0)),
            "prompt": s.get("prompt") or "",
            "response": s.get("response") or "",
            "raw_first_line": s.get("raw_first_line") or "",
            "action_exec": s.get("action_exec") or "",
            "action_normalized": s.get("vote_key") or "",
            "admissible": bool(s.get("admissible")),
            "reject_reason": s.get("reject_reason"),
            "parse_method": s.get("parse_method"),
            "tokens_generated": int(s.get("tokens_generated") or 0),
            "tle": s.get("tle"),
            "mean_logprob": s.get("mean_logprob"),
            "is_winner": bool(
                winner_index is not None and int(s.get("sample_index") or 0) == int(winner_index)
            ),
        }
        for s in samples
    ]
    attach_lmstudio_diagnostics_to_subcalls(model, subcalls)
    call_detail = {
        "stage": "C2",
        "method": "self_consistency_majority_vote",
        "n_samples": int(n),
        "n_samples_admissible": int(n_admissible),
        "n_samples_rejected": int(n_rejected),
        "step_outcome": step_outcome,
        "truncation_reason": truncation_reason,
        "enable_thinking": True,
        "sample_temperature": float(sample_temperature),
        "sample_max_tokens": int(sample_max_tokens),
        "winner_index": int(winner_index) if winner_index is not None else None,
        "winning_vote_key": winning_key,
        "tie_broken": bool(tie_broken),
        "vote_counts": vote_counts,
        "vote_agreement": float(vote_agreement),
        "unique_actions": int(unique_actions),
        "winner_raw_first_line": winner_raw_first,
        "winner_mean_logprob": winner_mean_logprob,
        "subcalls": subcalls,
    }
    return (
        winner_action_exec,
        winner_tle,
        vc,
        int(total_tokens),
        int(lm_calls),
        lp_saved,
        vc_detail,
        prompt,
        response_full,
        call_detail,
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
