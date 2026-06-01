"""C2: self-consistency sampling and majority vote."""

from __future__ import annotations

import random
from typing import Any

from src.agent.stages.shared import (
    _SINGLE_LINE_OUTPUT_INSTRUCTION,
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    StepReturn,
    _action_generate_kwargs,
    _build_prompt,
    _extract_first_line,
    _normalize_action_for_execution,
    _normalize_vote_key,
    _run_vc_followup,
    _seeded_rng,
)
from src.signals import token_entropy, verbalized_confidence
from src.utils.inference.lmstudio.wrapper import attach_lmstudio_diagnostics_to_subcalls


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
    # Force thinking OFF for self-consistency sampling; only C1-CoT uses thinking.
    gen_kw["enable_thinking"] = False
    # Per sample: raw_first_line, action_exec, vote_key, raw_text, logprobs
    samples: list[dict[str, Any]] = []
    total_tokens = 0
    n = max(1, int(n_samples))
    # Optional backend optimization: vLLM can generate N samples in one call.
    if hasattr(model, "generate_many") and callable(getattr(model, "generate_many")):
        outs = model.generate_many(prompt, n=n, logprobs=True, **gen_kw)
        for i, (text, logprobs) in enumerate(outs):
            first_raw = _extract_first_line(text)
            action_exec = _normalize_action_for_execution(first_raw)
            vote_key = _normalize_vote_key(first_raw)
            samples.append(
                {
                    "kind": "sample",
                    "sample_index": int(i),
                    "prompt": prompt,
                    "response": text,
                    "raw_first_line": first_raw,
                    "action_exec": action_exec,
                    "vote_key": vote_key,
                    "logprobs": logprobs,
                    "tokens_generated": int(len(logprobs) if logprobs else 0),
                }
            )
            total_tokens += len(logprobs) if logprobs else 0
    else:
        for i in range(n):
            text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
            first_raw = _extract_first_line(text)
            action_exec = _normalize_action_for_execution(first_raw)
            vote_key = _normalize_vote_key(first_raw)
            samples.append(
                {
                    "kind": "sample",
                    "sample_index": int(i),
                    "prompt": prompt,
                    "response": text,
                    "raw_first_line": first_raw,
                    "action_exec": action_exec,
                    "vote_key": vote_key,
                    "logprobs": logprobs,
                    "tokens_generated": int(len(logprobs) if logprobs else 0),
                }
            )
            total_tokens += len(logprobs) if logprobs else 0

    vote_keys = [str(s.get("vote_key") or "") for s in samples]
    rng = _seeded_rng(tie_break_seed, call_index=int(call_index))
    winning_key, tie_broken, vote_counts = majority_vote(vote_keys, rng=rng)
    max_count = max(vote_counts.values()) if vote_counts else 0
    vote_agreement = (float(max_count) / float(n)) if n > 0 else 0.0
    unique_actions = len({k for k in vote_keys if k})

    winner_index: int | None = None
    for s in samples:
        if str(s.get("vote_key") or "") == winning_key:
            winner_index = int(s.get("sample_index") or 0)
            break
    if winner_index is None and samples:
        winner_index = int(samples[0].get("sample_index") or 0)

    # Per-sample metrics (secondary; primary TLE is the winner's).
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

    winner_action_exec = ""
    winner_raw_first = ""
    winner_tle: dict[str, float] | None = None
    winner_mean_logprob: float | None = None
    for s in samples:
        if str(s.get("vote_key") or "") == winning_key:
            winner_action_exec = str(s.get("action_exec") or "")
            winner_raw_first = str(s.get("raw_first_line") or "")
            winner_tle = s.get("tle")
            mlp = s.get("mean_logprob")
            winner_mean_logprob = float(mlp) if isinstance(mlp, (int, float)) else None
            break
    if not winner_action_exec and samples:
        s0 = samples[0]
        winner_action_exec = str(s0.get("action_exec") or "")
        winner_raw_first = str(s0.get("raw_first_line") or "")
        winner_tle = s0.get("tle")
        mlp = s0.get("mean_logprob")
        winner_mean_logprob = float(mlp) if isinstance(mlp, (int, float)) else None

    vc: float | None = None
    vc_detail: dict[str, Any] | None = None
    extra_tok = 0
    extra_calls = 0
    mode = (vc_mode or "inline").strip().lower()
    if mode == "inline":
        for s in samples:
            if str(s.get("vote_key") or "") == winning_key:
                vc = verbalized_confidence.parse_confidence(str(s.get("response") or ""))
                break
        if vc is None and samples:
            vc = verbalized_confidence.parse_confidence(str(samples[0].get("response") or ""))
    elif mode == "none":
        vc = None
    else:
        vc, vc_detail, extra_tok, extra_calls = _run_vc_followup(
            model,
            observation=observation,
            history=history,
            prompt_prefix=prompt_prefix,
            stage_tag="C2",
            action_line=winner_action_exec,
            vc_followup_instruction=vc_followup_instruction,
            c2_n_samples=n,
            c2_sample_first_lines=[str(s.get("raw_first_line") or "") for s in samples],
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
        # For C2 we keep all samples' action logprobs so posthoc analysis can study agreement vs uncertainty.
        lp_saved = [
            s.get("logprobs") if isinstance(s.get("logprobs"), list) else None for s in samples
        ]
    else:
        lp_saved = None
    sample_blocks = [
        f"=== sample {int(s.get('sample_index', 0)) + 1}/{n} (first_line={str(s.get('raw_first_line') or '')!r}) ===\n{str(s.get('response') or '')}"
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
        followup_max_tokens=4,
        followup_temperature=0.0,
        vc_followup_logprobs=False,
        followup_max_context_chars=None,
        followup_cot_max_chars=12000,
        vc_raw_completion_max_chars=8000,
    )
    return r[0], r[1], r[2], r[3], r[4]
