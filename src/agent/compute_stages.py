"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (best-of-N).
"""
from __future__ import annotations

from typing import Any

from src.signals import token_entropy, verbalized_confidence

# (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None, prompt_full, response_full)
StepReturn = tuple[str, dict[str, float] | None, float | None, int, int, Any, Any, str, str]


def _build_prompt(observation: str, history: list[str], prompt_prefix: str) -> str:
    obs = (observation or "").strip()
    if history:
        last = (history[-1] or "").strip()
        # If the caller already stored the current observation in history (common when history
        # stores ACTION/OBSERVATION pairs), avoid duplicating it.
        if last == obs or last == f"OBSERVATION: {obs}":
            body = "\n".join(history)
        else:
            body = "\n".join(history + [observation])
    else:
        body = observation
    pfx = (prompt_prefix or "").strip()
    if pfx:
        return f"{pfx}\n\n{body}"
    return body


def _extract_first_line(text: str) -> str:
    """First non-empty line of model output; used as the env action (defense in depth)."""
    for line in (text or "").strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return (text or "").strip()


def _action_generate_kwargs(
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None = None,
) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if action_max_tokens is not None:
        kw["max_tokens"] = int(action_max_tokens)
    if action_temperature is not None:
        kw["temperature"] = float(action_temperature)
    if action_stop:
        kw["stop"] = action_stop
    return kw


def _vc_followup_prompt(action_text: str) -> str:
    truncated = (action_text or "").strip()
    if len(truncated) > 500:
        truncated = truncated[:500] + "…"
    return (
        f"You just chose the action:\n{truncated}\n\n"
        "Rate your confidence that this action is correct for the current "
        "task, from 0 (no confidence) to 100 (certain).\n"
        "Respond with ONLY a single integer. Example: 75"
    )


def _run_vc_followup(
    model: Any,
    action_text: str,
    *,
    followup_max_tokens: int,
    followup_temperature: float,
    request_logprobs: bool,
) -> tuple[float | None, dict[str, Any] | None, int, int]:
    """Second LM call for verbalized confidence. Returns (vc, detail, extra_tokens, extra_calls)."""
    prompt = _vc_followup_prompt(action_text)
    gen_kw = {"max_tokens": int(followup_max_tokens), "temperature": float(followup_temperature)}
    if request_logprobs:
        text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
    else:
        text, logprobs = model.generate(prompt, logprobs=False, **gen_kw)
    detail = verbalized_confidence.extract_vc_from_followup(prompt, text, logprobs)
    vc_val = detail.get("vc_value")
    vc_f: float | None
    if isinstance(vc_val, (int, float)):
        vc_f = float(vc_val)
    else:
        vc_f = None
    extra_tokens = int(detail.get("vc_tokens_used") or 0)
    return vc_f, detail, extra_tokens, 1


def _resolve_vc(
    model: Any,
    action_text: str,
    *,
    vc_mode: str,
    inline_text: str,
    vc_followup_logprobs: bool,
    followup_max_tokens: int,
    followup_temperature: float,
) -> tuple[float | None, dict[str, Any] | None, int, int]:
    """Returns (vc, vc_detail, extra_tokens, extra_lm_calls)."""
    mode = (vc_mode or "inline").strip().lower()
    if mode == "none":
        return None, None, 0, 0
    if mode == "followup":
        return _run_vc_followup(
            model,
            action_text,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            request_logprobs=vc_followup_logprobs,
        )
    vc = verbalized_confidence.parse_confidence(inline_text)
    return vc, None, 0, 0


def _c0_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    save_action_logprobs: bool,
    vc_mode: str,
    prompt_prefix: str,
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None,
    followup_max_tokens: int,
    followup_temperature: float,
    vc_followup_logprobs: bool,
) -> StepReturn:
    prompt = _build_prompt(observation, history, prompt_prefix)
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
    text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    tokens_used = len(logprobs) if logprobs else 0
    lm_calls = 1

    action = _extract_first_line(text)

    vc, vc_detail, extra_tok, extra_calls = _resolve_vc(
        model,
        action,
        vc_mode=vc_mode,
        inline_text=text,
        vc_followup_logprobs=vc_followup_logprobs,
        followup_max_tokens=followup_max_tokens,
        followup_temperature=followup_temperature,
    )
    tokens_used += extra_tok
    lm_calls += extra_calls

    lp_out: list[dict[str, Any]] | None = logprobs if save_action_logprobs else None
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, prompt, text)


def _c1_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    save_action_logprobs: bool,
    vc_mode: str,
    prompt_prefix: str,
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None,
    followup_max_tokens: int,
    followup_temperature: float,
    vc_followup_logprobs: bool,
) -> StepReturn:
    return _c0_step_core(
        observation,
        history,
        model,
        save_action_logprobs=save_action_logprobs,
        vc_mode=vc_mode,
        prompt_prefix=prompt_prefix,
        action_max_tokens=action_max_tokens,
        action_temperature=action_temperature,
        action_stop=action_stop,
        followup_max_tokens=followup_max_tokens,
        followup_temperature=followup_temperature,
        vc_followup_logprobs=vc_followup_logprobs,
    )


def _majority_vote(actions: list[str]) -> str:
    if not actions:
        return ""
    from collections import Counter

    counts = Counter(actions)
    max_count = max(counts.values())
    for a in actions:
        if counts[a] == max_count:
            return a
    return actions[0]


def _c2_step_core(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
    *,
    save_action_logprobs: bool,
    vc_mode: str,
    prompt_prefix: str,
    action_max_tokens: int | None,
    action_temperature: float | None,
    action_stop: list[str] | None,
    followup_max_tokens: int,
    followup_temperature: float,
    vc_followup_logprobs: bool,
) -> StepReturn:
    prompt = _build_prompt(observation, history, prompt_prefix)
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
    # (first_line_action, raw_text, logprobs) per sample — TLE uses full completion; vote uses first line.
    samples: list[tuple[str, str, Any]] = []
    total_tokens = 0
    for _ in range(n_samples):
        text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
        first = _extract_first_line(text)
        samples.append((first, text, logprobs))
        total_tokens += len(logprobs) if logprobs else 0
    actions = [s[0] for s in samples]
    winner = _majority_vote(actions)
    tle = None
    win_logprobs: list[dict[str, Any]] | None = None
    for first, raw_text, logprobs in samples:
        if first == winner:
            tle = token_entropy.extract_tle_from_response(raw_text, logprobs) if logprobs else None
            win_logprobs = logprobs
            break
    if tle is None and samples:
        _first, raw_text, logprobs = samples[0]
        tle = token_entropy.extract_tle_from_response(raw_text, logprobs) if logprobs else None
        win_logprobs = logprobs

    vc: float | None = None
    vc_detail: dict[str, Any] | None = None
    extra_tok = 0
    extra_calls = 0
    mode = (vc_mode or "inline").strip().lower()
    if mode == "inline":
        for first, raw_text, _lp in samples:
            if first == winner:
                vc = verbalized_confidence.parse_confidence(raw_text)
                break
        if vc is None and samples:
            vc = verbalized_confidence.parse_confidence(samples[0][1])
    elif mode == "none":
        vc = None
    else:
        vc, vc_detail, extra_tok, extra_calls = _run_vc_followup(
            model,
            winner,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            request_logprobs=vc_followup_logprobs,
        )

    total_tokens += extra_tok
    lm_calls = int(n_samples) + extra_calls
    lp_saved = win_logprobs if save_action_logprobs else None
    sample_blocks = [
        f"=== sample {i + 1}/{n_samples} (first_line={first!r}) ===\n{raw_text}"
        for i, (first, raw_text, _lp) in enumerate(samples)
    ]
    response_full = "\n\n".join(sample_blocks)
    return (winner, tle, vc, total_tokens, lm_calls, lp_saved, vc_detail, prompt, response_full)


def c0_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C0: One action call with logprobs (TLE); optional VC prompt."""
    r = _c0_step_core(
        observation,
        history,
        model,
        save_action_logprobs=False,
        vc_mode="inline",
        prompt_prefix="",
        action_max_tokens=None,
        action_temperature=None,
        action_stop=None,
        followup_max_tokens=4,
        followup_temperature=0.0,
        vc_followup_logprobs=False,
    )
    return r[0], r[1], r[2], r[3], r[4]


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C1: CoT + verify stub — single call."""
    return c0_step(observation, history, model)


def c2_step(
    observation: str,
    history: list[str],
    model: Any,
    n_samples: int = 3,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C2: Best-of-N samples + majority vote."""
    r = _c2_step_core(
        observation,
        history,
        model,
        n_samples,
        save_action_logprobs=False,
        vc_mode="inline",
        prompt_prefix="",
        action_max_tokens=None,
        action_temperature=None,
        action_stop=None,
        followup_max_tokens=4,
        followup_temperature=0.0,
        vc_followup_logprobs=False,
    )
    return r[0], r[1], r[2], r[3], r[4]


def get_step_fn(
    stage: str,
    *,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    vc_mode: str = "inline",
    prompt_prefix: str = "",
    action_max_tokens: int | None = None,
    action_temperature: float | None = None,
    action_stop: list[str] | None = None,
    followup_max_tokens: int = 4,
    followup_temperature: float = 0.0,
):
    """
    Return the step function for stage 'C0', 'C1', or 'C2'.

    Returns a 9-tuple:
    (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None,
     prompt_full, response_full).

    ``save_logprob_distributions``: persist raw per-token rows for the *action* completion.

    ``save_vc_distributions``: request logprobs on the VC follow-up call (when ``vc_mode`` is followup).

    ``vc_mode``: ``followup`` | ``inline`` | ``none``.
    """
    core_map = {
        "C0": _c0_step_core,
        "C1": _c1_step_core,
        "C2": _c2_step_core,
    }
    fn = core_map.get(stage, _c0_step_core)
    vc_followup_logprobs = bool(save_vc_distributions) and (vc_mode or "").strip().lower() == "followup"

    if stage == "C2":

        def _w2(obs: str, hist: list[str], m: Any):
            return fn(
                obs,
                hist,
                m,
                save_action_logprobs=save_logprob_distributions,
                vc_mode=vc_mode,
                prompt_prefix=prompt_prefix,
                action_max_tokens=action_max_tokens,
                action_temperature=action_temperature,
                action_stop=action_stop,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
                vc_followup_logprobs=vc_followup_logprobs,
            )

        return _w2

    def _w(obs: str, hist: list[str], m: Any):
        return fn(
            obs,
            hist,
            m,
            save_action_logprobs=save_logprob_distributions,
            vc_mode=vc_mode,
            prompt_prefix=prompt_prefix,
            action_max_tokens=action_max_tokens,
            action_temperature=action_temperature,
            action_stop=action_stop,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            vc_followup_logprobs=vc_followup_logprobs,
        )

    return _w

