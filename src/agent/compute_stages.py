"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (best-of-N).
"""
from __future__ import annotations

from typing import Any

from src.signals import token_entropy, verbalized_confidence

# (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None, prompt_full, response_full)
StepReturn = tuple[str, dict[str, float] | None, float | None, int, int, Any, Any, str, str]

# Present in every VC follow-up prompt — tests and mocks can detect the second call without coupling to wording details.
VC_FOLLOWUP_PROMPT_MARKER = "=== YOUR OUTPUT TO JUDGE ==="

# Default when YAML omits ``vc.followup_instruction`` (overridable per experiment).
DEFAULT_VC_FOLLOWUP_INSTRUCTION = (
    "How likely is the chosen action to be correct in this situation?\n"
    "Respond with only a single integer between 0 and 100,\n"
    "where 0 means certainly wrong and 100 means certainly correct.\n\n"
    "Confidence:"
)


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


def _normalize_action_line(text: str) -> str:
    line = _extract_first_line(text)
    if line.upper().startswith("ACTION:"):
        return line.split(":", 1)[1].strip()
    return line


def _extract_draft_action_from_cot(cot_text: str) -> str:
    """Prefer an explicit ``ACTION:`` line from the CoT output; else first non-empty line."""
    for line in (cot_text or "").splitlines():
        ls = line.strip()
        if ls.upper().startswith("ACTION:"):
            return ls.split(":", 1)[1].strip()
    return _normalize_action_line(cot_text)


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


def _truncate_text(text: str, *, max_chars: int, head_ratio: float = 0.5) -> str:
    """Same head/tail strategy as ``base_agent._truncate_for_history`` (duplicated to avoid import cycles)."""
    t = text or ""
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    r = float(head_ratio)
    if r <= 0.0:
        head = 0
    elif r >= 1.0:
        head = max_chars
    else:
        head = int(max_chars * r)
    tail = max_chars - head
    return f"{t[:head]}\n…[snip]…\n{t[-tail:]}"


def _build_model_output_to_judge_section(
    stage_tag: str,
    action_line: str,
    *,
    raw_action_completion: str | None,
    cot_text: str | None,
    verify_completion: str | None,
    c2_n_samples: int | None,
    c2_sample_first_lines: list[str] | None,
    followup_cot_max_chars: int,
    raw_completion_max_chars: int,
) -> str:
    """Text block for VC: must reflect what the model actually produced this turn (stage-dependent)."""
    al = (action_line or "").strip()
    tag = (stage_tag or "C0").strip().upper()
    if tag == "C0":
        raw = (raw_action_completion or "").strip()
        raw_t = _truncate_text(raw, max_chars=raw_completion_max_chars) if raw else ""
        parts = [f"--- Chosen command (first line, executed) ---\n{al}"]
        if raw_t:
            parts.append(f"--- Your full model completion (same turn) ---\n{raw_t}")
        return "\n\n".join(parts)
    if tag == "C1":
        cot = (cot_text or "").strip()
        cot_t = _truncate_text(cot, max_chars=followup_cot_max_chars) if cot else "(empty)"
        ver = (verify_completion or "").strip()
        ver_cap = max(4000, int(followup_cot_max_chars))
        ver_t = _truncate_text(ver, max_chars=ver_cap) if ver else ""
        parts = [
            "--- Chain-of-thought (your reasoning for this turn) ---",
            cot_t,
            f"--- Final command after self-check (executed) ---\n{al}",
        ]
        if ver_t:
            parts.extend(["--- Verify-phase raw completion ---", ver_t])
        return "\n\n".join(parts)
    if tag == "C2":
        n = int(c2_n_samples or 0)
        lines = c2_sample_first_lines or []
        summary = "\n".join(f"  sample {i + 1}: {fl!r}" for i, fl in enumerate(lines))
        return (
            f"--- Selected command (majority vote among {n} samples) ---\n{al}\n\n"
            f"--- Sample first-line commands ---\n{summary}"
        )
    return f"--- Chosen command ---\n{al}"


def _build_vc_followup_prompt(
    observation: str,
    history: list[str],
    prompt_prefix: str,
    *,
    stage_tag: str,
    action_line: str,
    instruction: str,
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    followup_cot_max_chars: int = 12000,
    followup_max_context_chars: int | None = None,
    raw_completion_max_chars: int = 8000,
) -> str:
    """
    VC follow-up uses the same task context as the action call (``observation`` / ``history`` / ``prompt_prefix``
    must match what was passed into the step for action generation).
    """
    task_context = _build_prompt(observation, history, prompt_prefix)
    judged = _build_model_output_to_judge_section(
        stage_tag,
        action_line,
        raw_action_completion=raw_action_completion,
        cot_text=cot_text,
        verify_completion=verify_completion,
        c2_n_samples=c2_n_samples,
        c2_sample_first_lines=c2_sample_first_lines,
        followup_cot_max_chars=followup_cot_max_chars,
        raw_completion_max_chars=raw_completion_max_chars,
    )
    instr = (instruction or "").strip() or DEFAULT_VC_FOLLOWUP_INSTRUCTION
    full = (
        "=== TASK CONTEXT ===\n"
        f"{task_context}\n\n"
        f"{VC_FOLLOWUP_PROMPT_MARKER}\n"
        f"{judged}\n\n"
        "=== INSTRUCTION ===\n"
        f"{instr}"
    )
    if followup_max_context_chars is not None and followup_max_context_chars > 0:
        full = _truncate_text(full, max_chars=followup_max_context_chars)
    return full


def _run_vc_followup(
    model: Any,
    *,
    observation: str,
    history: list[str],
    prompt_prefix: str,
    stage_tag: str,
    action_line: str,
    vc_followup_instruction: str,
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    followup_max_tokens: int,
    followup_temperature: float,
    request_logprobs: bool,
    followup_max_context_chars: int | None = None,
    followup_cot_max_chars: int = 12000,
    raw_completion_max_chars: int = 8000,
) -> tuple[float | None, dict[str, Any] | None, int, int]:
    """Second LM call for verbalized confidence. Returns (vc, detail, extra_tokens, extra_calls)."""
    prompt = _build_vc_followup_prompt(
        observation,
        history,
        prompt_prefix,
        stage_tag=stage_tag,
        action_line=action_line,
        instruction=vc_followup_instruction,
        raw_action_completion=raw_action_completion,
        cot_text=cot_text,
        verify_completion=verify_completion,
        c2_n_samples=c2_n_samples,
        c2_sample_first_lines=c2_sample_first_lines,
        followup_cot_max_chars=followup_cot_max_chars,
        followup_max_context_chars=followup_max_context_chars,
        raw_completion_max_chars=raw_completion_max_chars,
    )
    gen_kw = {
        "max_tokens": int(followup_max_tokens),
        "temperature": float(followup_temperature),
        "enable_thinking": False,
    }
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
    *,
    vc_mode: str,
    inline_text: str,
    observation: str,
    history: list[str],
    prompt_prefix: str,
    stage_tag: str,
    action_line: str,
    vc_followup_instruction: str,
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    vc_followup_logprobs: bool,
    followup_max_tokens: int,
    followup_temperature: float,
    followup_max_context_chars: int | None = None,
    followup_cot_max_chars: int = 12000,
    raw_completion_max_chars: int = 8000,
) -> tuple[float | None, dict[str, Any] | None, int, int]:
    """Returns (vc, vc_detail, extra_tokens, extra_lm_calls)."""
    mode = (vc_mode or "inline").strip().lower()
    if mode == "none":
        return None, None, 0, 0
    if mode == "followup":
        return _run_vc_followup(
            model,
            observation=observation,
            history=history,
            prompt_prefix=prompt_prefix,
            stage_tag=stage_tag,
            action_line=action_line,
            vc_followup_instruction=vc_followup_instruction,
            raw_action_completion=raw_action_completion,
            cot_text=cot_text,
            verify_completion=verify_completion,
            c2_n_samples=c2_n_samples,
            c2_sample_first_lines=c2_sample_first_lines,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            request_logprobs=vc_followup_logprobs,
            followup_max_context_chars=followup_max_context_chars,
            followup_cot_max_chars=followup_cot_max_chars,
            raw_completion_max_chars=raw_completion_max_chars,
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
    prompt = _build_prompt(observation, history, prompt_prefix)
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
    text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
    tle = token_entropy.extract_tle_from_response(text, logprobs) if logprobs else None
    tokens_used = len(logprobs) if logprobs else 0
    lm_calls = 1

    action = _extract_first_line(text)

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
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, prompt, text)


def _c1_step_core(
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
    """
    C1: two LM calls — (1) short chain-of-thought with a draft ``ACTION:`` line,
    (2) self-verify pass that outputs the final single-line imperative with logprobs (TLE).
    """
    base_prompt = _build_prompt(observation, history, prompt_prefix)
    cot_instruction = (
        "\n\nThink step by step (brief bullets or short sentences are fine). "
        "Then output exactly one line starting with ACTION: followed by your proposed "
        "single imperative command for this turn."
    )
    cot_prompt = f"{base_prompt}{cot_instruction}"
    act_tok = int(action_max_tokens) if action_max_tokens is not None else 32
    cot_max_tokens = max(128, act_tok * 2)
    cot_kw: dict[str, Any] = {
        "max_tokens": cot_max_tokens,
        "temperature": float(action_temperature) if action_temperature is not None else 0.5,
    }
    cot_text, cot_lp = model.generate(cot_prompt, logprobs=True, **cot_kw)
    draft = _extract_draft_action_from_cot(cot_text or "")
    if not draft:
        draft = "(no draft parsed)"

    verify_instruction = (
        "\n\n--- Self-check ---\n"
        f"Your draft command was: {draft}\n"
        "Re-read the game/task text above. Output exactly one imperative command on a single line "
        "(no ACTION: prefix, no quotes, no explanation). If the draft is still correct, repeat it verbatim; "
        "otherwise output your revised command only."
    )
    verify_prompt = f"{base_prompt}{verify_instruction}"
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
    final_text, logprobs = model.generate(verify_prompt, logprobs=True, **gen_kw)
    tle = token_entropy.extract_tle_from_response(final_text, logprobs) if logprobs else None
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
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, verify_prompt, response_full)


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
            observation=observation,
            history=history,
            prompt_prefix=prompt_prefix,
            stage_tag="C2",
            action_line=winner,
            vc_followup_instruction=vc_followup_instruction,
            c2_n_samples=n_samples,
            c2_sample_first_lines=list(actions),
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            request_logprobs=vc_followup_logprobs,
            followup_max_context_chars=followup_max_context_chars,
            followup_cot_max_chars=followup_cot_max_chars,
            raw_completion_max_chars=vc_raw_completion_max_chars,
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


def c1_step(
    observation: str,
    history: list[str],
    model: Any,
) -> tuple[str, dict[str, float] | None, float | None, int, int]:
    """C1: CoT + self-verify (two action LM calls); VC disabled (legacy helper)."""
    r = _c1_step_core(
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
    )
    return r[0], r[1], r[2], r[3], r[4]


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


def get_step_fn(
    stage: str,
    *,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
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

