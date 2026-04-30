"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (self-consistency / majority vote).
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from src.signals import token_entropy, verbalized_confidence

# (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None, prompt_full, response_full, call_detail|None)
StepReturn = tuple[str, dict[str, float] | None, float | None, int, int, Any, Any, str, str, Any]

# Present in every VC follow-up prompt — tests and mocks can detect the second call without coupling to wording details.
VC_FOLLOWUP_PROMPT_MARKER = "<output_to_judge>"

# Default when YAML omits ``vc.followup_instruction`` (overridable per experiment).
DEFAULT_VC_FOLLOWUP_INSTRUCTION = (
    "How likely is the chosen action to be correct in this situation?\n"
    "Respond with only a single integer between 0 and 100,\n"
    "where 0 means certainly wrong and 100 means certainly correct.\n\n"
    "Confidence:"
)

DEFAULT_C1_VERIFY_INSTRUCTION = (
    "You are doing a verification pass over a draft command.\n"
    "Check the following criteria against the task context above:\n"
    "1) Format: Output exactly ONE imperative command on a single line (no quotes, no explanation, no ACTION: prefix).\n"
    "2) Plausibility: The command must be feasible in the current state (e.g., referenced objects/pegs exist).\n"
    "3) Goal-advancement: Prefer commands that make progress toward the stated goal; avoid repeating no-op actions.\n"
    "4) Consistency: Avoid trivial oscillation unless the observation/state has changed in a way that warrants it.\n\n"
    "Output exactly ONE imperative command on a single line (no ACTION: prefix, no quotes, no explanation).\n"
)


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _xml_block(tag: str, content: str, *, attrs: str = "") -> str:
    attr_text = f" {attrs}" if attrs else ""
    return f"<{tag}{attr_text}>\n{content or ''}\n</{tag}>"


def _unwrap_history_entry(entry: str, prefix: str) -> str | None:
    line = entry or ""
    if line.startswith(prefix):
        rest = line.split(":", 1)[1]
        if rest.startswith(" "):
            rest = rest[1:]
        return rest
    return None


def _history_to_xml(history: list[str]) -> str:
    """
    Render internal history lines (ACTION:/OBSERVATION:) to a flat XML schema.

    Hierarchy: <history> -> <reset_observation>, <step>, and optional <pinned_recipe>.
    """
    if not history:
        return _xml_block("history", "")

    parts: list[str] = []
    first_obs = _unwrap_history_entry(history[0], "OBSERVATION:")
    if first_obs is not None:
        parts.append(_xml_block("reset_observation", (first_obs or "").rstrip()))

    idx = 1
    step_index = 1
    while idx < len(history):
        entry = history[idx] or ""

        if entry.startswith("PINNED RECIPE:"):
            recipe = _unwrap_history_entry(entry, "PINNED RECIPE:")
            if recipe is None:
                recipe = entry.split(":", 1)[1].lstrip() if ":" in entry else entry
            parts.append(_xml_block("pinned_recipe", (recipe or "").rstrip()))
            idx += 1
            continue

        act = _unwrap_history_entry(entry, "ACTION:")
        obs = _unwrap_history_entry(history[idx + 1], "OBSERVATION:") if (idx + 1) < len(history) else None
        if act is not None and obs is not None:
            step_body = "\n".join(
                [
                    f"<action>{act.rstrip()}</action>",
                    f"<observation>{obs.rstrip()}</observation>",
                ]
            )
            parts.append(_xml_block("step", step_body, attrs=f'index=\"{step_index}\"'))
            step_index += 1
            idx += 2
            continue

        parts.append(_xml_block("history_item", entry.rstrip()))
        idx += 1

    return _xml_block("history", "\n\n".join(parts))


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "")


def _build_prompt(observation: str, history: list[str], prompt_prefix: str) -> str:
    """
    Build the action-generation prompt from:
    - domain prefix (instructions)
    - compact history (reset + last N action/obs pairs)
    - current observation

    Important: do NOT strip leading whitespace from observations. TextWorld observations often
    contain ASCII art / fixed-width formatting where leading spaces carry structure. We only
    rstrip() to normalize trailing newlines for stable duplication checks.
    """

    def _rstrip(s: str) -> str:
        return (s or "").rstrip()

    obs_now = _rstrip(observation or "")
    include_current_obs = True
    if history:
        last_line = history[-1] or ""
        last_obs_raw = _unwrap_history_entry(last_line, "OBSERVATION:")
        last_obs = _rstrip(last_obs_raw) if last_obs_raw is not None else None
        # If the caller already stored the *current* observation in history (true for step 0 reset),
        # don't append it again (TextWorld reset text can be very large).
        if last_obs is not None and last_obs == obs_now:
            include_current_obs = False

    pfx = (prompt_prefix or "").strip()

    parts: list[str] = []
    if pfx:
        parts.append(_xml_block("task", pfx))
    if history:
        parts.append(_history_to_xml(history))
    if include_current_obs and (observation is not None):
        parts.append(_xml_block("state", observation or ""))
    return "\n\n".join(parts).rstrip()


def _extract_first_line(text: str) -> str:
    """First non-empty line of model output; used as the env action (defense in depth)."""
    for line in (text or "").strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return (text or "").strip()


def _normalize_action_line(text: str) -> str:
    line = _extract_first_line(_strip_think_blocks(text or ""))
    if line.upper().startswith("ACTION:"):
        return line.split(":", 1)[1].strip()
    return line


_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")


def _normalize_vote_key(action_line: str) -> str:
    """
    Normalization used ONLY for voting / agreement statistics.

    We intentionally do not over-normalize (e.g. we don't remove internal punctuation),
    but we do collapse common surface-form variation so self-consistency is not
    artificially deflated.
    """
    s = _strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _normalize_action_for_execution(action_line: str) -> str:
    """
    Canonical action string used for env.step(...).

    Must be consistent with vote keys while being conservative (do NOT casefold),
    so domains like Tower of Hanoi keep their expected surface form (e.g. "A->C").
    """
    s = _strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _seeded_rng(seed_base: str | int | None, *, call_index: int) -> random.Random:
    """
    Deterministic RNG for tie-breaking in C2.

    We use a stable hash to avoid Python's randomized hash() across processes.
    """
    base = "" if seed_base is None else str(seed_base)
    h = hashlib.md5(f"{base}::{int(call_index)}".encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    return random.Random(seed_int)


def _extract_draft_action_from_cot(cot_text: str) -> str:
    """
    Extract draft command from CoT output.

    Preferred format (Qwen3-native): <think>...</think> then a single-line command.
    Fallback: legacy ACTION: line or first non-empty line.
    """
    t = cot_text or ""
    close = t.lower().rfind("</think>")
    if close >= 0:
        after = t[close + len("</think>") :]
        draft = _extract_first_line(after)
        if draft:
            return draft
    for line in (t or "").splitlines():
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
        f"{_xml_block('task_context', task_context)}\n\n"
        f"{_xml_block('output_to_judge', judged)}\n\n"
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
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, prompt, text, None)


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
    cot_max_tokens = int(c1_cot_max_tokens) if c1_cot_max_tokens is not None else max(128, act_tok * 2)
    if cot_max_tokens <= 0:
        cot_max_tokens = max(128, act_tok * 2)
    cot_temp = c1_cot_temperature
    if cot_temp is None:
        cot_temp = float(action_temperature) if action_temperature is not None else 0.5
    cot_kw: dict[str, Any] = {
        "max_tokens": cot_max_tokens,
        "temperature": float(cot_temp),
    }
    cot_text, cot_lp = model.generate(cot_prompt, logprobs=True, **cot_kw)
    draft = _extract_draft_action_from_cot(cot_text or "")
    if not draft:
        draft = "(no draft parsed)"

    verify_instr = (c1_verify_instruction or "").strip() or DEFAULT_C1_VERIFY_INSTRUCTION
    verify_instruction = (
        "\n\n"
        f"<draft>{draft}</draft>\n\n"
        "Verify the draft above against the rules in <task>. If it is correct, output it again. "
        "Otherwise output a corrected single command. Output exactly one command on a single line.\n\n"
        f"{verify_instr.strip()}"
    )
    verify_prompt = f"{base_prompt}{verify_instruction}"
    verify_max_tokens = c1_verify_max_tokens if c1_verify_max_tokens is not None else action_max_tokens
    verify_stop = c1_verify_stop if c1_verify_stop is not None else action_stop
    gen_kw = _action_generate_kwargs(verify_max_tokens, float(c1_verify_temperature), verify_stop)
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
    return (action, tle, vc, tokens_used, lm_calls, lp_out, vc_detail, verify_prompt, response_full, call_detail)


def _majority_vote(
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


def _c2_step_core(
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
    prompt = _build_prompt(observation, history, prompt_prefix)
    gen_kw = _action_generate_kwargs(action_max_tokens, action_temperature, action_stop)
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
    winning_key, tie_broken, vote_counts = _majority_vote(vote_keys, rng=rng)
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
            vals = [float(x.get("logprob")) for x in lp if isinstance(x, dict) and x.get("logprob") is not None]
            s["mean_logprob"] = (sum(vals) / len(vals)) if vals else None
        else:
            s["mean_logprob"] = None

    winner_action_exec = ""
    winner_raw_first = ""
    win_logprobs: list[dict[str, Any]] | None = None
    winner_tle: dict[str, float] | None = None
    winner_mean_logprob: float | None = None
    for s in samples:
        if str(s.get("vote_key") or "") == winning_key:
            winner_action_exec = str(s.get("action_exec") or "")
            winner_raw_first = str(s.get("raw_first_line") or "")
            win_logprobs = s.get("logprobs")
            winner_tle = s.get("tle")
            mlp = s.get("mean_logprob")
            winner_mean_logprob = float(mlp) if isinstance(mlp, (int, float)) else None
            break
    if not winner_action_exec and samples:
        s0 = samples[0]
        winner_action_exec = str(s0.get("action_exec") or "")
        winner_raw_first = str(s0.get("raw_first_line") or "")
        win_logprobs = s0.get("logprobs")
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
        lp_saved = [s.get("logprobs") if isinstance(s.get("logprobs"), list) else None for s in samples]
    else:
        lp_saved = None
    sample_blocks = [
        f"=== sample {int(s.get('sample_index', 0)) + 1}/{n} (first_line={str(s.get('raw_first_line') or '')!r}) ===\n{str(s.get('response') or '')}"
        for s in samples
    ]
    response_full = "\n\n".join(sample_blocks)
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
        "subcalls": [
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
                    winner_index is not None
                    and int(s.get("sample_index") or 0) == int(winner_index)
                ),
            }
            for s in samples
        ],
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
    """C2: Self-consistency sampling (N samples + majority vote)."""
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
        "C0": _c0_step_core,
        "C1": _c1_step_core,
        "C2": _c2_step_core,
    }
    fn = core_map.get(stage, _c0_step_core)
    vc_followup_logprobs = bool(save_vc_distributions) and (vc_mode or "").strip().lower() == "followup"

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

