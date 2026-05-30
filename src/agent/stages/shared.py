"""
Compute stages: C0 (direct + logprobs), C1 (CoT + verify), C2 (self-consistency / majority vote).
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from src.agent import compute_prompt_utils, cot_parser
from src.signals import verbalized_confidence

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

_SINGLE_LINE_OUTPUT_INSTRUCTION: str = (
    "Write one valid game action on a single line (examples: go north, take key, inventory). "
    "Do not explain, add XML tags, or repeat these instructions."
)

# Shared generation instruction body (reused to keep C0/C1 parity stable over time).
_C0_GENERATION_INSTRUCTION: str = (
    "Choose one valid game action from <task>, <history>, and <state>."
)

DEFAULT_C1_VERIFY_INSTRUCTION = (
    "You are doing a verification pass over a draft_action.\n"
    "Use <task>, <history>, and <state> as the only source of truth.\n\n"
    'If <draft_status> is "parsed":\n'
    "- Check draft_action against the task constraints and current state.\n"
    "- If it is valid and useful, output it again unchanged.\n"
    "- If it is invalid, output a corrected single command.\n\n"
    'If <draft_status> is "unparsed":\n'
    "- Ignore draft_action.\n"
    f"- {_C0_GENERATION_INSTRUCTION}\n\n"
    f"{_SINGLE_LINE_OUTPUT_INSTRUCTION}\n"
)


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_CAPTURE_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def _looks_like_tag_artifact(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    sl = s.casefold()
    if sl.startswith("<") and sl.endswith(">") and " " not in s:
        return True
    if "<think" in sl or "</think" in sl:
        return True
    if "<draft" in sl or "</draft" in sl:
        return True
    if "<task" in sl or "<history" in sl or "<state" in sl:
        return True
    return False


def _parse_cot_action(cot_text: str) -> dict[str, str]:
    """
    Structured parse result for C1 CoT outputs.

    Returns:
      {
        "action": str,                # "" if unparsed
        "status": "parsed" | "unparsed",
        "parse_method": "post_think" | "legacy_action_prefix" | "first_line_fallback" | "none",
        "reasoning_internal": str,    # last complete <think>...</think> inner content (trace-only)
        "raw": str,
      }
    """
    raw = cot_text or ""
    t = raw

    reasoning_internal = ""
    matches = list(_THINK_CAPTURE_RE.finditer(t))
    if matches:
        reasoning_internal = (matches[-1].group(1) or "").strip()

    # 1) Preferred: first plausible line after the final </think>
    close = t.casefold().rfind("</think>")
    if close >= 0:
        after = t[close + len("</think>") :]
        for line in (after or "").splitlines():
            s = (line or "").strip()
            if not s:
                continue
            if _looks_like_tag_artifact(s):
                continue
            if re.match(r"(?i)^(reasoning|thoughts|analysis|plan)\s*:", s):
                continue
            action = _normalize_action_line(s)
            if action and not _looks_like_tag_artifact(action):
                return {
                    "action": action,
                    "status": "parsed",
                    "parse_method": "post_think",
                    "reasoning_internal": reasoning_internal,
                    "raw": raw,
                }

    # 2) Legacy: ACTION: prefix
    for line in (t or "").splitlines():
        ls = (line or "").strip()
        if ls.upper().startswith("ACTION:"):
            action = ls.split(":", 1)[1].strip()
            action = _normalize_action_line(action)
            if action and not _looks_like_tag_artifact(action):
                return {
                    "action": action,
                    "status": "parsed",
                    "parse_method": "legacy_action_prefix",
                    "reasoning_internal": reasoning_internal,
                    "raw": raw,
                }

    # 3) Best-effort fallback: remove any complete <think>...</think> blocks, then strip think tags
    # (even if incomplete) and take the first plausible remaining line.
    cleaned = _strip_think_blocks(t)
    cleaned = re.sub(r"</?\s*think\s*>", "", cleaned, flags=re.IGNORECASE)
    for line in (cleaned or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        if _looks_like_tag_artifact(s):
            continue
        if re.match(r"(?i)^(reasoning|thoughts|analysis|plan)\s*:", s):
            continue
        action = _normalize_action_line(s)
        if action and not _looks_like_tag_artifact(action):
            return {
                "action": action,
                "status": "parsed",
                "parse_method": "first_line_fallback",
                "reasoning_internal": reasoning_internal,
                "raw": raw,
            }

    return {
        "action": "",
        "status": "unparsed",
        "parse_method": "none",
        "reasoning_internal": reasoning_internal,
        "raw": raw,
    }


# Use dedicated parser module while keeping local symbol for compatibility.
_parse_cot_action = cot_parser.parse_cot_action


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
        obs = (
            _unwrap_history_entry(history[idx + 1], "OBSERVATION:")
            if (idx + 1) < len(history)
            else None
        )
        if act is not None and obs is not None:
            step_body = "\n".join(
                [
                    f"<action>{act.rstrip()}</action>",
                    f"<observation>{obs.rstrip()}</observation>",
                ]
            )
            parts.append(_xml_block("step", step_body, attrs=f'index="{step_index}"'))
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


# Route prompt and action-normalization helpers through dedicated module while
# keeping local names stable for callers and tests during transition.
_xml_block = compute_prompt_utils.xml_block
_unwrap_history_entry = compute_prompt_utils.unwrap_history_entry
_history_to_xml = compute_prompt_utils.history_to_xml
_strip_think_blocks = compute_prompt_utils.strip_think_blocks
_build_prompt = compute_prompt_utils.build_prompt
_extract_first_line = compute_prompt_utils.extract_first_line
_normalize_action_line = compute_prompt_utils.normalize_action_line
_normalize_vote_key = compute_prompt_utils.normalize_vote_key
_normalize_action_for_execution = compute_prompt_utils.normalize_action_for_execution


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
    parsed = _parse_cot_action(cot_text or "")
    return str(parsed.get("action") or "")


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
        # VC prompt asks for a single scalar (0-100). Cut trailing explanations early.
        "stop": ["\n"],
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
