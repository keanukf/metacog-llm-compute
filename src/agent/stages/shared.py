"""
Compute stages: C0 (direct + logprobs), C1 (single reasoning call), C2 (self-consistency / majority vote).
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from src.agent import compute_prompt_utils, cot_parser
from src.signals import token_entropy, verbalized_confidence
from src.utils.inference.lmstudio.wrapper import attach_lmstudio_diagnostics_to_subcalls

# (action, tle, vc, tokens_used, lm_calls, action_logprobs_raw|None, vc_detail|None, prompt_full, response_full, call_detail|None)
StepReturn = tuple[str, dict[str, float] | None, float | None, int, int, Any, Any, str, str, Any]

# Present in every VC follow-up prompt — tests and mocks can detect the second call without coupling to wording details.
VC_FOLLOWUP_PROMPT_MARKER = "<output_to_judge>"

# Default when YAML omits ``vc.followup_instruction`` (overridable per experiment).
DEFAULT_VC_FOLLOWUP_INSTRUCTION = (
    "How likely is the chosen action to be correct in this situation?\n"
    "Respond with only a single integer between 0 and 100,\n"
    "where 0 means certainly wrong and 100 means certainly correct."
)

_SINGLE_LINE_OUTPUT_INSTRUCTION: str = (
    "Write one valid game action on a single line, in the format already shown above. "
    "Do not explain, add XML tags, or repeat these instructions."
)

# Shared between C1 and C2 -- both stages reason inside a native <think> block before committing
# an action, so both use the same instruction text (previously C2 used the no-thinking
# _SINGLE_LINE_OUTPUT_INSTRUCTION by accident of code reuse from C0; unified 2026-07-21, see
# docs/consistency_log.md).
_REASONING_OUTPUT_INSTRUCTION: str = (
    "Before answering, briefly reason inside <think>...</think> tags. "
    "After </think>, write one valid game action on its own line, in the format already shown above."
)

# Shared generation instruction body (reused to keep C0/C1 parity stable over time).
_C0_GENERATION_INSTRUCTION: str = (
    "Choose one valid game action from <task>, <history>, and <state>."
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
    judged_context: str = "action_only",
    raw_action_completion: str | None,
    cot_text: str | None,
    verify_completion: str | None,
    c2_n_samples: int | None,
    c2_sample_first_lines: list[str] | None,
    c2_winner_completion: str | None = None,
    followup_cot_max_chars: int,
    raw_completion_max_chars: int,
) -> str:
    """Text block for VC follow-up (stage-dependent when ``judged_context=full``)."""
    al = (action_line or "").strip()
    tag = (stage_tag or "C0").strip().upper()
    mode = (judged_context or "action_only").strip().lower()
    if mode == "action_only":
        return f"[{tag}] {al}"
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
        return (
            "--- Chain-of-thought (your reasoning for this turn) ---\n"
            f"{cot_t}\n\n"
            f"--- Final command (executed) ---\n{al}"
        )
    if tag == "C2":
        n = int(c2_n_samples or 0)
        winner_c = (c2_winner_completion or cot_text or "").strip()
        winner_t = (
            _truncate_text(winner_c, max_chars=followup_cot_max_chars) if winner_c else "(empty)"
        )
        lines = c2_sample_first_lines or []
        summary = "\n".join(f"  sample {i + 1}: {fl!r}" for i, fl in enumerate(lines))
        parts = [
            f"--- Selected command (majority vote among {n} samples) ---\n{al}",
            "--- Reasoning from the winning sample ---",
            winner_t,
        ]
        if summary:
            parts.append(f"--- All sample first-line commands ---\n{summary}")
        return "\n\n".join(parts)
    return f"--- Chosen command ---\n{al}"


def _build_vc_followup_prompt(
    observation: str,
    history: list[str],
    prompt_prefix: str,
    *,
    stage_tag: str,
    action_line: str,
    instruction: str,
    judged_context: str = "action_only",
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    c2_winner_completion: str | None = None,
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
        judged_context=judged_context,
        raw_action_completion=raw_action_completion,
        cot_text=cot_text,
        verify_completion=verify_completion,
        c2_n_samples=c2_n_samples,
        c2_sample_first_lines=c2_sample_first_lines,
        c2_winner_completion=c2_winner_completion,
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
    judged_context: str = "action_only",
    retry_on_parse_failure: bool = True,
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    c2_winner_completion: str | None = None,
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
        judged_context=judged_context,
        raw_action_completion=raw_action_completion,
        cot_text=cot_text,
        verify_completion=verify_completion,
        c2_n_samples=c2_n_samples,
        c2_sample_first_lines=c2_sample_first_lines,
        c2_winner_completion=c2_winner_completion,
        followup_cot_max_chars=followup_cot_max_chars,
        followup_max_context_chars=followup_max_context_chars,
        raw_completion_max_chars=raw_completion_max_chars,
    )
    gen_kw = {
        "max_tokens": int(followup_max_tokens),
        "temperature": float(followup_temperature),
        "enable_thinking": False,
        "stop": ["\n"],
    }

    def _one_call(temp: float) -> tuple[str, Any, dict[str, Any]]:
        kw = dict(gen_kw)
        kw["temperature"] = float(temp)
        if request_logprobs:
            text, logprobs = model.generate(prompt, logprobs=True, **kw)
        else:
            text, logprobs = model.generate(prompt, logprobs=False, **kw)
        detail = verbalized_confidence.extract_vc_from_followup(prompt, text, logprobs)
        return text, logprobs, detail

    text, _logprobs, detail = _one_call(followup_temperature)
    vc_val = detail.get("vc_value")
    retry_used = False
    extra_calls = 1
    extra_tokens = int(detail.get("vc_tokens_used") or 0)
    if vc_val is None and retry_on_parse_failure:
        _text2, _logprobs2, detail_retry = _one_call(0.0)
        detail = dict(detail_retry)
        retry_used = True
        extra_calls = 2
        extra_tokens += int(detail_retry.get("vc_tokens_used") or 0)
        vc_val = detail.get("vc_value")
    detail["retry_used"] = retry_used
    vc_f: float | None
    if isinstance(vc_val, (int, float)):
        vc_f = float(vc_val)
    else:
        vc_f = None
    return vc_f, detail, extra_tokens, extra_calls


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
    judged_context: str = "action_only",
    retry_on_parse_failure: bool = True,
    raw_action_completion: str | None = None,
    cot_text: str | None = None,
    verify_completion: str | None = None,
    c2_n_samples: int | None = None,
    c2_sample_first_lines: list[str] | None = None,
    c2_winner_completion: str | None = None,
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
            judged_context=judged_context,
            retry_on_parse_failure=retry_on_parse_failure,
            raw_action_completion=raw_action_completion,
            cot_text=cot_text,
            verify_completion=verify_completion,
            c2_n_samples=c2_n_samples,
            c2_sample_first_lines=c2_sample_first_lines,
            c2_winner_completion=c2_winner_completion,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            request_logprobs=vc_followup_logprobs,
            followup_max_context_chars=followup_max_context_chars,
            followup_cot_max_chars=followup_cot_max_chars,
            raw_completion_max_chars=raw_completion_max_chars,
        )
    vc = verbalized_confidence.parse_confidence(inline_text)
    return vc, None, 0, 0


# --------------------------------------------------------------------------------------------
# Shared reasoning engine (C1 = 1 candidate / no vote, C2 = N candidates / majority vote).
#
# Both stages generate a native <think>...</think> reasoning block then commit one action; C2
# additionally draws several candidates and votes. Before 2026-07-21 this admissibility/parsing
# logic existed only in C2 (assess_c2_sample_admissibility); C1 used a separate, more naive path
# (_normalize_action_line) with no closed-thinking check, so a candidate whose reasoning never
# closed got its literal "<think>" text parsed as the action, and TLE got computed over the
# opening-tag tokens instead of a real decision point (docs/consistency_log.md, 2026-07-20/21).
# --------------------------------------------------------------------------------------------

_THINK_CLOSE_TAG = "</think>"


def _thinking_block_closed(text: str) -> bool:
    return _THINK_CLOSE_TAG.casefold() in (text or "").casefold()


def majority_vote(
    vote_keys_in_order: list[str],
    *,
    rng: random.Random | None,
) -> tuple[str, bool, dict[str, int]]:
    """
    Majority vote over pre-normalized vote keys.

    Returns (winning_key, tie_broken, counts). Degenerates correctly for a single key (C1's
    n_samples=1 case): one key, one vote, no tie.
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


def assess_candidate_admissibility(response: str) -> dict[str, Any]:
    """
    A reasoning candidate is admissible only with a closed thinking block and a parseable
    post-think action. Shared by C1 (single candidate, no vote) and C2 (N candidates + vote).
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
    parsed = _parse_cot_action(text)
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


def _build_reasoning_candidate(
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
    sample.update(assess_candidate_admissibility(str(text or "")))
    # TLE only for an admissible candidate -- otherwise the "action window" the extractor would
    # slice is undefined (no closed think block means no real post-decision token span), and
    # computing TLE on the opening-tag tokens would be a genuine signal-quality bug, not a
    # harmless fallback.
    lp = sample.get("logprobs")
    raw_text = str(sample.get("response") or "")
    if sample["admissible"] and lp:
        sample["tle"] = token_entropy.extract_action_tle_from_response(raw_text, lp)
    else:
        sample["tle"] = None
    if lp and isinstance(lp, list):
        vals: list[float] = []
        for x in lp:
            if not isinstance(x, dict):
                continue
            lp_val = x.get("logprob")
            if isinstance(lp_val, (int, float)):
                vals.append(float(lp_val))
        sample["mean_logprob"] = (sum(vals) / len(vals)) if vals else None
    else:
        sample["mean_logprob"] = None
    return sample


def reasoning_step_core(
    observation: str,
    history: list[str],
    model: Any,
    *,
    n_samples: int,
    sample_temperature: float,
    cot_max_tokens: int | None,
    stage_tag: str,
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
    vc_judged_context: str = "action_only",
    vc_retry_on_parse_failure: bool = True,
) -> StepReturn:
    """
    N reasoning candidates (native <think> block) -> majority vote. C1 calls this with
    n_samples=1 (the vote is then trivial: the one admissible candidate, or none); C2 calls it
    with n_samples>1 for genuine self-consistency. ``action_logprobs_raw`` is always returned as
    one list per candidate (``[cand0_logprobs, cand1_logprobs, ...]``) -- callers that need a
    single flat list (C1's external contract) unwrap it themselves.
    """
    prompt = (
        f"{_build_prompt(observation, history, prompt_prefix)}\n\n{_REASONING_OUTPUT_INSTRUCTION}"
    )
    act_tok = int(action_max_tokens) if action_max_tokens is not None else 32
    sample_max_tokens = int(cot_max_tokens) if cot_max_tokens is not None else max(128, act_tok * 2)
    if sample_max_tokens <= 0:
        sample_max_tokens = max(128, act_tok * 2)
    gen_kw = _action_generate_kwargs(action_max_tokens, float(sample_temperature), None)
    gen_kw["max_tokens"] = sample_max_tokens
    gen_kw["enable_thinking"] = True

    samples: list[dict[str, Any]] = []
    total_tokens = 0
    n = max(1, int(n_samples))
    if hasattr(model, "generate_many") and callable(getattr(model, "generate_many")):
        outs = model.generate_many(prompt, n=n, logprobs=True, **gen_kw)
        for i, (text, logprobs) in enumerate(outs):
            samples.append(
                _build_reasoning_candidate(
                    sample_index=i, prompt=prompt, text=text, logprobs=logprobs
                )
            )
            total_tokens += len(logprobs) if logprobs else 0
    else:
        for i in range(n):
            text, logprobs = model.generate(prompt, logprobs=True, **gen_kw)
            samples.append(
                _build_reasoning_candidate(
                    sample_index=i, prompt=prompt, text=text, logprobs=logprobs
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

    winner_sample = samples[winner_index] if winner_index is not None else {}
    winner_action_exec = str(winner_sample.get("action_exec") or "")
    winner_raw_first = str(winner_sample.get("raw_first_line") or "")
    winner_tle: dict[str, float] | None = (
        winner_sample.get("tle") if winner_index is not None else None
    )
    winner_mean_logprob: float | None = None
    if winner_index is not None:
        mlp = winner_sample.get("mean_logprob")
        winner_mean_logprob = float(mlp) if isinstance(mlp, (int, float)) else None
    winner_completion = str(winner_sample.get("response") or "")

    vc, vc_detail, extra_tok, extra_calls = None, None, 0, 0
    if step_outcome == "vote" and winner_action_exec:
        admissible_first_lines = [str(s.get("raw_first_line") or "") for s in admissible_samples]
        vc, vc_detail, extra_tok, extra_calls = _resolve_vc(
            model,
            vc_mode=vc_mode,
            inline_text=winner_completion,
            observation=observation,
            history=history,
            prompt_prefix=prompt_prefix,
            stage_tag=stage_tag,
            action_line=winner_action_exec,
            vc_followup_instruction=vc_followup_instruction,
            judged_context=vc_judged_context,
            retry_on_parse_failure=vc_retry_on_parse_failure,
            # cot_text feeds C1's "full" judged-context branch, c2_winner_completion feeds C2's --
            # both point at the same winning completion so either stage's branch in
            # _build_model_output_to_judge_section resolves correctly regardless of stage_tag.
            cot_text=winner_completion,
            c2_n_samples=n_admissible,
            c2_sample_first_lines=admissible_first_lines,
            c2_winner_completion=winner_completion,
            vc_followup_logprobs=vc_followup_logprobs,
            followup_max_tokens=followup_max_tokens,
            followup_temperature=followup_temperature,
            followup_max_context_chars=followup_max_context_chars,
            followup_cot_max_chars=followup_cot_max_chars,
            raw_completion_max_chars=vc_raw_completion_max_chars,
        )

    total_tokens += extra_tok
    lm_calls = int(n) + extra_calls
    if save_action_logprobs:
        lp_saved: list[Any] | None = [
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
        "stage": stage_tag,
        "method": "self_consistency_majority_vote" if n > 1 else "single_reasoning_call",
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
