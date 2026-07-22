"""Prompt assembly and action-string normalization shared by all compute stages.

This module owns the frozen XML-tag prompt schema (``<task>``, ``<history>`` with nested
``<step index="...">``/``<action>``/``<observation>``, ``<state>``). The schema is a preregistered
operationalization choice, not cosmetic: explicit section markers keep the growing episode history
unambiguous over long prompts and give the TLE action window a stable boundary (thesis §5.2.B). The
schema is never mixed with other separators and never varied per domain/cell.

It also owns the three action-normalization functions, which intentionally differ:
``normalize_vote_key`` (casefolded, for self-consistency agreement stats), ``normalize_action_for_
execution`` (NOT casefolded, so surface forms like Tower-of-Hanoi "A->C" reach the env intact), and
``normalize_action_line`` (extraction + instruction-echo guard + verbose-output fallbacks).
"""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")
_INSTRUCTION_ECHO_RE = re.compile(
    r"(?i)^\s*(?:"
    r"just\s+the\s+command"
    r"|just\s+the\s+action"
    r"|just\s+the\s+next\s+action"
    r"|output\s+exactly\s+one\s+(?:imperative\s+)?command(?:\s+on\s+a\s+single\s+line)?"
    r"|one\s+command\s+on\s+a\s+single\s+line"
    r"|no\s+reasoning,\s*no\s+tags,\s*no\s+preamble"
    r"|write\s+one\s+valid\s+game\s+action\s+on\s+a\s+single\s+line"
    r"|do\s+not\s+explain,\s*add\s+xml\s+tags"
    r")\.?\s*$"
)
_ACTION_PATTERN = (
    r"go\s+(?:north|south|east|west|up|down)"
    r"|take\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|drop\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|examine\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|open\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|close\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|put\s+[a-z0-9][a-z0-9 _-]{0,40}\s+in\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|cook\s+[a-z0-9][a-z0-9 _-]{0,40}\s+with\s+[a-z0-9][a-z0-9 _-]{0,40}"
    r"|inventory"
    r"|look"
    r"|[ABC]\s*->\s*[ABC]"
)
_QUOTED_ACTION_CANDIDATE_RE = re.compile(rf"(?i)[\"']\s*({_ACTION_PATTERN})\s*[\"']")
_CUE_ACTION_CANDIDATE_RE = re.compile(
    rf"(?i)\b(?:action|command)\b(?:\s+\w+){{0,3}}\s+"
    rf"(?:is|should\s+be|would\s+be|must\s+be|to\s+use)\s*[:=]?\s*[\"']?\s*"
    rf"({_ACTION_PATTERN})\b"
)
_DECISION_ACTION_CANDIDATE_RE = re.compile(
    rf"(?i)\b(?:need(?:s)?\s+to|should|must|next\s+step\s+is\s+to)\s+"
    rf"({_ACTION_PATTERN})\b"
)


def xml_block(tag: str, content: str, *, attrs: str = "") -> str:
    attr_text = f" {attrs}" if attrs else ""
    return f"<{tag}{attr_text}>\n{content or ''}\n</{tag}>"


def unwrap_history_entry(entry: str, prefix: str) -> str | None:
    line = entry or ""
    if line.startswith(prefix):
        rest = line.split(":", 1)[1]
        if rest.startswith(" "):
            rest = rest[1:]
        return rest
    return None


def history_to_xml(history: list[str]) -> str:
    if not history:
        return xml_block("history", "")

    parts: list[str] = []
    first_obs = unwrap_history_entry(history[0], "OBSERVATION:")
    if first_obs is not None:
        parts.append(xml_block("reset_observation", (first_obs or "").rstrip()))

    idx = 1
    step_index = 1
    while idx < len(history):
        entry = history[idx] or ""

        if entry.startswith("PINNED RECIPE:"):
            recipe = unwrap_history_entry(entry, "PINNED RECIPE:")
            if recipe is None:
                recipe = entry.split(":", 1)[1].lstrip() if ":" in entry else entry
            parts.append(xml_block("pinned_recipe", (recipe or "").rstrip()))
            idx += 1
            continue

        act = unwrap_history_entry(entry, "ACTION:")
        obs = (
            unwrap_history_entry(history[idx + 1], "OBSERVATION:")
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
            parts.append(xml_block("step", step_body, attrs=f'index="{step_index}"'))
            step_index += 1
            idx += 2
            continue

        parts.append(xml_block("history_item", entry.rstrip()))
        idx += 1

    return xml_block("history", "\n\n".join(parts))


def strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "")


def build_prompt(observation: str, history: list[str], prompt_prefix: str) -> str:
    """Assemble the action-generation prompt: domain ``<task>`` + ``<history>`` + current ``<state>``.

    Do NOT lstrip observations: TextWorld observations carry fixed-width/ASCII-art structure in
    leading spaces, so only trailing newlines are normalized (rstrip). If the caller already stored
    the current observation as the last history entry (true for the step-0 reset text, which can be
    very large), it is not appended again as ``<state>``.
    """

    def _rstrip(s: str) -> str:
        return (s or "").rstrip()

    obs_now = _rstrip(observation or "")
    include_current_obs = True
    if history:
        last_line = history[-1] or ""
        last_obs_raw = unwrap_history_entry(last_line, "OBSERVATION:")
        last_obs = _rstrip(last_obs_raw) if last_obs_raw is not None else None
        if last_obs is not None and last_obs == obs_now:
            include_current_obs = False

    pfx = (prompt_prefix or "").strip()
    parts: list[str] = []
    if pfx:
        parts.append(xml_block("task", pfx))
    if history:
        parts.append(history_to_xml(history))
    if include_current_obs and (observation is not None):
        parts.append(xml_block("state", observation or ""))
    return "\n\n".join(parts).rstrip()


def extract_first_line(text: str) -> str:
    for line in (text or "").strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return (text or "").strip()


def normalize_action_line(text: str) -> str:
    """Extract a single executable action line from raw model text.

    Strips any <think> block, takes the first non-empty line, drops an "ACTION:" prefix, and blanks
    it if it is merely an echo of the output instruction. Only if nothing usable remains does it fall
    back to mining an embedded command out of verbose output (quoted, cue-phrase, or decision-phrase
    patterns) -- these fallbacks recover an intended action from chatty completions without ever
    supplying one the model did not produce. Returns "" when no action can be recovered.
    """
    stripped = strip_think_blocks(text or "")
    line = extract_first_line(stripped)
    if line.upper().startswith("ACTION:"):
        line = line.split(":", 1)[1].strip()
    s = (line or "").strip()
    if not s:
        s = ""
    # Some endpoints occasionally echo instruction text instead of an action.
    if _INSTRUCTION_ECHO_RE.match(s):
        s = ""
    if s:
        return s

    # Fallback for verbose verify outputs: recover the first embedded action candidate.
    # Example: '... correct command should be "go east" ...'
    search_text = stripped if stripped.strip() else (text or "")
    for rx in (
        _QUOTED_ACTION_CANDIDATE_RE,
        _CUE_ACTION_CANDIDATE_RE,
        _DECISION_ACTION_CANDIDATE_RE,
    ):
        matches = list(rx.finditer(search_text))
        if not matches:
            continue
        cand = re.sub(r"\s+", " ", (matches[-1].group(1) or "").strip())
        if _INSTRUCTION_ECHO_RE.match(cand):
            return ""
        return cand
    return ""


def normalize_vote_key(action_line: str) -> str:
    """Canonical key for C2 self-consistency voting/agreement (casefolded).

    Collapses only surface-form variation (trailing punctuation, whitespace, case) so equivalent
    commands are not counted as disagreement; internal punctuation is deliberately preserved so
    genuinely different actions stay distinct. Casefolds because vote agreement is case-insensitive.
    """
    s = strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def normalize_action_for_execution(action_line: str) -> str:
    """Canonical action string passed to ``env.step`` -- same cleanup as the vote key but NOT
    casefolded, so case-sensitive domains (e.g. Tower of Hanoi "A->C") keep their expected form.
    """
    s = strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s
