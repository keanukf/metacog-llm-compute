from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")
_INSTRUCTION_ECHO_RE = re.compile(
    r"(?i)^\s*(?:"
    r"just\s+the\s+command"
    r"|output\s+exactly\s+one\s+command(?:\s+on\s+a\s+single\s+line)?"
    r"|one\s+command\s+on\s+a\s+single\s+line"
    r"|no\s+reasoning,\s*no\s+tags,\s*no\s+preamble"
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
    for rx in (_QUOTED_ACTION_CANDIDATE_RE, _CUE_ACTION_CANDIDATE_RE):
        m = rx.search(search_text)
        if not m:
            continue
        cand = re.sub(r"\s+", " ", (m.group(1) or "").strip())
        if _INSTRUCTION_ECHO_RE.match(cand):
            return ""
        return cand
    return ""


def normalize_vote_key(action_line: str) -> str:
    s = strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def normalize_action_for_execution(action_line: str) -> str:
    s = strip_think_blocks(action_line or "").strip()
    if not s:
        return ""
    if s.upper().startswith("ACTION:"):
        s = s.split(":", 1)[1].strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s
