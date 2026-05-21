from __future__ import annotations

import re

from src.agent.compute_prompt_utils import normalize_action_line, strip_think_blocks

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


def parse_cot_action(cot_text: str) -> dict[str, str]:
    """Structured parse result for C1 CoT outputs."""
    raw = cot_text or ""
    t = raw
    reasoning_internal = ""
    matches = list(_THINK_CAPTURE_RE.finditer(t))
    if matches:
        reasoning_internal = (matches[-1].group(1) or "").strip()

    close = t.casefold().rfind("</think>")
    if close >= 0:
        after = t[close + len("</think>") :]
        for line in (after or "").splitlines():
            s = (line or "").strip()
            if not s or _looks_like_tag_artifact(s):
                continue
            if re.match(r"(?i)^(reasoning|thoughts|analysis|plan)\s*:", s):
                continue
            action = normalize_action_line(s)
            if action and not _looks_like_tag_artifact(action):
                return {
                    "action": action,
                    "status": "parsed",
                    "parse_method": "post_think",
                    "reasoning_internal": reasoning_internal,
                    "raw": raw,
                }

    for line in (t or "").splitlines():
        ls = (line or "").strip()
        if ls.upper().startswith("ACTION:"):
            action = normalize_action_line(ls.split(":", 1)[1].strip())
            if action and not _looks_like_tag_artifact(action):
                return {
                    "action": action,
                    "status": "parsed",
                    "parse_method": "legacy_action_prefix",
                    "reasoning_internal": reasoning_internal,
                    "raw": raw,
                }

    cleaned = strip_think_blocks(t)
    cleaned = re.sub(r"</?\s*think\s*>", "", cleaned, flags=re.IGNORECASE)
    for line in (cleaned or "").splitlines():
        s = (line or "").strip()
        if not s or _looks_like_tag_artifact(s):
            continue
        if re.match(r"(?i)^(reasoning|thoughts|analysis|plan)\s*:", s):
            continue
        action = normalize_action_line(s)
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
