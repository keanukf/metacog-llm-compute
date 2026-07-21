"""Parser that recovers the committed action from a native-thinking (C1/C2) model response.

Load-bearing for measurement integrity: TLE is defined over the *committed action* tokens, so which
span counts as "the action" is a dependent-variable question, not a formatting one. This parser is
allowed to affect only whether an action the model *already intended* gets recognized -- it must
never change which action is chosen, and never inject domain hints (the project-wide "red line"; see
CLAUDE.md). Hence the strict cascade of ``parse_method`` values, tried in order of trust:
``lmstudio_command_tag`` (explicit <command> wrapper) -> ``post_think`` (first real line after the
final </think>, the preregistered target) -> ``legacy_action_prefix`` (an "ACTION:" line) ->
``first_line_fallback`` -> ``embedded_action_fallback`` (last resort, mining a command mention out of
verbose reasoning). Admissibility in ``shared.assess_candidate_admissibility`` only trusts
``post_think``; the looser methods exist for trace diagnostics, not for gating a step.
"""

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
    """Structured parse of a C1/C2 reasoning output.

    Returns a dict with ``action`` ("" if unparsed), ``status`` (parsed|unparsed), ``parse_method``
    (which rule in the trust cascade fired -- see module docstring), ``reasoning_internal`` (inner
    text of the last complete <think>...</think>, kept for traces only, never scored), and ``raw``.
    """
    raw = cot_text or ""
    t = raw
    reasoning_internal = ""
    matches = list(_THINK_CAPTURE_RE.finditer(t))
    if matches:
        reasoning_internal = (matches[-1].group(1) or "").strip()

    cmd_match = re.search(r"<command>\s*([\s\S]*?)\s*</command>", t, flags=re.IGNORECASE)
    if cmd_match:
        action = normalize_action_line((cmd_match.group(1) or "").strip())
        if action and not _looks_like_tag_artifact(action):
            return {
                "action": action,
                "status": "parsed",
                "parse_method": "lmstudio_command_tag",
                "reasoning_internal": reasoning_internal,
                "raw": raw,
            }

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

    # Last resort: recover embedded command mentions from verbose reasoning.
    embedded = normalize_action_line(raw)
    if embedded and not _looks_like_tag_artifact(embedded):
        return {
            "action": embedded,
            "status": "parsed",
            "parse_method": "embedded_action_fallback",
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
