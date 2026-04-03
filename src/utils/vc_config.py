"""
Resolve VC / prompt settings for ``get_step_fn`` from experiment YAML.
"""
from __future__ import annotations

from typing import Any

_DEFAULT_TOH_PREFIX = (
    "You are solving Tower of Hanoi. Only the top disk on a peg may move; place it on an empty "
    "peg or on a larger-numbered disk. Each turn you must choose exactly one move from the "
    "line starting with \"Valid moves:\" in the task text—do not output any other peg pair. "
    "Respond with ONLY your move as X->Y using peg letters A, B, or C (e.g. A->C). "
    "Do not use disk numbers. Do not explain your reasoning."
)

_DEFAULT_TEXTWORLD_PREFIX = (
    "You are playing a parser-based text adventure (interactive fiction). "
    "Base each reply on the latest game text shown below. "
    "Output exactly one imperative command on a single line—typical forms include movement "
    "(go north), looking (look), and object use (take knife, open door). "
    "Do not add narration, quotes around the command, role-play, multiple commands, or reasoning. "
    "If the text includes a line starting with \"Valid commands this turn:\", choose one of "
    "those commands when possible."
)


def vc_step_fn_kwargs(config: dict, domain: str) -> dict[str, Any]:
    """
    Build kwargs for ``get_step_fn(..., **vc_step_fn_kwargs(config, domain))``.

    ``vc.prompt_prefix`` may be a dict mapping domain name -> instruction string.
    Default prefixes apply when the dict omits ``textworld`` / ``tower_of_hanoi`` (see
    ``_DEFAULT_TEXTWORLD_PREFIX`` / ``_DEFAULT_TOH_PREFIX``).
    ``vc.action_max_tokens`` may be a dict mapping domain -> int; omitted keys use
    domain defaults (32 for TextWorld and Tower of Hanoi) or ``inference.max_tokens`` for
    other domains when ``vc.use_inference_max_tokens_for_action`` is true (default).
    ``vc.textworld_action_stop`` overrides OpenAI-style ``stop`` for action generation on
    TextWorld (default ``["\\n"]``); use ``[]`` to disable.
    """
    vc = config.get("vc") or {}
    inf = config.get("inference") or {}
    mode = str(vc.get("mode", "inline"))
    prefixes = vc.get("prompt_prefix")
    prompt_prefix = ""
    if isinstance(prefixes, dict):
        if domain in prefixes:
            prompt_prefix = str(prefixes[domain] or "")
        elif domain == "tower_of_hanoi":
            prompt_prefix = str(vc.get("default_tower_of_hanoi_prefix", _DEFAULT_TOH_PREFIX))
        elif domain == "textworld":
            prompt_prefix = str(vc.get("default_textworld_prefix", _DEFAULT_TEXTWORLD_PREFIX))
    elif isinstance(prefixes, str) and prefixes.strip():
        prompt_prefix = prefixes.strip()
    if domain == "tower_of_hanoi" and not prompt_prefix:
        prompt_prefix = str(vc.get("default_tower_of_hanoi_prefix", _DEFAULT_TOH_PREFIX))
    if domain == "textworld" and not prompt_prefix:
        prompt_prefix = str(vc.get("default_textworld_prefix", _DEFAULT_TEXTWORLD_PREFIX))

    amap = vc.get("action_max_tokens")
    action_max_tokens: int | None = None
    if isinstance(amap, dict) and domain in amap and amap[domain] is not None:
        action_max_tokens = int(amap[domain])
    elif domain == "tower_of_hanoi":
        action_max_tokens = int(vc.get("tower_of_hanoi_default_action_max_tokens", 32))
    elif domain == "textworld":
        action_max_tokens = int(vc.get("textworld_default_action_max_tokens", 32))
    elif vc.get("use_inference_max_tokens_for_action", True):
        mt = inf.get("max_tokens")
        if mt is not None:
            action_max_tokens = int(mt)

    at = vc.get("action_temperature")
    if at is not None:
        action_temperature = float(at)
    else:
        action_temperature = float(inf.get("temperature", 0.3))

    action_stop: list[str] | None = None
    if domain == "textworld":
        if "textworld_action_stop" in vc:
            ts = vc["textworld_action_stop"]
            if ts is None or ts is False:
                action_stop = None
            elif isinstance(ts, (list, tuple)):
                action_stop = [str(x) for x in ts] if ts else None
            else:
                action_stop = [str(ts)]
        else:
            action_stop = ["\n"]

    return {
        "vc_mode": mode,
        "prompt_prefix": prompt_prefix,
        "action_max_tokens": action_max_tokens,
        "action_temperature": action_temperature,
        "action_stop": action_stop,
        "followup_max_tokens": int(vc.get("followup_max_tokens", 4)),
        "followup_temperature": float(vc.get("followup_temperature", 0.0)),
    }
