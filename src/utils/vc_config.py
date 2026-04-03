"""
Resolve VC / prompt settings for ``get_step_fn`` from experiment YAML.
"""
from __future__ import annotations

from typing import Any

_DEFAULT_TOH_PREFIX = (
    "You are solving a Tower of Hanoi puzzle. Respond with ONLY your move in the format "
    "X->Y (e.g., A->C). Do not explain your reasoning."
)


def vc_step_fn_kwargs(config: dict, domain: str) -> dict[str, Any]:
    """
    Build kwargs for ``get_step_fn(..., **vc_step_fn_kwargs(config, domain))``.

    ``vc.prompt_prefix`` may be a dict mapping domain name -> instruction string.
    ``vc.action_max_tokens`` may be a dict mapping domain -> int; omitted keys use
    ``inference.max_tokens`` when ``vc.use_inference_max_tokens_for_action`` is true (default),
    else None (model default).
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
    elif isinstance(prefixes, str) and prefixes.strip():
        prompt_prefix = prefixes.strip()
    if domain == "tower_of_hanoi" and not prompt_prefix:
        prompt_prefix = str(vc.get("default_tower_of_hanoi_prefix", _DEFAULT_TOH_PREFIX))

    amap = vc.get("action_max_tokens")
    action_max_tokens: int | None = None
    if isinstance(amap, dict) and domain in amap and amap[domain] is not None:
        action_max_tokens = int(amap[domain])
    elif domain == "tower_of_hanoi":
        action_max_tokens = int(vc.get("tower_of_hanoi_default_action_max_tokens", 32))
    elif vc.get("use_inference_max_tokens_for_action", True):
        mt = inf.get("max_tokens")
        if mt is not None:
            action_max_tokens = int(mt)

    at = vc.get("action_temperature")
    if at is not None:
        action_temperature = float(at)
    else:
        action_temperature = float(inf.get("temperature", 0.3))

    return {
        "vc_mode": mode,
        "prompt_prefix": prompt_prefix,
        "action_max_tokens": action_max_tokens,
        "action_temperature": action_temperature,
        "followup_max_tokens": int(vc.get("followup_max_tokens", 8)),
        "followup_temperature": float(vc.get("followup_temperature", 0.0)),
    }
