"""
Resolve VC / prompt settings for ``get_step_fn`` from experiment YAML.
"""
from __future__ import annotations

import warnings
from typing import Any

# Legacy-only fallback defaults for old configs that omitted any prefix configuration.
# New configs should provide these via config["domain_prompts"][domain]["prefix"].
_LEGACY_DEFAULT_PREFIX_BY_DOMAIN: dict[str, str] = {
    "tower_of_hanoi": (
        "You are solving Tower of Hanoi. Only the top disk on a peg may move; place it on an empty "
        "peg or on a larger-numbered disk. Each turn you must choose exactly one move from the "
        "line starting with \"Valid moves:\" in the task text—do not output any other peg pair. "
        "Respond with ONLY your move as X->Y using peg letters A, B, or C (e.g. A->C). "
        "Do not use disk numbers. Do not explain your reasoning."
    ),
    "textworld": (
        "You are playing a parser-based text adventure (interactive fiction). "
        "Base each reply on the latest game text shown below. "
        "Output exactly one imperative command on a single line—typical forms include movement "
        "(go north), looking (look), and object use (take knife, open door). "
        "Do not add narration, quotes around the command, role-play, multiple commands, or reasoning. "
        "If the text includes a line starting with \"Valid commands this turn:\", choose one of "
        "those commands when possible."
    ),
}


def resolve_step_fn_kwargs(config: dict, domain: str) -> dict[str, Any]:
    """
    Build kwargs for ``get_step_fn(..., **resolve_step_fn_kwargs(config, domain))``.

    This function intentionally separates:

    - Domain prompting / action generation (environment-specific):
      ``config["domain_prompts"][domain]`` with keys:
        - ``prefix`` (str)
        - ``action_max_tokens`` (int, optional)
        - ``action_temperature`` (float, optional)
        - ``action_stop`` (list[str] | str | null, optional)

    - VC signal extraction (signal-specific):
      ``config["vc"]`` with keys:
        - ``mode`` (inline | followup | none)
        - ``followup_max_tokens`` (int)
        - ``followup_temperature`` (float)

    Backward compatibility:
    Older configs used ``vc.prompt_prefix`` / ``vc.action_max_tokens`` / ``vc.textworld_action_stop``
    and default-* keys. Those are supported with deprecation warnings.
    """
    vc = config.get("vc") or {}
    inf = config.get("inference") or {}
    mode = str(vc.get("mode", "inline")).strip() or "inline"

    domain_prompts = config.get("domain_prompts") or {}
    dom_cfg = domain_prompts.get(domain) if isinstance(domain_prompts, dict) else None

    prompt_prefix = ""
    if isinstance(dom_cfg, dict) and (dom_cfg.get("prefix") is not None):
        prompt_prefix = str(dom_cfg.get("prefix") or "").strip()
    else:
        # Legacy: vc.prompt_prefix dict or string
        prefixes = vc.get("prompt_prefix")
        if isinstance(prefixes, dict):
            if domain in prefixes:
                warnings.warn(
                    "Deprecated config: use top-level domain_prompts.<domain>.prefix "
                    "instead of vc.prompt_prefix.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                prompt_prefix = str(prefixes[domain] or "").strip()
        elif isinstance(prefixes, str) and prefixes.strip():
            warnings.warn(
                "Deprecated config: use top-level domain_prompts.<domain>.prefix "
                "instead of vc.prompt_prefix.",
                DeprecationWarning,
                stacklevel=2,
            )
            prompt_prefix = prefixes.strip()

        if not prompt_prefix:
            # Legacy: vc.default_*_prefix
            if domain == "tower_of_hanoi" and "default_tower_of_hanoi_prefix" in vc:
                warnings.warn(
                    "Deprecated config: use domain_prompts.tower_of_hanoi.prefix "
                    "instead of vc.default_tower_of_hanoi_prefix.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                prompt_prefix = str(vc.get("default_tower_of_hanoi_prefix") or "").strip()
            if domain == "textworld" and "default_textworld_prefix" in vc:
                warnings.warn(
                    "Deprecated config: use domain_prompts.textworld.prefix "
                    "instead of vc.default_textworld_prefix.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                prompt_prefix = str(vc.get("default_textworld_prefix") or "").strip()

        if not prompt_prefix and domain in _LEGACY_DEFAULT_PREFIX_BY_DOMAIN:
            warnings.warn(
                f"No domain_prompts.{domain}.prefix configured; falling back to legacy in-code default. "
                "Add domain_prompts to your YAML config to make this explicit.",
                DeprecationWarning,
                stacklevel=2,
            )
            prompt_prefix = _LEGACY_DEFAULT_PREFIX_BY_DOMAIN[domain]

    action_max_tokens: int | None = None
    if isinstance(dom_cfg, dict) and dom_cfg.get("action_max_tokens") is not None:
        action_max_tokens = int(dom_cfg["action_max_tokens"])
    else:
        # Legacy: vc.action_max_tokens map and default action caps
        amap = vc.get("action_max_tokens")
        if isinstance(amap, dict) and domain in amap and amap[domain] is not None:
            warnings.warn(
                "Deprecated config: use domain_prompts.<domain>.action_max_tokens "
                "instead of vc.action_max_tokens.",
                DeprecationWarning,
                stacklevel=2,
            )
            action_max_tokens = int(amap[domain])
        elif domain == "tower_of_hanoi" and "tower_of_hanoi_default_action_max_tokens" in vc:
            warnings.warn(
                "Deprecated config: use domain_prompts.tower_of_hanoi.action_max_tokens "
                "instead of vc.tower_of_hanoi_default_action_max_tokens.",
                DeprecationWarning,
                stacklevel=2,
            )
            action_max_tokens = int(vc.get("tower_of_hanoi_default_action_max_tokens", 32))
        elif domain == "textworld" and "textworld_default_action_max_tokens" in vc:
            warnings.warn(
                "Deprecated config: use domain_prompts.textworld.action_max_tokens "
                "instead of vc.textworld_default_action_max_tokens.",
                DeprecationWarning,
                stacklevel=2,
            )
            action_max_tokens = int(vc.get("textworld_default_action_max_tokens", 32))
        elif vc.get("use_inference_max_tokens_for_action", True):
            mt = inf.get("max_tokens")
            if mt is not None:
                action_max_tokens = int(mt)

    # Prefer per-domain action temperature; fallback to legacy vc.action_temperature; then inference.temperature.
    at = dom_cfg.get("action_temperature") if isinstance(dom_cfg, dict) else None
    if at is not None:
        action_temperature = float(at)
    else:
        legacy_at = vc.get("action_temperature")
        if legacy_at is not None:
            warnings.warn(
                "Deprecated config: use domain_prompts.<domain>.action_temperature "
                "instead of vc.action_temperature.",
                DeprecationWarning,
                stacklevel=2,
            )
            action_temperature = float(legacy_at)
        else:
            action_temperature = float(inf.get("temperature", 0.3))

    action_stop: list[str] | None = None
    if isinstance(dom_cfg, dict) and "action_stop" in dom_cfg:
        ts = dom_cfg.get("action_stop")
        if ts is None or ts is False:
            action_stop = None
        elif isinstance(ts, (list, tuple)):
            action_stop = [str(x) for x in ts] if ts else None
        else:
            action_stop = [str(ts)]
    elif domain == "textworld":
        # Legacy: vc.textworld_action_stop
        if "textworld_action_stop" in vc:
            warnings.warn(
                "Deprecated config: use domain_prompts.textworld.action_stop "
                "instead of vc.textworld_action_stop.",
                DeprecationWarning,
                stacklevel=2,
            )
            ts = vc["textworld_action_stop"]
            if ts is None or ts is False:
                action_stop = None
            elif isinstance(ts, (list, tuple)):
                action_stop = [str(x) for x in ts] if ts else None
            else:
                action_stop = [str(ts)]
        else:
            # Preserve old default behavior for TextWorld even if domain_prompts is missing.
            action_stop = ["\n"]

    return {
        "vc_mode": mode,
        "prompt_prefix": prompt_prefix,
        "action_max_tokens": action_max_tokens,
        "action_temperature": action_temperature,
        "action_stop": action_stop,
        "followup_max_tokens": int(vc.get("followup_max_tokens", 4)),
        "followup_temperature": float(vc.get("followup_temperature", 0.0)),
        # History / memory controls (consumed by base_agent.run_episode / run_adaptive_episode).
        # These live under domain_prompts.<domain> so they're experiment-controlled and visible in YAML.
        "history_keep_last_pairs": int(dom_cfg.get("history_keep_last_pairs", 4)) if isinstance(dom_cfg, dict) else 4,
        "history_max_obs_chars": int(dom_cfg.get("history_max_obs_chars", 1000)) if isinstance(dom_cfg, dict) else 1000,
        "history_current_obs_max_chars": int(dom_cfg.get("history_current_obs_max_chars", 1000)) if isinstance(dom_cfg, dict) else 1000,
        "history_obs_head_ratio": float(dom_cfg.get("history_obs_head_ratio", 0.15)) if isinstance(dom_cfg, dict) else 0.15,
        "pin_recipe": bool(dom_cfg.get("pin_recipe", False)) if isinstance(dom_cfg, dict) else False,
    }


# Backward-compatible alias: old name used throughout earlier iterations of the repo.
def vc_step_fn_kwargs(config: dict, domain: str) -> dict[str, Any]:
    warnings.warn(
        "vc_step_fn_kwargs is deprecated; use resolve_step_fn_kwargs instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_step_fn_kwargs(config, domain)
