"""History truncation guard for preregistered full-history default (H3)."""

from __future__ import annotations

import sys
from typing import Any


def history_truncation_active(step_cfg: dict[str, Any]) -> bool:
    hk = step_cfg.get("history_keep_last_pairs")
    if hk is not None and int(hk) > 0:
        return True
    for key in ("history_max_obs_chars", "history_current_obs_max_chars"):
        v = step_cfg.get(key)
        if v is not None and int(v) > 0:
            return True
    fmc = step_cfg.get("followup_max_context_chars")
    if fmc is not None and int(fmc) > 0:
        return True
    return False


def enforce_full_history_or_exit(
    step_cfg: dict[str, Any],
    *,
    allow_history_truncation: bool,
    script_name: str,
) -> bool:
    """
    Return True if truncation is allowed (explicit flag). Exit process if active without flag.
    """
    if not history_truncation_active(step_cfg):
        return False
    if allow_history_truncation:
        return True
    print(
        f"{script_name}: history truncation parameters are active in step config; "
        "re-run with --allow-history-truncation to override (not valid for confirmatory H3).",
        file=sys.stderr,
    )
    raise SystemExit(2)
