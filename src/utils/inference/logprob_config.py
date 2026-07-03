"""Shared top-k logprob width for all inference backends."""

from __future__ import annotations

from typing import Any

DEFAULT_TOP_LOGPROBS = 20


def resolve_top_logprobs(
    inference_cfg: dict[str, Any] | None = None,
    *,
    default: int = DEFAULT_TOP_LOGPROBS,
    **kwargs: Any,
) -> int:
    """
    Resolve top-k logprob width for TLE (Shannon entropy over renormalized top-k).

    Precedence: explicit ``top_logprobs`` kwarg > ``inference_cfg['top_logprobs']``
    > deprecated ``lmstudio_top_logprobs`` (kwarg or cfg) > ``default``.
    """
    for key in ("top_logprobs", "lmstudio_top_logprobs"):
        if key in kwargs and kwargs[key] is not None:
            return max(1, int(kwargs[key]))
    cfg = inference_cfg or {}
    if cfg.get("top_logprobs") is not None:
        return max(1, int(cfg["top_logprobs"]))
    if cfg.get("lmstudio_top_logprobs") is not None:
        return max(1, int(cfg["lmstudio_top_logprobs"]))
    return max(1, int(default))
