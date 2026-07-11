"""Shared vLLM server/client logprob helpers (minimal — no chat-template logic)."""

from __future__ import annotations

from typing import Any

from src.utils.inference.logprobs import normalize_logprobs


def normalize_chat_completion_logprobs(raw_logprobs: Any) -> list[dict[str, Any]] | None:
    """
    Map OpenAI chat ``choices[].logprobs`` to canonical per-token records for TLE.

    Expects ``content`` list entries with ``token``, ``logprob``, ``top_logprobs``.
    """
    if raw_logprobs is None:
        return None
    content = None
    if isinstance(raw_logprobs, dict):
        content = raw_logprobs.get("content")
    else:
        content = getattr(raw_logprobs, "content", None)
    if content is None:
        return None
    return normalize_logprobs(content)
