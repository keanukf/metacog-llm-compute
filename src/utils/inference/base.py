"""Shared model wrapper contract."""

from __future__ import annotations

from typing import Any


class ModelWrapper:
    """
    Minimal interface for LM inference.

    - ``generate(prompt, logprobs=False)`` returns ``(text, logprobs_or_none)``.
    - vLLM supports logprobs natively; LM Studio uses ``POST /v1/responses``.
    """

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        raise NotImplementedError("Use VLLMWrapper or LMStudioWrapper in production")

    def generate_many(
        self,
        prompt: str,
        *,
        n: int,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> list[tuple[str, list[dict[str, Any]] | None]]:
        nn = max(1, int(n))
        out: list[tuple[str, list[dict[str, Any]] | None]] = []
        for _ in range(nn):
            out.append(
                self.generate(
                    prompt,
                    logprobs=logprobs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            )
        return out
