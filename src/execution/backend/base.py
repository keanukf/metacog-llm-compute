"""Synchronous inference backend protocol for episode workers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """
    Minimal sync interface matching ``ModelWrapper`` for agent stages.

    ``inprocess`` backend is a future option — Phase runners with ``--real`` use
    ``ServerBackend`` against ``vllm serve``.
    """

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]: ...

    def generate_many(
        self,
        prompt: str,
        *,
        n: int,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> list[tuple[str, list[dict[str, Any]] | None]]: ...
