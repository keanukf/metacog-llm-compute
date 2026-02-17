"""
vLLM wrapper with fallback to HuggingFace Transformers.
Abstract interface: generate(prompt, logprobs=False) -> text, optional logprobs.
Stub: no real backend; for real use implement VLLMWrapper / HFWrapper.
"""
from __future__ import annotations

from typing import Any


class ModelWrapper:
    """
    Minimal interface for LM inference.
    - generate(prompt, logprobs=False) returns (text, logprobs_or_none).
    - vLLM supports logprobs natively; HF Transformers use output_scores=True.
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
        """
        Generate a completion for the given prompt.

        Returns:
            (generated_text, logprobs_or_none). logprobs is a list of token-level
            logprob dicts when logprobs=True, else None.
        """
        raise NotImplementedError("Use VLLMWrapper or HFWrapper in production")


def create_wrapper(backend: str = "vllm", **kwargs: Any) -> ModelWrapper:
    """Factory: create wrapper by backend name. Stub returns a no-op placeholder."""
    # Stub: return a simple mock-friendly base
    return ModelWrapper()
