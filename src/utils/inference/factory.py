"""Factory for inference wrappers (vLLM, LM Studio)."""

from __future__ import annotations

from typing import Any

from src.utils.inference.base import ModelWrapper
from src.utils.inference.lmstudio.wrapper import LMStudioWrapper
from src.utils.inference.logprob_config import resolve_top_logprobs
from src.utils.inference.vllm import VLLMWrapper

_TOP_LOGPROB_KWARG_KEYS = frozenset({"top_logprobs", "lmstudio_top_logprobs"})


def create_wrapper(
    backend: str = "vllm",
    model_name: str | None = None,
    dtype: str = "float16",
    device: str | None = None,  # noqa: ARG001 — kept for call-site compatibility
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelWrapper:
    """
    Create wrapper by backend name.

    - ``vllm`` + ``model_name`` -> :class:`VLLMWrapper` (requires CUDA).
    - ``lmstudio`` + ``model_name`` -> :class:`LMStudioWrapper` (responses-only HTTP).
    - Otherwise returns a stub :class:`ModelWrapper` (raises on generate; used in mocks).
    """
    top_k = resolve_top_logprobs(None, **kwargs)
    rest = {k: v for k, v in kwargs.items() if k not in _TOP_LOGPROB_KWARG_KEYS}

    if model_name and backend == "vllm":
        return VLLMWrapper(model_name=model_name, dtype=dtype, top_logprobs=top_k, **rest)
    if model_name and backend == "lmstudio":
        url = base_url or rest.pop("lmstudio_base_url", None) or kwargs.get("lmstudio_base_url")
        if not url:
            url = "http://localhost:1234/v1"
        api_key = rest.pop("lmstudio_api_key", None) or rest.pop("api_key", None)
        if api_key is None:
            api_key = kwargs.get("lmstudio_api_key") or kwargs.get("api_key")
        api_host = rest.pop("api_host", None)
        return LMStudioWrapper(
            model_name=model_name,
            base_url=url,
            api_key=api_key,
            top_logprobs=top_k,
            api_host=api_host,
            **rest,
        )
    if model_name and backend == "hf":
        raise ValueError(
            'backend "hf" was removed; use --pilot-mode lmstudio (local) or cuda (vLLM on GPU).'
        )
    return ModelWrapper()
