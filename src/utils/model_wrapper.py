"""
Backward-compatible re-exports for inference wrappers.

Implementation lives in :mod:`src.utils.inference`.
"""

from __future__ import annotations

from src.utils.inference import (
    LMStudioWrapper,
    ModelWrapper,
    VLLMWrapper,
    create_wrapper,
    normalize_logprobs,
    normalize_openai_base_url,
    openai_completion_logprobs_to_list,
    parse_lmstudio_responses_json,
)

# Legacy private names used by tests
_normalize_logprobs = normalize_logprobs
_openai_completion_logprobs_to_list = openai_completion_logprobs_to_list

__all__ = [
    "LMStudioWrapper",
    "ModelWrapper",
    "VLLMWrapper",
    "create_wrapper",
    "normalize_openai_base_url",
    "normalize_logprobs",
    "openai_completion_logprobs_to_list",
    "parse_lmstudio_responses_json",
    "_normalize_logprobs",
    "_openai_completion_logprobs_to_list",
]
