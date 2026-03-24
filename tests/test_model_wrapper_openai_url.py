"""OpenAI client base URL normalization (litellm / lmstudio) and logprob parsing."""
from __future__ import annotations

from types import SimpleNamespace

from src.utils.model_wrapper import _openai_completion_logprobs_to_list, normalize_openai_base_url


def test_normalize_openai_base_url_appends_v1():
    assert normalize_openai_base_url("http://litellm.home/") == "http://litellm.home/v1"
    assert normalize_openai_base_url("http://192.168.178.173:1234") == "http://192.168.178.173:1234/v1"


def test_normalize_openai_base_url_preserves_existing_v1():
    assert (
        normalize_openai_base_url("http://192.168.178.173:1234/v1")
        == "http://192.168.178.173:1234/v1"
    )
    assert normalize_openai_base_url("http://host/v1/") == "http://host/v1"


def test_normalize_openai_base_url_no_duplicate_v1():
    assert normalize_openai_base_url("http://host/v1") != "http://host/v1/v1"


def test_openai_completion_logprobs_from_token_logprobs():
    raw = SimpleNamespace(token_logprobs=[-0.1, None, -0.2])
    out = _openai_completion_logprobs_to_list(raw)
    assert out is not None
    assert len(out) == 2
    assert out[0]["logprob"] == -0.1


def test_openai_completion_logprobs_dict_token_logprobs():
    out = _openai_completion_logprobs_to_list({"token_logprobs": [-1.0, -2.0]})
    assert len(out) == 2


def test_openai_completion_logprobs_prefers_content_when_present():
    raw = SimpleNamespace(content=[{"logprob": -0.5}])
    out = _openai_completion_logprobs_to_list(raw)
    assert len(out) == 1
