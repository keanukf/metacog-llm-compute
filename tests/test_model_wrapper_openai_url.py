"""OpenAI client base URL normalization (LM Studio / OpenAI-compatible) and logprob parsing."""

from __future__ import annotations

from types import SimpleNamespace

from src.utils.model_wrapper import (
    _openai_completion_logprobs_to_list,
    normalize_openai_base_url,
    parse_lmstudio_responses_json,
)


def test_normalize_openai_base_url_appends_v1():
    assert normalize_openai_base_url("http://192.168.1.10:1234/") == "http://192.168.1.10:1234/v1"
    assert (
        normalize_openai_base_url("http://192.168.178.173:1234") == "http://192.168.178.173:1234/v1"
    )


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


def test_parse_lmstudio_responses_json_message_output_text():
    fixture = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "go north",
                        "logprobs": [
                            {
                                "token": "go",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "go", "logprob": -0.1},
                                    {"token": "take", "logprob": -2.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    text, lp = parse_lmstudio_responses_json(fixture)
    assert text == "go north"
    assert lp is not None
    assert len(lp) == 1
    assert lp[0]["token"] == "go"
    assert "top_logprobs" in lp[0]
    assert len(lp[0]["top_logprobs"]) == 2


def test_parse_lmstudio_responses_json_flat_output_text_block():
    fixture = {
        "output": [
            {
                "type": "output_text",
                "text": "hi",
                "logprobs": [
                    {
                        "token": "hi",
                        "logprob": -0.5,
                        "top_logprobs": [{"token": "hi", "logprob": -0.5}],
                    }
                ],
            }
        ]
    }
    text, lp = parse_lmstudio_responses_json(fixture)
    assert "hi" in text
    assert lp is not None and len(lp) == 1
