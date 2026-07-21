"""LM Studio wrapper: responses-only path, no completions fallback."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.utils.inference.lmstudio.parse import build_lmstudio_call_diagnostics
from src.utils.inference.lmstudio.wrapper import LMStudioWrapper


def test_build_lmstudio_call_diagnostics_reasoning_only():
    diag = build_lmstudio_call_diagnostics(
        endpoint="http://localhost:1234/v1/responses",
        data={
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                }
            ]
        },
        reasoning_text="thinking",
        message_text="",
        token_records=[],
        assembled_text="",
        logprobs_requested=True,
        enable_thinking=False,
    )
    assert diag["status"] == "reasoning_only"
    assert diag["route"] == "POST /v1/responses"
    assert diag["has_reasoning_text"] is True
    assert diag["has_message_text"] is False


@patch("src.utils.inference.lmstudio.wrapper.post_v1_responses")
@patch(
    "src.utils.inference.lmstudio.wrapper.resolve_lmstudio_api_host", return_value="127.0.0.1:1234"
)
def test_lmstudio_wrapper_uses_responses_only_no_completions(mock_host, mock_post):
    fixture = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "go east",
                        "logprobs": [{"token": "go", "logprob": -0.1}],
                    }
                ],
            }
        ]
    }
    mock_post.return_value = (fixture, 200, None)
    wrapper = LMStudioWrapper(model_name="test-model", base_url="http://127.0.0.1:1234/v1")
    text, lp = wrapper.generate("prompt", logprobs=True, max_tokens=8, temperature=0.0)
    assert text == "go east"
    assert lp is not None and len(lp) == 1
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs.get("include_logprobs") is True

    diags = wrapper.consume_call_diagnostics()
    assert len(diags) == 1
    assert diags[0]["status"] == "ok"
    assert diags[0]["has_token_logprobs"] is True

    mock_post.reset_mock()
    mock_post.return_value = (fixture, 200, None)
    wrapper.generate("again", logprobs=False)
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs.get("include_logprobs") is False


@patch("src.utils.inference.lmstudio.wrapper.post_v1_responses")
@patch(
    "src.utils.inference.lmstudio.wrapper.resolve_lmstudio_api_host", return_value="127.0.0.1:1234"
)
def test_lmstudio_wrapper_raises_on_http_failure(mock_host, mock_post):
    mock_post.return_value = (None, 500, "server error")
    wrapper = LMStudioWrapper(model_name="test-model", base_url="http://127.0.0.1:1234/v1")
    with pytest.raises(Exception, match="responses failed"):
        wrapper.generate("prompt", logprobs=True)
    diags = wrapper.consume_call_diagnostics()
    assert diags[0]["status"] == "http_error"
    assert diags[0]["http_status"] == 500


def test_create_wrapper_rejects_hf_backend():
    from src.utils.inference.factory import create_wrapper

    with pytest.raises(ValueError, match="hf"):
        create_wrapper(backend="hf", model_name="any")
