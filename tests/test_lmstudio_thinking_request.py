"""LM Studio /v1/responses request body: thinking controls (Open Responses wire format)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.utils.inference.lmstudio.request_body import (
    build_v1_responses_body,
    reasoning_effort_for_enable_thinking,
    thinking_control_fields,
)
from src.utils.inference.lmstudio.responses import post_v1_responses


def test_thinking_control_fields_off_uses_openresponses_none():
    fields = thinking_control_fields(False)
    assert fields["enable_thinking"] is False
    assert fields["chat_template_kwargs"] == {"enable_thinking": False}
    assert fields["reasoning"] == {"effort": "none"}


def test_thinking_control_fields_on_uses_low_by_default():
    fields = thinking_control_fields(True)
    assert fields["enable_thinking"] is True
    assert fields["reasoning"] == {"effort": "low"}


def test_thinking_control_fields_on_respects_override():
    fields = thinking_control_fields(True, thinking_effort_when_on="medium")
    assert fields["reasoning"] == {"effort": "medium"}


def test_thinking_control_fields_none_is_empty():
    assert thinking_control_fields(None) == {}


def test_reasoning_effort_mapping():
    assert reasoning_effort_for_enable_thinking(False) == "none"
    assert reasoning_effort_for_enable_thinking(True) == "low"
    assert reasoning_effort_for_enable_thinking(True, thinking_effort_when_on="high") == "high"


def test_build_v1_responses_body_merges_logprobs_and_thinking():
    body = build_v1_responses_body(
        model="qwen/qwen3-4b",
        prompt="go north",
        max_tokens=64,
        temperature=0.0,
        top_logprobs=5,
        enable_thinking=False,
    )
    assert body["reasoning"] == {"effort": "none"}


@patch("src.utils.inference.lmstudio.responses.httpx.Client")
@patch(
    "src.utils.inference.lmstudio.responses.resolve_lmstudio_api_host",
    return_value="127.0.0.1:1234",
)
def test_post_v1_responses_sends_reasoning_effort_none(mock_host, mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "go north", "logprobs": []}],
            }
        ],
        "usage": {"output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0}},
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    data, status, err = post_v1_responses(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="qwen/qwen3-4b",
        prompt="test",
        max_tokens=32,
        temperature=0.0,
        top_logprobs=5,
        enable_thinking=False,
    )
    assert err is None
    assert status == 200
    assert data is not None

    sent_body = mock_client.post.call_args.kwargs["json"]
    assert sent_body["reasoning"] == {"effort": "none"}


@patch("src.utils.inference.lmstudio.responses.httpx.Client")
@patch(
    "src.utils.inference.lmstudio.responses.resolve_lmstudio_api_host",
    return_value="127.0.0.1:1234",
)
def test_post_v1_responses_thinking_on_uses_low_effort(mock_host, mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"output": [], "usage": {}}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    post_v1_responses(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="qwen/qwen3-4b",
        prompt="think then answer",
        max_tokens=128,
        temperature=0.5,
        top_logprobs=5,
        enable_thinking=True,
    )
    sent_body = mock_client.post.call_args.kwargs["json"]
    assert sent_body["reasoning"] == {"effort": "low"}
    assert sent_body["enable_thinking"] is True
