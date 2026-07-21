"""Inference factory after module split."""

from __future__ import annotations

from unittest.mock import patch

from src.utils.inference import ModelWrapper, create_wrapper
from src.utils.inference.logprob_config import DEFAULT_TOP_LOGPROBS
from src.utils.model_wrapper import parse_lmstudio_responses_json


def test_stub_wrapper_factory():
    w = create_wrapper(backend="vllm", model_name=None)
    assert isinstance(w, ModelWrapper)


def test_create_wrapper_passes_top_logprobs_to_vllm():
    with patch("src.utils.inference.factory.VLLMWrapper") as mock_cls:
        mock_cls.return_value = ModelWrapper()
        create_wrapper(backend="vllm", model_name="m", top_logprobs=12)
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["top_logprobs"] == 12


def test_create_wrapper_default_top_logprobs():
    with patch("src.utils.inference.factory.VLLMWrapper") as mock_cls:
        mock_cls.return_value = ModelWrapper()
        create_wrapper(backend="vllm", model_name="m")
        assert mock_cls.call_args.kwargs["top_logprobs"] == DEFAULT_TOP_LOGPROBS


def test_parse_reexport_from_model_wrapper():
    text, lp = parse_lmstudio_responses_json(
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]}
    )
    assert text == "x"
    assert lp is None
