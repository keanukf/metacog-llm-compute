"""Model-wrapper factory after the inference module split.

Verifies the factory builds the stub/vLLM wrappers, threads ``top_logprobs`` through to vLLM with
the right default, and keeps ``parse`` re-exported from the old ``model_wrapper`` path. The
top_logprobs count directly bounds how many candidates TLE can be computed over, so its passthrough
is an RQ1-signal-fidelity concern, not just plumbing; the re-export guards call sites the module
split would otherwise have broken.
"""

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
