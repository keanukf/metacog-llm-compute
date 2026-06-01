"""Inference factory after module split."""

from __future__ import annotations

from src.utils.inference import ModelWrapper, create_wrapper
from src.utils.model_wrapper import parse_lmstudio_responses_json


def test_stub_wrapper_factory():
    w = create_wrapper(backend="vllm", model_name=None)
    assert isinstance(w, ModelWrapper)


def test_parse_reexport_from_model_wrapper():
    text, lp = parse_lmstudio_responses_json(
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]}
    )
    assert text == "x"
    assert lp is None
