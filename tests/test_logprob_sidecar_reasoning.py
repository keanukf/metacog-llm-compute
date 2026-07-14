"""Sidecar schema: reasoning-token logprobs are preserved (data retention only)."""

from __future__ import annotations

import json
from pathlib import Path

from src.signals.token_entropy import slice_action_logprob_tokens
from src.utils.logging_utils import write_logprob_distribution_artifacts

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def _count_tokens_before_think_close(logprob_tokens: list[dict]) -> int:
    text = "".join(str(t.get("token", "")) for t in logprob_tokens)
    close = text.lower().find(THINK_CLOSE)
    if close < 0:
        return 0
    target = close + len(THINK_CLOSE)
    cursor = 0
    n_before = 0
    for rec in logprob_tokens:
        tok = str(rec.get("token", ""))
        end = cursor + len(tok)
        if end <= target:
            n_before += 1
        cursor = end
    return n_before


def test_c1_sidecar_logprob_tokens_include_reasoning_block(tmp_path: Path):
    """Sidecar stores the full completion; reasoning precedes committed action."""
    think_tokens = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "b", "logprob": -0.1}]},
        {"token": "reason", "logprob": -0.2, "top_logprobs": [{"token": "b", "logprob": -0.3}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
    ]
    action_tokens = [
        {"token": "go", "logprob": -0.5, "top_logprobs": [{"token": "x", "logprob": -0.6}]},
        {"token": " north\n", "logprob": -0.4, "top_logprobs": [{"token": "y", "logprob": -0.5}]},
    ]
    full = think_tokens + action_tokens
    text = "".join(t["token"] for t in full)

    written = write_logprob_distribution_artifacts(
        "ep_test_C1",
        [full],
        tmp_path,
        export_format="json",
        logprob_subdir="logprobs",
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    stored = payload["steps"][0]["logprob_tokens"]

    n_think = _count_tokens_before_think_close(stored)
    assert n_think >= 4
    assert stored[0]["token"] == THINK_OPEN
    assert len(stored) == len(full)
    assert THINK_CLOSE in text
    action_slice = slice_action_logprob_tokens(stored, text=text)
    assert len(action_slice) == 2
    assert action_slice[0]["token"] == "go"


def test_c2_sidecar_samples_include_reasoning_per_sample(tmp_path: Path):
    """Schema v2: each sample's logprob_tokens includes the thinking block."""
    sample_a = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "b", "logprob": -0.1}]},
        {"token": "x", "logprob": -0.2, "top_logprobs": [{"token": "b", "logprob": -0.3}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
        {"token": "go east\n", "logprob": -0.5, "top_logprobs": [{"token": "y", "logprob": -0.6}]},
    ]
    sample_b = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "b", "logprob": -0.1}]},
        {"token": "y", "logprob": -0.2, "top_logprobs": [{"token": "b", "logprob": -0.3}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
        {"token": "go west\n", "logprob": -0.5, "top_logprobs": [{"token": "z", "logprob": -0.6}]},
    ]

    written = write_logprob_distribution_artifacts(
        "ep_test_C2",
        [[sample_a, sample_b]],
        tmp_path,
        export_format="json",
        logprob_subdir="logprobs",
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    samples = payload["steps"][0]["samples"]
    assert len(samples) == 2
    for sample in samples:
        toks = sample["logprob_tokens"]
        assert _count_tokens_before_think_close(toks) >= 4
        assert toks[0]["token"] == THINK_OPEN
        assert len(toks) > 1
