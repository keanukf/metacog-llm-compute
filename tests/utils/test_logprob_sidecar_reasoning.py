"""Logprob-sidecar reasoning scope: ``full`` vs ``action_window``.

Verifies a C1 ``full`` sidecar retains the reasoning block while ``action_window`` excludes it, and
that C2 writes one action-window sidecar per sample. The scope decides whether entropy is measured
over the chain-of-thought or only the committed action tokens -- a substantive RQ1 definitional
choice, since including reasoning tokens would measure a different quantity than the action-line
TLE the study reports.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.signals.token_entropy import slice_action_logprob_tokens
from src.utils.logging_utils import write_logprob_distribution_artifacts
from src.utils.logprob_sidecar import filter_logprob_raw_for_sidecar

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


def test_c1_full_sidecar_retains_reasoning_block(tmp_path: Path):
    """Production exploratory subset: full scope keeps reasoning tokens."""
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

    written = write_logprob_distribution_artifacts(
        "ep_test_C1",
        [full],
        tmp_path,
        export_format="json",
        logprob_subdir="logprobs",
        sidecar_scope="full",
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    stored = payload["steps"][0]["logprob_tokens"]
    assert payload["sidecar_scope"] == "full"
    assert _count_tokens_before_think_close(stored) >= 4
    assert len(stored) == len(full)


def test_c1_action_window_sidecar_excludes_reasoning(tmp_path: Path):
    think_tokens = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
    ]
    action_tokens = [
        {"token": "go", "logprob": -0.5, "top_logprobs": [{"token": "x", "logprob": -0.6}]},
        {"token": " north\n", "logprob": -0.4, "top_logprobs": [{"token": "y", "logprob": -0.5}]},
    ]
    full = think_tokens + action_tokens
    filtered = filter_logprob_raw_for_sidecar([full], "action_window")

    written = write_logprob_distribution_artifacts(
        "ep_test_C1",
        filtered,
        tmp_path,
        export_format="json",
        logprob_subdir="logprobs",
        sidecar_scope="action_window",
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    stored = payload["steps"][0]["logprob_tokens"]
    assert payload["sidecar_scope"] == "action_window"
    assert _count_tokens_before_think_close(stored) == 0
    assert len(stored) == 2
    action_slice = slice_action_logprob_tokens(full, text="".join(t["token"] for t in full))
    assert [t["token"] for t in stored] == [t["token"] for t in action_slice]


def test_c2_action_window_sidecar_per_sample(tmp_path: Path):
    sample_a = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
        {"token": "go east\n", "logprob": -0.5, "top_logprobs": [{"token": "y", "logprob": -0.6}]},
    ]
    sample_b = [
        {"token": THINK_OPEN, "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.2}]},
        {"token": THINK_CLOSE, "logprob": -0.1, "top_logprobs": [{"token": "c", "logprob": -0.2}]},
        {"token": "\n", "logprob": -0.05, "top_logprobs": [{"token": "d", "logprob": -0.1}]},
        {"token": "go west\n", "logprob": -0.5, "top_logprobs": [{"token": "z", "logprob": -0.6}]},
    ]
    filtered = filter_logprob_raw_for_sidecar([[sample_a, sample_b]], "action_window")

    written = write_logprob_distribution_artifacts(
        "ep_test_C2",
        filtered,
        tmp_path,
        export_format="json",
        logprob_subdir="logprobs",
        sidecar_scope="action_window",
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    for sample in payload["steps"][0]["samples"]:
        toks = sample["logprob_tokens"]
        assert _count_tokens_before_think_close(toks) == 0
        assert len(toks) == 1
