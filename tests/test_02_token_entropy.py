"""
Pilot Test 2 — Token-Entropie-Extraktion.
Unit tests for signals.token_entropy: synthetic logprobs, TLE differs for easy vs hard.
"""
from __future__ import annotations

import pytest

from src.signals.token_entropy import compute_tle, extract_tle_from_response


def test_compute_tle_returns_mean_and_max():
    logprobs = [{"logprob": -0.5}] * 10
    out = compute_tle(logprobs)
    assert "mean_entropy" in out
    assert "max_entropy" in out
    assert out["mean_entropy"] >= 0
    assert out["max_entropy"] >= 0


def test_compute_tle_higher_entropy_for_uncertain_tokens():
    # More uncertain (lower logprob / more uniform) -> higher entropy
    easy = [{"logprob": -0.1}] * 10   # high prob
    hard = [{"logprob": -2.0}] * 10   # lower prob
    easy_tle = compute_tle(easy)
    hard_tle = compute_tle(hard)
    assert hard_tle["mean_entropy"] > easy_tle["mean_entropy"]


def test_compute_tle_float_logprobs():
    float_lp = [-0.3] * 5
    out = compute_tle(float_lp)
    assert "mean_entropy" in out
    assert isinstance(out["mean_entropy"], (int, float))


def test_extract_tle_from_response_with_logprobs():
    text = "answer"
    logprobs = [{"logprob": -0.5}] * 3
    out = extract_tle_from_response(text, logprobs)
    assert out is not None
    assert out["mean_entropy"] >= 0


def test_extract_tle_from_response_without_logprobs():
    out = extract_tle_from_response("answer", None)
    assert out is None
