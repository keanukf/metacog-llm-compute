"""
Pilot Test 2 — Token-Entropie-Extraktion.
Unit tests for signals.token_entropy: synthetic logprobs, TLE differs for easy vs hard.
"""

from __future__ import annotations

from src.signals.token_entropy import (
    compute_tle,
    entropy_shannon_from_top_logprobs,
    extract_action_tle_from_response,
    extract_tle_from_response,
)


def test_compute_tle_returns_mean_and_max():
    logprobs = [{"logprob": -0.5}] * 10
    out = compute_tle(logprobs)
    assert "mean_entropy" in out
    assert "max_entropy" in out
    assert out["mean_entropy"] >= 0
    assert out["max_entropy"] >= 0


def test_compute_tle_higher_entropy_for_uncertain_tokens():
    # More uncertain (lower logprob / more uniform) -> higher entropy
    easy = [{"logprob": -0.1}] * 10  # high prob
    hard = [{"logprob": -2.0}] * 10  # lower prob
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


def test_entropy_shannon_from_top_logprobs_two_candidates_nonzero():
    top = [
        {"token": "a", "logprob": -0.1},
        {"token": "b", "logprob": -3.0},
    ]
    h = entropy_shannon_from_top_logprobs(top)
    assert h > 0.01


def test_entropy_shannon_single_candidate_zero():
    assert entropy_shannon_from_top_logprobs([{"token": "x", "logprob": -0.1}]) == 0.0


def test_compute_tle_with_top_logprobs_per_token():
    logprobs = [
        {
            "token": "a",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "a", "logprob": -0.1},
                {"token": "b", "logprob": -2.0},
            ],
        },
        {
            "token": "b",
            "logprob": -0.2,
            "top_logprobs": [
                {"token": "b", "logprob": -0.2},
                {"token": "c", "logprob": -1.5},
            ],
        },
    ]
    out = compute_tle(logprobs)
    assert out["mean_entropy"] > 0.01
    assert out["max_entropy"] >= out["mean_entropy"]


def test_extract_action_tle_slices_first_line_tokens_only():
    # First line: "go north\n"; second line: "reasoning"
    lp = [
        {"token": "go", "logprob": -0.1},
        {"token": " ", "logprob": -0.1},
        {"token": "north", "logprob": -0.1},
        {"token": "\n", "logprob": -0.1},
        {"token": "reasoning", "logprob": -3.0},
    ]
    full = extract_tle_from_response("go north\nreasoning", lp)
    sliced = extract_action_tle_from_response("go north\nreasoning", lp)
    assert full is not None and sliced is not None
    # Sliced should not include the low-prob reasoning token, so mean entropy differs.
    assert sliced["mean_entropy"] != full["mean_entropy"]


def test_extract_action_tle_returns_none_when_multiline_without_tokens():
    # Without token strings we refuse to mix reasoning tokens into action TLE.
    lp = [{"logprob": -0.2}] * 5
    assert extract_action_tle_from_response("go north\nreasoning", lp) is None


def test_extract_action_tle_falls_back_for_single_line_without_tokens():
    lp = [{"logprob": -0.2}] * 5
    a = extract_action_tle_from_response("go north", lp)
    b = extract_tle_from_response("go north", lp)
    assert a == b


def test_extract_action_tle_slices_first_line_after_think_close():
    lp = [
        {"token": "<think>", "logprob": -1.0},
        {"token": "\n", "logprob": -1.0},
        {"token": "x", "logprob": -1.0},
        {"token": "\n", "logprob": -1.0},
        {"token": "</think>", "logprob": -1.0},
        {"token": "\n", "logprob": -1.0},
        {"token": "A", "logprob": -0.2},
        {"token": "-", "logprob": -0.2},
        {"token": ">", "logprob": -0.2},
        {"token": "C", "logprob": -0.2},
        {"token": "\n", "logprob": -0.2},
        {"token": "extra", "logprob": -6.0},
    ]
    sliced = extract_action_tle_from_response("<think>\nx\n</think>\nA->C\nextra", lp)
    assert sliced == compute_tle(lp[6:11])
