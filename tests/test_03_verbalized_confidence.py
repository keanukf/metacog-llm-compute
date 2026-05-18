"""
Pilot Test 3 — Verbalisierte Konfidenz.
Unit tests for signals.verbalized_confidence: parse numeric 0-100 from strings.
"""

from __future__ import annotations

from src.signals.verbalized_confidence import (
    extract_vc_from_followup,
    parse_confidence,
    parse_confidence_with_meta,
)


def test_parse_confidence_explicit_label():
    s = "The answer is X. Confidence: 85"
    assert parse_confidence(s) == 85.0


def test_parse_confidence_0_100_format():
    s = "0-100: 70"
    assert parse_confidence(s) == 70.0


def test_parse_confidence_in_range():
    assert parse_confidence("Confidence: 0") == 0.0
    assert parse_confidence("Confidence: 100") == 100.0
    assert parse_confidence("Confidence: 50") == 50.0


def test_parse_confidence_unparseable_returns_none():
    assert parse_confidence("No number here at all.") is None
    assert parse_confidence("") is None


def test_parse_confidence_ignores_out_of_range():
    # Parser may still find 999 in text; our patterns restrict 0-100
    s = "Confidence: 85. So 999 is not valid."
    assert parse_confidence(s) == 85.0


def test_parse_confidence_with_meta_returns_pattern():
    val, pat = parse_confidence_with_meta("Confidence: 42")
    assert val == 42.0
    assert pat == "confidence_label"


def test_extract_vc_from_followup_builds_record():
    lp = [{"token": "8", "logprob": -0.1, "top_logprobs": []}]
    d = extract_vc_from_followup("prompt here", "82", lp)
    assert d["vc_value"] == 82.0
    assert d["vc_raw_text"] == "82"
    assert d["vc_prompt"] == "prompt here"
    assert d["vc_tokens_used"] == 1
    assert d["vc_logprobs"] is lp
    assert d["vc_pattern_matched"] == "bare_number"


def test_bare_number_pattern():
    val, pat = parse_confidence_with_meta("75")
    assert val == 75.0
    assert pat == "bare_number"
