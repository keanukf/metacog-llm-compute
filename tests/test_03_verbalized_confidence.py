"""
Pilot Test 3 — Verbalisierte Konfidenz.
Unit tests for signals.verbalized_confidence: parse numeric 0-100 from strings.
"""
from __future__ import annotations

import pytest

from src.signals.verbalized_confidence import parse_confidence


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
