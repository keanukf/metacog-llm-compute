"""Regression test for the obs_ceiling divergence between the two Gate D abort
diagnostic scripts (inspect_gate_d_abort_actions.py vs. analyze_gate_d_abort_distance.py)."""

from __future__ import annotations

from scripts.inspect_gate_d_abort_actions import _resolve_obs_ceiling


def test_resolve_obs_ceiling_defaults_to_sweep_value():
    """Without an explicit --obs-ceiling, must match the sweep being replayed, not a
    stale hardcoded 25 (the bug: the sibling script, analyze_gate_d_abort_distance.py,
    already reads obs_ceiling from the sweep; this one didn't)."""
    sweep = {"obs_ceiling": 70, "seed": 42}
    assert _resolve_obs_ceiling(None, sweep) == 70


def test_resolve_obs_ceiling_explicit_override_wins():
    sweep = {"obs_ceiling": 70}
    assert _resolve_obs_ceiling(45, sweep) == 45


def test_resolve_obs_ceiling_falls_back_to_25_when_sweep_lacks_field():
    assert _resolve_obs_ceiling(None, {}) == 25
