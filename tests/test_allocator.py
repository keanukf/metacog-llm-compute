"""Tests for allocator and frozen allocation policy."""

from __future__ import annotations

import warnings
from pathlib import Path

from src.agent.allocation_policy import FrozenPolicy, load_policy
from src.agent.allocator import allocate, eager_fixed_stage_from_signal

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "policy_artifact_v1.json"


def test_frozen_policy_percentile_and_stage():
    p = FrozenPolicy(
        signal="tle_mean_entropy",
        domain="textworld",
        ecdf_ref=(0.2, 0.4, 0.6, 0.8),
        theta1=0.25,
        theta2=0.75,
        direction="higher_is_uncertain",
    )
    assert p.percentile(0.4) == 0.375
    assert p.stage(0.1) == "C0"
    assert p.stage(0.5) == "C1"
    assert p.stage(0.9) == "C2"


def test_load_policy_roundtrip():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    assert pol.theta1 == 0.33
    assert len(pol.ecdf_ref) == 9


def test_allocate_uses_policy_not_pilot_thresholds():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stage = allocate({"mean_entropy": 0.95}, "adaptive_tle", policy=pol)
        assert not w
    assert stage == "C2"


def test_allocate_pilot_fallback_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stage = allocate({"mean_entropy": 0.95}, "adaptive_tle")
        assert any("hardcoded pilot" in str(x.message) for x in w)
    assert stage == "C2"


def test_eager_fixed_stage_from_policy():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    fixed = eager_fixed_stage_from_signal({"mean_entropy": 0.15}, policy=pol)
    assert fixed in {"C0", "C1", "C2"}


def test_eager_style_step0_is_c0():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    assert allocate(None, "eager_style", step_index=0, policy=pol) == "C0"


def test_eager_style_uses_episode_fixed_stage():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    assert (
        allocate(
            {"mean_entropy": 0.5}, "eager_style", step_index=2, policy=pol, episode_fixed_stage="C1"
        )
        == "C1"
    )
