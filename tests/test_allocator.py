"""Tests for allocator and frozen allocation policy."""

from __future__ import annotations

import warnings
from pathlib import Path

from src.agent.allocation_policy import FrozenPolicy, load_policy
from src.agent.allocator import allocate, eager_fixed_stage_from_signal

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "policy_artifact_v1.json"
LEGACY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "policy_artifact_legacy_pooled.json"


def test_frozen_policy_percentile_and_stage():
    p = FrozenPolicy(
        signal="tle_mean_entropy",
        domain="textworld",
        ecdf_by_stage={
            "C0": (0.2, 0.4, 0.6, 0.8),
            "C1": (0.2, 0.4, 0.6, 0.8),
            "C2": (0.2, 0.4, 0.6, 0.8),
        },
        theta1=0.25,
        theta2=0.75,
        direction="higher_is_uncertain",
    )
    assert p.percentile(0.4, source_stage="C0") == 0.375
    assert p.stage(0.1, source_stage="C0") == "C0"
    assert p.stage(0.5, source_stage="C0") == "C1"
    assert p.stage(0.9, source_stage="C0") == "C2"


def test_stage_wise_ecdf_c1_low_raw_maps_to_mid_percentile():
    """C1/C2 collapsed raw magnitudes stay rank-informative within stage."""
    p = FrozenPolicy(
        signal="tle_mean_entropy",
        domain="textworld",
        ecdf_by_stage={
            "C0": (0.02, 0.05, 0.08),
            "C1": (1e-6, 2e-6, 5e-6),
            "C2": (1e-7, 2e-7, 3e-7),
        },
        theta1=0.25,
        theta2=0.75,
        direction="higher_is_uncertain",
    )
    assert p.percentile(2e-6, source_stage="C1") == 0.5
    assert p.percentile(2e-6, source_stage="C0") < 0.2
    assert p.stage(2e-6, source_stage="C1") == "C1"
    assert p.stage(2e-6, source_stage="C0") == "C0"


def test_load_policy_rejects_legacy_pooled_ecdf_without_opt_in():
    import pytest

    from src.agent.allocation_policy import LEGACY_POOLED_ECDF_ERROR

    with pytest.raises(ValueError) as exc:
        load_policy(LEGACY_FIXTURE, domain="textworld", signal="tle_mean_entropy")
    assert "ecdf_ref" in str(exc.value)
    assert LEGACY_POOLED_ECDF_ERROR.split(".")[0] in str(exc.value)


def test_load_policy_legacy_pooled_ecdf_opt_in_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pol = load_policy(
            LEGACY_FIXTURE,
            domain="textworld",
            signal="tle_mean_entropy",
            allow_legacy_pooled_ecdf=True,
        )
        assert any("legacy pooled ecdf_ref" in str(x.message).lower() for x in w)
    assert pol.ecdf_by_stage["C0"] == pol.ecdf_by_stage["C1"]


def test_load_policy_roundtrip():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    assert pol.theta1 == 0.33
    assert len(pol.ecdf_by_stage["C0"]) == 9
    assert len(pol.ecdf_ref) == 27


def test_allocate_uses_policy_not_pilot_thresholds():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stage = allocate(
            {"mean_entropy": 0.95},
            "adaptive_tle",
            policy=pol,
            signal_source_stage="C0",
        )
        assert not w
    assert stage == "C2"


def test_allocate_uses_signal_source_stage_for_ecdf():
    pol = load_policy(FIXTURE, domain="textworld", signal="tle_mean_entropy")
    stage_c1 = allocate(
        {"mean_entropy": 9e-6},
        "adaptive_tle",
        policy=pol,
        signal_source_stage="C1",
    )
    stage_c0 = allocate(
        {"mean_entropy": 9e-6},
        "adaptive_tle",
        policy=pol,
        signal_source_stage="C0",
    )
    assert stage_c1 == "C2"
    assert stage_c0 == "C0"


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
