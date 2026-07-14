"""
Rule-based allocator + baselines: Adaptive-TLE, Adaptive-VC, Always-C0, Always-C2, Random, EAGer-style.

First-step rule (preregistered, thesis §5.4): step 0 receives ``signal=None`` and defaults to C0.
From step 1 onward, allocation uses the signal extracted from the *previous* step's TLE / VC.
"""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.allocation_policy import FrozenPolicy

# Strategy names as in experiment_core.yaml
STRATEGIES = ("adaptive_tle", "adaptive_vc", "always_c0", "always_c2", "random", "eager_style")

_POLICY_STRATEGIES = frozenset({"adaptive_tle", "adaptive_vc"})


def _raw_signal_value(signal: dict[str, Any] | None, strategy: str) -> float | None:
    if not signal:
        return None
    if strategy in ("adaptive_tle", "eager_style"):
        v = signal.get("mean_entropy")
        if v is not None:
            return float(v)
    if strategy == "adaptive_vc":
        v = signal.get("vc")
        if v is not None:
            return float(v)
    return None


def _pilot_threshold_stage(raw: float, strategy: str) -> str:
    warnings.warn(
        "using hardcoded pilot thresholds; not valid for Phase 2",
        UserWarning,
        stacklevel=3,
    )
    if strategy in ("adaptive_tle", "eager_style"):
        if raw > 0.8:
            return "C2"
        if raw > 0.4:
            return "C1"
        return "C0"
    if strategy == "adaptive_vc":
        if raw < 50:
            return "C2"
        if raw < 75:
            return "C1"
        return "C0"
    return "C0"


def allocate(
    signal: dict[str, Any] | None,
    strategy: str,
    step_index: int = 0,
    rng: random.Random | None = None,
    *,
    policy: FrozenPolicy | None = None,
    episode_fixed_stage: str | None = None,
    signal_source_stage: str | None = None,
) -> str:
    """
    Choose compute stage (C0, C1, C2) for this step given strategy and optional signal.

    Args:
        signal: Dict with ``mean_entropy`` (TLE) and/or ``vc`` (0–100), or None on step 0.
        strategy: One of STRATEGIES.
        step_index: Current step index in the episode.
        rng: Random generator for ``random`` strategy.
        policy: Frozen Phase-1 policy; when set, adaptive strategies use ``policy.stage()`` only.
        episode_fixed_stage: For ``eager_style`` after step 0 — fixed stage from step-0 signal.
        signal_source_stage: Compute stage that produced ``signal`` (previous step); required for
            stage-wise ECDF lookup when ``policy`` is set and ``signal`` is not None.

    Returns:
        ``"C0"`` | ``"C1"`` | ``"C2"``.
    """
    rng = rng or random.Random()
    source_stage = (signal_source_stage or "C0").upper()
    if strategy == "always_c0":
        return "C0"
    if strategy == "always_c2":
        return "C2"
    if strategy == "random":
        return rng.choice(["C0", "C1", "C2"])
    if strategy == "eager_style":
        if step_index == 0:
            return "C0"
        if episode_fixed_stage is not None:
            return episode_fixed_stage
        raw = _raw_signal_value(signal, strategy)
        if raw is not None and policy is not None:
            return policy.stage(raw, source_stage=source_stage)
        if raw is not None:
            return _pilot_threshold_stage(raw, strategy)
        return "C0"
    if strategy in _POLICY_STRATEGIES:
        raw = _raw_signal_value(signal, strategy)
        if raw is None:
            return "C0"
        if policy is not None:
            return policy.stage(raw, source_stage=source_stage)
        return _pilot_threshold_stage(raw, strategy)
    return "C0"


def baseline_always_c0() -> str:
    return "C0"


def baseline_always_c2() -> str:
    return "C2"


def eager_fixed_stage_from_signal(
    signal: dict[str, Any] | None,
    *,
    policy: FrozenPolicy | None = None,
) -> str | None:
    """Map step-0 signal to episodenfixed stage for ``eager_style`` (Table 5.1)."""
    raw = _raw_signal_value(signal, "eager_style")
    if raw is None:
        return None
    if policy is not None:
        return policy.stage(raw, source_stage="C0")
    return _pilot_threshold_stage(raw, "eager_style")


POLICY_REQUIRED_STRATEGIES = frozenset({"adaptive_tle", "adaptive_vc", "eager_style"})
