"""
Rule-based allocator + baselines: Adaptive-TLE, Adaptive-VC, Always-C0, Always-C2, Random, EAGer-style.
"""

from __future__ import annotations

import random
from typing import Any

# Strategy names as in experiment_core.yaml
STRATEGIES = ("adaptive_tle", "adaptive_vc", "always_c0", "always_c2", "random", "eager_style")


def allocate(
    signal: dict[str, Any] | None,
    strategy: str,
    step_index: int = 0,
    rng: random.Random | None = None,
) -> str:
    """
    Choose compute stage (C0, C1, C2) for this step given strategy and optional signal.

    Args:
        signal: Dict with e.g. 'tle' (mean_entropy), 'vc' (0-100), or None.
        strategy: One of STRATEGIES.
        step_index: Current step (for EAGer-style fixed stage per episode).
        rng: Random generator for 'random' strategy.

    Returns:
        "C0" | "C1" | "C2".
    """
    rng = rng or random.Random()
    if strategy == "always_c0":
        return "C0"
    if strategy == "always_c2":
        return "C2"
    if strategy == "random":
        return rng.choice(["C0", "C1", "C2"])
    if strategy == "eager_style":
        return ["C0", "C1", "C2"][step_index % 3]
    if strategy == "adaptive_tle" and signal and "mean_entropy" in signal:
        h = signal["mean_entropy"]
        if h > 0.8:
            return "C2"
        if h > 0.4:
            return "C1"
        return "C0"
    if strategy == "adaptive_vc" and signal and "vc" in signal:
        vc = signal.get("vc")
        if vc is not None and vc < 50:
            return "C2"
        if vc is not None and vc < 75:
            return "C1"
        return "C0"
    return "C0"


def baseline_always_c0() -> str:
    return "C0"


def baseline_always_c2() -> str:
    return "C2"


def baseline_random(rng: random.Random | None = None) -> str:
    return allocate(None, "random", rng=rng)
