"""Frozen allocation policy from Phase 1 holdout (thesis §5.4)."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

SIGNAL_TLE = "tle_mean_entropy"
SIGNAL_VC = "vc"
POLICY_SIGNALS = (SIGNAL_TLE, SIGNAL_VC)


@dataclass(frozen=True)
class FrozenPolicy:
    """Frozen allocation policy: ECDF percentile thresholds on holdout reference sample."""

    signal: str
    domain: str
    ecdf_ref: tuple[float, ...]
    theta1: float
    theta2: float
    direction: str  # "higher_is_uncertain" | "lower_is_uncertain"

    def percentile(self, x: float) -> float:
        if not self.ecdf_ref:
            raise ValueError("empty ECDF reference")
        lo = bisect.bisect_left(self.ecdf_ref, x)
        hi = bisect.bisect_right(self.ecdf_ref, x)
        return ((lo + hi) / 2.0) / len(self.ecdf_ref)

    def uncertainty_score(self, x: float) -> float:
        p = self.percentile(x)
        if self.direction == "higher_is_uncertain":
            return p
        return 1.0 - p

    def stage(self, x: float) -> str:
        s = self.uncertainty_score(x)
        if s < self.theta1:
            return "C0"
        if s < self.theta2:
            return "C1"
        return "C2"


def load_policy(path: str | Path, *, domain: str, signal: str) -> FrozenPolicy:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    d = obj["by_domain"][domain][signal]
    return FrozenPolicy(
        signal=signal,
        domain=domain,
        ecdf_ref=tuple(sorted(float(v) for v in d["ecdf_ref"])),
        theta1=float(d["theta1"]),
        theta2=float(d["theta2"]),
        direction=str(d["direction"]),
    )


def policy_signal_for_strategy(strategy: str) -> str | None:
    if strategy == "adaptive_tle":
        return SIGNAL_TLE
    if strategy == "adaptive_vc":
        return SIGNAL_VC
    if strategy == "eager_style":
        return SIGNAL_TLE
    return None
