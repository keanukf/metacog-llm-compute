"""Frozen allocation policy from Phase 1 holdout (thesis §5.4)."""

from __future__ import annotations

import bisect
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

SIGNAL_TLE = "tle_mean_entropy"
SIGNAL_VC = "vc"
POLICY_SIGNALS = (SIGNAL_TLE, SIGNAL_VC)
COMPUTE_STAGES = ("C0", "C1", "C2")

LEGACY_POOLED_ECDF_ERROR = (
    "policy artifact uses deprecated pooled ecdf_ref without ecdf_by_stage; "
    "not Phase-1-eligible (regenerate with stage-wise ECDF via write_threshold_artifact). "
    "Pilot-only reload: pass allow_legacy_pooled_ecdf=True."
)


@dataclass(frozen=True)
class FrozenPolicy:
    """Frozen allocation policy: per-stage ECDF percentile thresholds on holdout reference."""

    signal: str
    domain: str
    ecdf_by_stage: dict[str, tuple[float, ...]]
    theta1: float
    theta2: float
    direction: str  # "higher_is_uncertain" | "lower_is_uncertain"

    @property
    def ecdf_ref(self) -> tuple[float, ...]:
        """Pooled ECDF (all stages); diagnostic only — use ``ecdf_by_stage`` at runtime."""
        pooled: list[float] = []
        for stage in COMPUTE_STAGES:
            pooled.extend(self.ecdf_by_stage.get(stage, ()))
        return tuple(sorted(pooled))

    def _ecdf_for_stage(self, source_stage: str) -> tuple[float, ...]:
        stage = str(source_stage).upper()
        ref = self.ecdf_by_stage.get(stage)
        if ref:
            return ref
        raise ValueError(f"no ECDF reference for compute stage {stage!r}")

    def percentile(self, x: float, *, source_stage: str = "C0") -> float:
        ecdf_ref = self._ecdf_for_stage(source_stage)
        if not ecdf_ref:
            raise ValueError(f"empty ECDF reference for stage {source_stage!r}")
        lo = bisect.bisect_left(ecdf_ref, x)
        hi = bisect.bisect_right(ecdf_ref, x)
        return ((lo + hi) / 2.0) / len(ecdf_ref)

    def uncertainty_score(self, x: float, *, source_stage: str = "C0") -> float:
        p = self.percentile(x, source_stage=source_stage)
        if self.direction == "higher_is_uncertain":
            return p
        return 1.0 - p

    def stage(self, x: float, *, source_stage: str = "C0") -> str:
        s = self.uncertainty_score(x, source_stage=source_stage)
        if s < self.theta1:
            return "C0"
        if s < self.theta2:
            return "C1"
        return "C2"


def _load_ecdf_by_stage(
    block: dict,
    *,
    allow_legacy_pooled_ecdf: bool = False,
) -> dict[str, tuple[float, ...]]:
    if "ecdf_by_stage" in block:
        raw = block["ecdf_by_stage"]
        if not isinstance(raw, dict):
            raise ValueError("ecdf_by_stage must be an object")
        out: dict[str, tuple[float, ...]] = {}
        for stage in COMPUTE_STAGES:
            vals = raw.get(stage)
            if vals is None:
                continue
            out[stage] = tuple(sorted(float(v) for v in vals))
        if not out:
            raise ValueError("ecdf_by_stage is empty")
        return out
    if "ecdf_ref" in block:
        if not allow_legacy_pooled_ecdf:
            raise ValueError(LEGACY_POOLED_ECDF_ERROR)
        warnings.warn(
            "loading legacy pooled ecdf_ref artifact; replicates the same ECDF on all stages — "
            "not valid for Phase 1/2 (pilot-only opt-in)",
            UserWarning,
            stacklevel=3,
        )
        ref = tuple(sorted(float(v) for v in block["ecdf_ref"]))
        return {stage: ref for stage in COMPUTE_STAGES}
    raise ValueError("policy block missing ecdf_by_stage")


def load_policy(
    path: str | Path,
    *,
    domain: str,
    signal: str,
    allow_legacy_pooled_ecdf: bool = False,
) -> FrozenPolicy:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    d = obj["by_domain"][domain][signal]
    return FrozenPolicy(
        signal=signal,
        domain=domain,
        ecdf_by_stage=_load_ecdf_by_stage(
            d,
            allow_legacy_pooled_ecdf=allow_legacy_pooled_ecdf,
        ),
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
