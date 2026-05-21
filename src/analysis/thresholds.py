"""
Phase-1 calibration and threshold learning utilities.

Goals:
- Fit a simple monotonic calibrator mapping a raw signal to p(optimal).
- Derive two thresholds to map signals into stages (C0/C1/C2) for allocators.

This module is intentionally light: uses scipy if available, otherwise falls back to
non-fitted heuristics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

SignalType = Literal["vc", "tle_mean_entropy"]


@dataclass(frozen=True)
class PlattCalibrator:
    """
    Logistic calibrator: p = sigmoid(a * x + b).
    """

    a: float
    b: float

    def predict_proba(self, x: float) -> float:
        z = self.a * float(x) + self.b
        # stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def to_json(self) -> dict[str, Any]:
        return {"type": "platt_logistic", "a": float(self.a), "b": float(self.b)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "PlattCalibrator":
        return PlattCalibrator(a=float(d["a"]), b=float(d["b"]))


def _extract_xy(
    steps_rows: Iterable[dict[str, Any]],
    *,
    signal: SignalType,
    label_key: str = "step_correct_optimal",
) -> tuple[list[float], list[int]]:
    xs: list[float] = []
    ys: list[int] = []
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        y = r.get(label_key)
        if y is None:
            continue
        try:
            y01 = 1 if int(y) == 1 else 0
        except Exception:
            continue
        v = r.get(signal)
        if v is None:
            continue
        try:
            xs.append(float(v))
            ys.append(int(y01))
        except Exception:
            continue
    return xs, ys


def fit_platt_calibrator(
    xs: list[float],
    ys: list[int],
    *,
    l2: float = 1e-3,
) -> PlattCalibrator | None:
    """
    Fit Platt scaling parameters (a,b) with L2 regularization.
    Returns None if fitting is not possible.
    """
    if len(xs) != len(ys) or len(xs) < 20:
        return None
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    try:
        import numpy as np  # type: ignore
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # Standardize x for numerics
    mu = float(x.mean())
    sig = float(x.std()) if float(x.std()) > 1e-8 else 1.0
    xz = (x - mu) / sig

    def _nll(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        z = a * xz + b
        # stable logistic loss
        # nll = sum(log(1+exp(z)) - y*z)
        nll = float(np.logaddexp(0.0, z).sum() - (y * z).sum())
        nll += float(l2 * (a * a + b * b))
        return nll

    res = minimize(_nll, x0=np.array([0.0, 0.0], dtype=float), method="BFGS")
    if not res.success:
        return None
    a_z, b = float(res.x[0]), float(res.x[1])
    # Undo standardization: a*xz + b = (a/sig)*x + (b - a*mu/sig)
    a = a_z / sig
    b0 = b - a_z * (mu / sig)
    return PlattCalibrator(a=float(a), b=float(b0))


def fit_calibrator_from_steps(
    steps_rows: Iterable[dict[str, Any]],
    *,
    signal: SignalType,
    label_key: str = "step_correct_optimal",
) -> dict[str, Any]:
    """
    Fit a calibrator from step rows and return a JSON-serializable artifact.
    """
    xs, ys = _extract_xy(steps_rows, signal=signal, label_key=label_key)
    cal = fit_platt_calibrator(xs, ys)
    if cal is None:
        return {
            "signal": signal,
            "calibrator": None,
            "n_samples": len(xs),
            "note": "platt fit unavailable; use heuristic mapping",
        }
    return {
        "signal": signal,
        "calibrator": cal.to_json(),
        "n_samples": len(xs),
        "note": "p(optimal)=sigmoid(a*x+b)",
    }


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round(q * (len(xs_sorted) - 1)))
    return float(xs_sorted[idx])


def derive_stage_thresholds(
    steps_rows: Iterable[dict[str, Any]],
    *,
    signal: SignalType,
    low_quantile: float = 0.33,
    high_quantile: float = 0.66,
) -> dict[str, Any]:
    """
    Derive two thresholds from the empirical signal distribution.

    For TLE mean entropy (higher=harder), the allocator escalates when signal is high.
    For VC (higher=easier), the allocator escalates when signal is low.
    """
    xs, _ys = _extract_xy(steps_rows, signal=signal, label_key="step_correct_optimal")
    if len(xs) < 10:
        return {"signal": signal, "t_c0_c1": None, "t_c1_c2": None, "n_samples": len(xs)}

    if signal == "tle_mean_entropy":
        t1 = _quantile(xs, high_quantile)  # escalate to C1 above t1
        t2 = _quantile(
            xs, min(0.95, high_quantile + (1 - high_quantile) / 2)
        )  # more extreme for C2
        # Ensure ordering
        if t2 < t1:
            t1, t2 = t2, t1
        return {
            "signal": signal,
            "t_c0_c1": float(t1),
            "t_c1_c2": float(t2),
            "direction": "higher_is_harder",
            "n_samples": len(xs),
        }

    # VC: lower = harder; thresholds are "below" cutoffs
    t2 = _quantile(xs, low_quantile)  # escalate to C2 below t2
    t1 = _quantile(
        xs, max(0.05, low_quantile / 2)
    )  # very low for C2? keep t1 <= t2? We want C2 below t2 and C1 below t1? Better:
    # For VC we define: if vc < t_c1_c2 -> C2; elif vc < t_c0_c1 -> C1; else C0, so t_c1_c2 < t_c0_c1.
    t_c1_c2 = _quantile(xs, low_quantile)
    t_c0_c1 = _quantile(xs, high_quantile)
    if t_c1_c2 > t_c0_c1:
        t_c1_c2, t_c0_c1 = t_c0_c1, t_c1_c2
    return {
        "signal": signal,
        "t_c0_c1": float(t_c0_c1),
        "t_c1_c2": float(t_c1_c2),
        "direction": "lower_is_harder",
        "n_samples": len(xs),
    }


def learn_thresholds_by_domain(
    steps_rows: Iterable[dict[str, Any]],
    *,
    signals: tuple[SignalType, ...] = ("vc", "tle_mean_entropy"),
) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        d = str(r.get("domain", "unknown"))
        by_domain.setdefault(d, []).append(r)

    out: dict[str, Any] = {"by_domain": {}}
    for d, rows in sorted(by_domain.items()):
        dom: dict[str, Any] = {}
        for sig in signals:
            dom[str(sig)] = {
                "calibrator": fit_calibrator_from_steps(rows, signal=sig),
                "thresholds": derive_stage_thresholds(rows, signal=sig),
            }
        out["by_domain"][d] = dom
    return out


def write_threshold_artifact(
    output_path: str | Path,
    steps_rows: Iterable[dict[str, Any]],
    *,
    signals: tuple[SignalType, ...] = ("vc", "tle_mean_entropy"),
) -> Path:
    """
    Write a JSON artifact for calibrators+thresholds (per domain).
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = learn_thresholds_by_domain(steps_rows, signals=signals)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p
