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
    collapse_policy: str = "optimal_only",
    git_commit: str | None = None,
    config_hash: str | None = None,
) -> Path:
    """
    Write policy artifact JSON (§5.4) with ECDF + gridsearch when holdout flags are present.
    Falls back to legacy quantile thresholds when no holdout metadata exists.
    """
    rows = [r for r in steps_rows if isinstance(r, dict)]
    has_holdout = any("holdout" in r for r in rows)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if has_holdout:
        obj = build_policy_artifact(
            rows,
            signals=signals,
            collapse_policy=collapse_policy,
            git_commit=git_commit,
            config_hash=config_hash,
        )
    else:
        obj = learn_thresholds_by_domain(rows, signals=signals)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p


# --- §5.4 gridsearch (step_level_proxy_v1) ---

OBJECTIVE_DEFINITION = "step_level_proxy_v1"
DEFAULT_QUANTILE_GRID: tuple[float, ...] = tuple(i / 10 for i in range(1, 10))


def _signal_column(signal: SignalType) -> str:
    return "tle_mean_entropy" if signal == "tle_mean_entropy" else "vc"


def _direction_for_signal(signal: SignalType) -> str:
    return "higher_is_uncertain" if signal == "tle_mean_entropy" else "lower_is_uncertain"


def _raw_signal_from_row(row: dict[str, Any], signal: SignalType) -> float | None:
    col = _signal_column(signal)
    v = row.get(col)
    if v is None and signal == "tle_mean_entropy":
        tle = row.get("tle")
        if isinstance(tle, dict):
            v = tle.get("mean_entropy")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_ecdf_ref(
    holdout_steps: Iterable[dict[str, Any]], *, signal: SignalType
) -> tuple[float, ...]:
    xs: list[float] = []
    for r in holdout_steps:
        if not isinstance(r, dict):
            continue
        v = _raw_signal_from_row(r, signal)
        if v is not None:
            xs.append(v)
    return tuple(sorted(xs))


def _label_from_row(row: dict[str, Any], label_key: str) -> int | None:
    v = row.get(label_key)
    if v is None:
        corr = row.get("correctness") or row.get("step_correctness_raw")
        if label_key == "y_optimal":
            if corr == "optimal":
                return 1
            if corr in ("legal", "illegal"):
                return 0
        elif label_key == "y_legal_or_optimal":
            if corr in ("optimal", "legal"):
                return 1
            if corr == "illegal":
                return 0
        return None
    try:
        return 1 if int(v) == 1 else 0
    except (TypeError, ValueError):
        return None


def _tokens_from_row(row: dict[str, Any]) -> float:
    for k in ("tokens_step", "tokens_generated"):
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _index_phase1_pool(rows: Iterable[dict[str, Any]], label_key: str) -> dict[tuple, list[dict]]:
    pool: dict[tuple, list[dict]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            inst_raw = r.get("instance")
            run_raw = r.get("run", 0)
            step_raw = r.get("step_index")
            if inst_raw is None or step_raw is None:
                continue
            inst = int(inst_raw)
            run = int(run_raw)
            step_index = int(step_raw)
            stage = str(r.get("compute_stage", "")).upper()
        except (TypeError, ValueError):
            continue
        if stage not in {"C0", "C1", "C2"}:
            continue
        y = _label_from_row(r, label_key)
        if y is None:
            continue
        key = (inst, run, step_index, stage)
        pool.setdefault(key, []).append(
            {"y": y, "tokens": _tokens_from_row(r), "step_index": step_index}
        )
    return pool


def _match_proxy(
    pool: dict[tuple, list[dict]],
    *,
    instance: int,
    run: int,
    step_index: int,
    stage: str,
) -> tuple[float | None, float | None, str]:
    """Fallback cascade; returns (y, tokens, match_level)."""
    stage = stage.upper()
    k1 = (instance, run, step_index, stage)
    if k1 in pool and pool[k1]:
        rec = pool[k1][0]
        return float(rec["y"]), float(rec["tokens"]), "exact"
    k2_matches = [
        pool[k][0] for k in pool if k[0] == instance and k[2] == step_index and k[3] == stage
    ]
    if k2_matches:
        ys = [m["y"] for m in k2_matches]
        ts = [m["tokens"] for m in k2_matches]
        return sum(ys) / len(ys), sum(ts) / len(ts), "mean_run"
    candidates = [(k, pool[k][0]) for k in pool if k[0] == instance and k[3] == stage]
    if candidates:
        nearest = min(candidates, key=lambda x: abs(int(x[0][2]) - step_index))
        rec = nearest[1]
        return float(rec["y"]), float(rec["tokens"]), "nearest_position"
    return None, None, "none"


def _pareto_nondominated(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nd: list[dict[str, Any]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["success_proxy"] >= p["success_proxy"] and q["token_proxy"] <= p["token_proxy"]:
                if q["success_proxy"] > p["success_proxy"] or q["token_proxy"] < p["token_proxy"]:
                    dominated = True
                    break
        if not dominated:
            nd.append(p)
    return nd


def grid_search_thresholds(
    holdout_steps: list[dict],
    phase1_pool_rows: list[dict],
    *,
    signal: SignalType,
    quantile_grid: tuple[float, ...] = DEFAULT_QUANTILE_GRID,
    label_key: str = "y_optimal",
) -> dict[str, Any]:
    """Thesis §5.4: select (theta1, theta2) on holdout ECDF with step-level proxy objective."""
    direction = _direction_for_signal(signal)
    ecdf_ref = build_ecdf_ref(holdout_steps, signal=signal)
    if len(ecdf_ref) < 10:
        return {
            "signal": signal,
            "theta1": None,
            "theta2": None,
            "ecdf_ref": list(ecdf_ref),
            "direction": direction,
            "grid_table": [],
            "objective_definition": OBJECTIVE_DEFINITION,
            "note": "insufficient holdout samples",
        }
    from src.agent.allocation_policy import FrozenPolicy

    pool = _index_phase1_pool(phase1_pool_rows, label_key)
    grid_table: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for i, t1 in enumerate(quantile_grid):
        for t2 in quantile_grid[i + 1 :]:
            pol = FrozenPolicy(
                signal=signal,
                domain="",
                ecdf_ref=ecdf_ref,
                theta1=float(t1),
                theta2=float(t2),
                direction=direction,
            )
            ys: list[float] = []
            toks: list[float] = []
            matched = 0
            total = 0
            for row in holdout_steps:
                raw = _raw_signal_from_row(row, signal)
                if raw is None:
                    continue
                total += 1
                stage = pol.stage(raw)
                try:
                    inst_raw = row.get("instance")
                    run_raw = row.get("run", 0)
                    step_raw = row.get("step_index")
                    if inst_raw is None or step_raw is None:
                        continue
                    inst = int(inst_raw)
                    run = int(run_raw)
                    step_index = int(step_raw)
                except (TypeError, ValueError):
                    continue
                y, tok, _lvl = _match_proxy(
                    pool, instance=inst, run=run, step_index=step_index, stage=stage
                )
                if y is not None and tok is not None:
                    matched += 1
                    ys.append(y)
                    toks.append(tok)
            success_proxy = sum(ys) / len(ys) if ys else 0.0
            token_proxy = sum(toks) if toks else float("inf")
            match_rate = (matched / total) if total else 0.0
            entry = {
                "theta1": float(t1),
                "theta2": float(t2),
                "success_proxy": success_proxy,
                "token_proxy": token_proxy,
                "match_rate": match_rate,
                "n_eval_steps": total,
            }
            grid_table.append(entry)
            candidates.append(entry)
    front = _pareto_nondominated(candidates)
    if not front:
        best = candidates[0] if candidates else None
    else:
        best = min(front, key=lambda x: (x["token_proxy"], -x["success_proxy"]))
    return {
        "signal": signal,
        "theta1": best["theta1"] if best else None,
        "theta2": best["theta2"] if best else None,
        "ecdf_ref": list(ecdf_ref),
        "direction": direction,
        "grid_table": grid_table,
        "objective_definition": OBJECTIVE_DEFINITION,
        "selected": best,
        "pareto_front": front,
    }


def build_policy_artifact(
    steps_rows: list[dict[str, Any]],
    *,
    signals: tuple[SignalType, ...] = ("vc", "tle_mean_entropy"),
    collapse_policy: str = "optimal_only",
    git_commit: str | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
    label_key = "y_optimal" if collapse_policy == "optimal_only" else "y_legal_or_optimal"
    holdout_rows = [r for r in steps_rows if bool(r.get("holdout"))]
    non_holdout_rows = [r for r in steps_rows if not bool(r.get("holdout"))]
    by_domain: dict[str, Any] = {}
    domains = sorted({str(r.get("domain", "unknown")) for r in steps_rows})
    for dom in domains:
        dom_all = [r for r in steps_rows if str(r.get("domain")) == dom]
        dom_holdout = [r for r in holdout_rows if str(r.get("domain")) == dom]
        dom_non_holdout = [r for r in non_holdout_rows if str(r.get("domain")) == dom]
        dom_out: dict[str, Any] = {}
        for sig in signals:
            gs = grid_search_thresholds(
                dom_holdout,
                dom_all,
                signal=sig,
                label_key=label_key,
            )
            platt_fit = fit_calibrator_from_steps(
                dom_holdout,
                signal=sig,
                label_key="step_correct_optimal" if label_key == "y_optimal" else label_key,
            )
            platt_eval_rows = dom_non_holdout
            xs_ev, ys_ev = _extract_xy(
                platt_eval_rows,
                signal=sig,
                label_key="step_correct_optimal" if label_key == "y_optimal" else label_key,
            )
            brier_eval = None
            cal = platt_fit.get("calibrator")
            if cal and isinstance(cal, dict) and xs_ev:
                from src.analysis.calibration import compute_brier

                pc = PlattCalibrator.from_json(cal)
                probs = [pc.predict_proba(x) for x in xs_ev]
                brier_eval = compute_brier(probs, ys_ev)
            dom_out[str(sig)] = {
                **{
                    k: gs[k]
                    for k in (
                        "ecdf_ref",
                        "theta1",
                        "theta2",
                        "direction",
                        "grid_table",
                        "objective_definition",
                    )
                },
                "calibrator": platt_fit,
                "platt_fit": "holdout",
                "platt_eval": "non_holdout_confirmatory",
                "brier_eval_non_holdout": brier_eval,
            }
        by_domain[dom] = dom_out
    return {
        "by_domain": by_domain,
        "collapse_policy": collapse_policy,
        "platt_fit": "holdout",
        "platt_eval": "non_holdout_confirmatory",
        "git_commit": git_commit,
        "config_hash": config_hash,
    }
