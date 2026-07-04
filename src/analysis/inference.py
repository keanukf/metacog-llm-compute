"""
Confirmatory inference utilities (thesis §5.8): cluster bootstrap, H2/H3/H4 contrasts, multiplicity control.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable


def _as_rows(df: Any) -> list[dict[str, Any]]:
    if isinstance(df, list):
        return [r for r in df if isinstance(r, dict)]
    if hasattr(df, "to_dict"):
        return list(df.to_dict(orient="records"))
    raise TypeError("df must be a list of dicts or pandas DataFrame")


def _skewness(vals: list[float]) -> float | None:
    if len(vals) < 3:
        return None
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    if v <= 0:
        return None
    s = math.sqrt(v)
    m3 = sum((x - m) ** 3 for x in vals) / len(vals)
    return m3 / (s**3)


def cluster_bootstrap(
    df: Any,
    stat_fn: Callable[[list[dict[str, Any]]], float],
    *,
    cluster_col: str = "instance_key",
    n_boot: int = 5000,
    seed: int = 20260703,
    ci: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """
    Resample clusters with replacement; percentile CI on ``stat_fn``.

    One-sided alpha=.05 uses the relevant bound of the 90% interval (thesis §5.8).
    """
    rows = _as_rows(df)
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cluster[str(r.get(cluster_col, "unknown"))].append(r)
    clusters = list(by_cluster.keys())
    if len(clusters) < 2:
        return {"point": None, "ci_low": None, "ci_high": None, "n_boot": n_boot, "skewness": None}
    rng = random.Random(seed)
    point = stat_fn(rows)
    reps: list[float] = []
    for _ in range(n_boot):
        sample_keys = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        boot_rows: list[dict[str, Any]] = []
        for k in sample_keys:
            boot_rows.extend(by_cluster[k])
        try:
            reps.append(float(stat_fn(boot_rows)))
        except Exception:
            continue
    if not reps:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot": n_boot, "skewness": None}
    reps.sort()
    lo_i = int(ci[0] * n_boot)
    hi_i = min(int(ci[1] * n_boot), n_boot - 1)
    return {
        "point": float(point),
        "ci_low": float(reps[lo_i]),
        "ci_high": float(reps[hi_i]),
        "n_boot": n_boot,
        "skewness": _skewness(reps),
    }


def delta_auroc(
    df: Any,
    *,
    signal_a: str = "tle",
    signal_b: str = "vc",
    label: str = "y_optimal",
) -> float:
    """AUROC(signal_a) - AUROC(signal_b); TLE score = negated mean entropy."""
    from src.analysis.calibration import compute_auroc

    rows = _as_rows(df)
    ys: list[int] = []
    sa: list[float] = []
    sb: list[float] = []
    for r in rows:
        y = r.get(label)
        if y is None:
            continue
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        tle = r.get("tle_mean_entropy")
        vc = r.get("vc")
        if tle is None or vc is None:
            continue
        ys.append(yi)
        sa.append(-float(tle))
        sb.append(float(vc))
    if len(set(ys)) < 2:
        return float("nan")
    return float(compute_auroc(sa, ys) - compute_auroc(sb, ys))


def delta_brier_after_mapping(
    df: Any,
    calibrator_tle: Any,
    *,
    label: str = "y_optimal",
) -> float:
    from src.analysis.calibration import compute_brier

    rows = _as_rows(df)
    ys: list[int] = []
    ps: list[float] = []
    for r in rows:
        y = r.get(label)
        tle = r.get("tle_mean_entropy")
        if y is None or tle is None:
            continue
        ys.append(int(y))
        ps.append(float(calibrator_tle.predict_proba(float(tle))))
    if not ys:
        return float("nan")
    return float(compute_brier(ps, ys))


def h4_diff_in_diff(df: Any, *, label: str = "y_optimal") -> float:
    """(AUROC_TLE_TW - AUROC_TLE_TOH) - (AUROC_VC_TW - AUROC_VC_TOH)."""
    rows = _as_rows(df)
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dom[str(r.get("domain", ""))].append(r)

    def _delta(dom: str) -> float:
        sub = by_dom.get(dom, [])
        return delta_auroc(sub, label=label)

    if "textworld" not in by_dom or "tower_of_hanoi" not in by_dom:
        return float("nan")
    return float(_delta("textworld") - _delta("tower_of_hanoi"))


def h2_paired(
    episodes: list[dict[str, Any]],
    *,
    policy_strategy: str = "adaptive_tle",
    baseline: str = "always_c2",
    delta: float = 0.05,
) -> dict[str, Any]:
    """
    Paired cluster contrast: success non-inferiority (ΔP > -δ) and log-token superiority.

    §5.8 default: ``delta=0.05``.
    """
    by_inst: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for ep in episodes:
        if bool(ep.get("holdout")):
            continue
        key = f"{ep.get('domain')}:{ep.get('instance')}"
        strat = str(ep.get("strategy", ""))
        by_inst[key][strat] = ep
    succ_diffs: list[float] = []
    log_tok_diffs: list[float] = []
    for _k, m in by_inst.items():
        if policy_strategy not in m or baseline not in m:
            continue
        p = m[policy_strategy]
        b = m[baseline]
        succ_diffs.append(float(bool(p.get("task_success"))) - float(bool(b.get("task_success"))))
        tp = max(1.0, float(p.get("total_tokens_generated") or 1))
        tb = max(1.0, float(b.get("total_tokens_generated") or 1))
        log_tok_diffs.append(math.log(tb) - math.log(tp))
    if not succ_diffs:
        return {"n_pairs": 0, "delta": delta}
    mean_succ = sum(succ_diffs) / len(succ_diffs)
    mean_log = sum(log_tok_diffs) / len(log_tok_diffs)
    return {
        "n_pairs": len(succ_diffs),
        "delta": delta,
        "mean_success_diff": mean_succ,
        "mean_log_token_diff": mean_log,
        "non_inferiority_holds": mean_succ > -delta,
        "token_superiority_holds": mean_log > 0,
    }


def holm(pvals_or_bounds: list[float], family: str = "") -> list[dict[str, Any]]:
    """Holm step-down adjustment (family A–C)."""
    m = len(pvals_or_bounds)
    order = sorted(range(m), key=lambda i: pvals_or_bounds[i])
    out: list[dict[str, Any]] = [{}] * m
    for rank, idx in enumerate(order):
        adj = min(1.0, pvals_or_bounds[idx] * (m - rank))
        out[idx] = {"index": idx, "raw": pvals_or_bounds[idx], "adjusted": adj, "family": family}
    return out


def bh(pvals_or_bounds: list[float]) -> list[dict[str, Any]]:
    """Benjamini–Hochberg FDR adjustment (exploratory level)."""
    m = len(pvals_or_bounds)
    order = sorted(range(m), key=lambda i: pvals_or_bounds[i])
    adj = [1.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(prev, pvals_or_bounds[idx] * m / (rank + 1))
        adj[idx] = val
        prev = val
    return [{"index": i, "raw": pvals_or_bounds[i], "adjusted": adj[i]} for i in range(m)]


def fit_h3_model(
    df: Any,
    *,
    signal: str = "tle",
    domain: str | None = None,
) -> dict[str, Any]:
    """
    GEE: y ~ z_signal * position_norm with exchangeable instance clustering.

    Fallback: returns error dict if statsmodels GEE fails to converge.
    """
    rows = _as_rows(df)
    if domain is not None:
        rows = [r for r in rows if str(r.get("domain")) == domain]
    y: list[int] = []
    z: list[float] = []
    pos: list[float] = []
    groups: list[str] = []
    for r in rows:
        if r.get("y_optimal") is None:
            continue
        tle = r.get("tle_mean_entropy")
        vc = r.get("vc")
        if signal == "tle":
            if tle is None:
                continue
            zv = -float(tle)
        else:
            if vc is None:
                continue
            zv = float(vc)
        y.append(int(r["y_optimal"]))
        z.append(zv)
        pos.append(float(r.get("position_norm") or r.get("relative_step_position") or 0))
        groups.append(str(r.get("instance_key", "unknown")))
    if len(y) < 20 or len(set(groups)) < 3:
        return {"converged": False, "note": "insufficient data"}
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial

        frame = pd.DataFrame({"y": y, "z": z, "position_norm": pos, "g": groups})
        frame["z_c"] = frame["z"] - frame["z"].mean()
        frame["p_c"] = frame["position_norm"] - frame["position_norm"].mean()
        model = sm.GEE(
            frame["y"],
            sm.add_constant(frame[["z_c", "p_c"]].assign(interaction=frame["z_c"] * frame["p_c"])),
            groups=frame["g"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        res = model.fit()
        return {
            "converged": True,
            "params": {k: float(v) for k, v in res.params.items()},
            "pvalues": {k: float(v) for k, v in res.pvalues.items()},
            "signal": signal,
            "domain": domain,
        }
    except Exception as e:
        return {"converged": False, "note": str(e), "signal": signal, "domain": domain}
