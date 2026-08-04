"""Intraclass correlation (ICC) estimation for instance-level clustering.

Lifted out of ``scripts/analysis_rehearsal/h3_power_simulation.py`` (2026-07-28,
revision_audit_2026-07.md P1-stat-8), where these two estimators existed and worked but were
private helpers only ever exercised against pilot data inside a Monte Carlo power simulation --
there was, before this, no ICC estimator anyone could actually call against the real Phase 1
data. Now public, shared (imported by both the power-sim script and the Stage 1 preanalysis
screen), and independently tested.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def anova_icc1(
    rows: list[dict[str, Any]],
    group_key: str = "instance_key",
    value_key: str = "y_optimal",
) -> float | None:
    """Classical one-way-ANOVA ICC(1) on a 0/1 outcome (cross-check for the GEE-based ICC)."""
    by_group: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        by_group.setdefault(str(r.get(group_key)), []).append(float(v))
    k = len(by_group)
    if k < 2:
        return None
    ns = np.array([len(v) for v in by_group.values()], dtype=float)
    means = np.array([float(np.mean(v)) for v in by_group.values()], dtype=float)
    all_vals = np.array([v for vs in by_group.values() for v in vs], dtype=float)
    grand_mean = float(all_vals.mean())
    n_total = float(ns.sum())
    n0 = (n_total - float((ns**2).sum()) / n_total) / (k - 1)
    msb = float((ns * (means - grand_mean) ** 2).sum()) / (k - 1)
    within_ss = 0.0
    for v in by_group.values():
        arr = np.array(v, dtype=float)
        within_ss += float(((arr - arr.mean()) ** 2).sum())
    dof_w = n_total - k
    if dof_w <= 0 or n0 <= 1:
        return None
    msw = within_ss / dof_w
    denom = msb + (n0 - 1) * msw
    if denom == 0:
        return None
    return (msb - msw) / denom


def gee_icc(
    rows: list[dict[str, Any]],
    group_key: str = "instance_key",
    value_key: str = "y_optimal",
) -> float | None:
    """Intercept-only GEE (Exchangeable, Binomial) -- ``dep_params`` is the working-correlation
    ICC, i.e. exactly the clustering parameter the H3 confirmatory GEE analysis itself relies
    on (methodologically matched, not just a generic ICC estimator)."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    y, g = [], []
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        y.append(float(v))
        g.append(str(r.get(group_key)))
    if len(y) < 20 or len(set(g)) < 3:
        return None
    frame = pd.DataFrame({"y": y, "g": g})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GEE(
                frame["y"],
                sm.add_constant(pd.Series(np.zeros(len(frame)), index=frame.index, name="dummy")),
                groups=frame["g"],
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            res = model.fit()
        return float(res.cov_struct.dep_params)
    except Exception:
        return None


def estimate_icc(
    rows: list[dict[str, Any]],
    *,
    group_key: str = "instance_key",
    value_key: str = "y_optimal",
) -> dict[str, Any]:
    """Both estimators together, GEE preferred / ANOVA as cross-check, plus the row/cluster
    counts the confirmatory GEE analyses themselves gate on (``fit_h3_model`` requires >=20 rows
    and >=3 clusters -- the same thresholds ``gee_icc`` uses internally)."""
    n_valid = [r for r in rows if r.get(value_key) is not None]
    n_clusters = len({str(r.get(group_key)) for r in n_valid})
    return {
        "icc_gee": gee_icc(rows, group_key, value_key),
        "icc_anova": anova_icc1(rows, group_key, value_key),
        "n_rows": len(n_valid),
        "n_clusters": n_clusters,
    }
