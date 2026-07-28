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


def _summarize_bootstrap_reps(
    point: float | None,
    reps: list[float],
    *,
    n_boot: int,
    n_nonfinite: int,
    ci: tuple[float, float],
) -> dict[str, Any]:
    """Shared point/CI/skewness tail logic for ``cluster_bootstrap`` and
    ``cluster_bootstrap_stratified`` -- percentile CI + skewness from a raw replicate list, so
    the two resampling strategies don't duplicate this bookkeeping."""
    if not reps:
        return {
            "point": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot": n_boot,
            "skewness": None,
            "reps": [],
        }
    sorted_reps = sorted(reps)
    n_eff = len(sorted_reps)
    lo_i = int(ci[0] * n_eff)
    hi_i = min(int(ci[1] * n_eff), n_eff - 1)
    return {
        "point": float(point) if point is not None else None,
        "ci_low": float(sorted_reps[lo_i]),
        "ci_high": float(sorted_reps[hi_i]),
        "n_boot": n_boot,
        "n_boot_effective": n_eff,
        "n_boot_nonfinite": n_nonfinite,
        "skewness": _skewness(sorted_reps),
        "reps": sorted_reps,
    }


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
        return _summarize_bootstrap_reps(None, [], n_boot=n_boot, n_nonfinite=0, ci=ci)
    rng = random.Random(seed)
    point = stat_fn(rows)
    reps: list[float] = []
    n_nonfinite = 0
    for _ in range(n_boot):
        sample_keys = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        boot_rows: list[dict[str, Any]] = []
        for k in sample_keys:
            boot_rows.extend(by_cluster[k])
        try:
            v = float(stat_fn(boot_rows))
        except Exception:
            n_nonfinite += 1
            continue
        # A resample of clusters can be degenerate (e.g. contain only one label class),
        # which makes some stat_fn implementations (e.g. delta_auroc) return nan/inf instead
        # of raising. Python's list.sort() does not order nan consistently, so a nan slipping
        # into `reps` silently corrupts the percentile lookup below -- this reproduced the
        # exact anomaly noted as an unexplained "rough edge" in the Gate E rehearsal
        # (docs/gate_e_rehearsal.md, section 6: point estimate landing outside the reported
        # CI, skewness NaN) on the real TextWorld pilot pool (9 clusters). Drop non-finite
        # replicates instead of letting them pollute the sorted percentile array.
        if not math.isfinite(v):
            n_nonfinite += 1
            continue
        reps.append(v)
    return _summarize_bootstrap_reps(point, reps, n_boot=n_boot, n_nonfinite=n_nonfinite, ci=ci)


def cluster_bootstrap_stratified(
    df: Any,
    stat_fn: Callable[[list[dict[str, Any]]], float],
    *,
    strata_col: str = "domain",
    cluster_col: str = "instance_key",
    n_boot: int = 5000,
    seed: int = 20260703,
    ci: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """
    Like ``cluster_bootstrap``, but resamples instance clusters *within each stratum
    independently* before recombining into one resample per replicate.

    H4's preregistration (verified verbatim in ``chapters/05_methodology.md``, thesis Ch.5 §5.8):
    "estimated by the cluster bootstrap with instances resampled within each domain." Plain
    ``cluster_bootstrap`` pools all clusters from every stratum into one flat resampling pool,
    which does not match that wording -- a single replicate could draw a domain-imbalanced mix
    of clusters purely by chance. ``stat_fn`` (e.g. ``h4_diff_in_diff``) needs zero changes: it
    already re-splits the combined rows by domain internally before computing its statistic.
    """
    rows = _as_rows(df)
    by_stratum_cluster: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        stratum = str(r.get(strata_col, "unknown"))
        cluster = str(r.get(cluster_col, "unknown"))
        by_stratum_cluster[stratum][cluster].append(r)

    strata = list(by_stratum_cluster.keys())
    cluster_lists = {s: list(by_stratum_cluster[s].keys()) for s in strata}
    if not strata or any(len(cluster_lists[s]) < 2 for s in strata):
        return _summarize_bootstrap_reps(None, [], n_boot=n_boot, n_nonfinite=0, ci=ci)

    rng = random.Random(seed)
    point = stat_fn(rows)
    reps: list[float] = []
    n_nonfinite = 0
    for _ in range(n_boot):
        boot_rows: list[dict[str, Any]] = []
        for s in strata:
            clusters_s = cluster_lists[s]
            sample_keys = [clusters_s[rng.randrange(len(clusters_s))] for _ in range(len(clusters_s))]
            for k in sample_keys:
                boot_rows.extend(by_stratum_cluster[s][k])
        try:
            v = float(stat_fn(boot_rows))
        except Exception:
            n_nonfinite += 1
            continue
        if not math.isfinite(v):
            n_nonfinite += 1
            continue
        reps.append(v)
    return _summarize_bootstrap_reps(point, reps, n_boot=n_boot, n_nonfinite=n_nonfinite, ci=ci)


def one_sided_bootstrap_pvalue(reps: list[float], *, null_value: float = 0.0) -> float:
    """One-sided bootstrap p-value for testing "statistic > null_value": the fraction of
    replicates at or below ``null_value``, with the standard ``(n_exceed + 1) / (n + 1)``
    continuity correction (Davison & Hinkley 1997) so a p-value is never exactly 0 -- that would
    misrepresent a finite-resample estimate as an exact certainty and would break Holm's
    downstream arithmetic (``raw * (m - rank)``, which is degenerate at exactly 0).

    Feeds ``holm()``/``bh()`` a real, monotonic-in-significance quantity from a cluster-bootstrap
    replicate list -- the CI bound itself is not a p-value and must not be passed to Holm
    directly.
    """
    n = len(reps)
    if n == 0:
        return 1.0
    n_le = sum(1 for r in reps if r <= null_value)
    return (n_le + 1) / (n + 1)


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
    """
    [AUROC_TLE - AUROC_VC]_ToH - [AUROC_TLE - AUROC_VC]_TextWorld.

    Sign matches the preregistered confirmatory contrast (thesis H_{1,4}, ch.4 §4.2.2):
    the TLE-over-VC discrimination advantage is predicted to be larger under full
    observability (ToH) than partial observability (TextWorld), so H4 is supported
    when this returns a value > 0 (a domain order swap here silently flips the
    confirmatory decision rule -- do not "simplify" this back to textworld-minus-toh).
    """
    rows = _as_rows(df)
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dom[str(r.get("domain", ""))].append(r)

    def _delta(dom: str) -> float:
        sub = by_dom.get(dom, [])
        return delta_auroc(sub, label=label)

    if "textworld" not in by_dom or "tower_of_hanoi" not in by_dom:
        return float("nan")
    return float(_delta("tower_of_hanoi") - _delta("textworld"))


def h2_paired(
    episodes: list[dict[str, Any]],
    *,
    policy_strategy: str = "adaptive_tle",
    baseline: str = "always_c2",
    delta: float = 0.05,
    n_boot: int = 5000,
    seed: int = 20260703,
) -> dict[str, Any]:
    """
    Paired cluster-bootstrap contrast: success non-inferiority and log-token superiority,
    decided on the bootstrap CI bound over resampled instances -- not the raw point estimate.

    §5.8 default: ``delta=0.05``; decision rule per thesis Ch.5 §5.8: "H2 holds ... only when
    both one-sided bootstrap intervals satisfy their bounds." An earlier version compared
    ``mean_success_diff``/``mean_log_token_diff`` directly against the threshold, which ignores
    the paired-cluster non-independence the rest of the inference engine exists to handle (see
    revision_audit P0-7, ``notes/praeregistrierung_auswertungsplan.md`` §5/§10 in
    ../metacog-thesis) -- a point estimate crossing a threshold is a materially weaker claim than
    a CI bound crossing it.

    Both ``succ_diff`` (policy success − baseline success) and ``log_tok_diff`` (log baseline
    tokens − log policy tokens) are oriented so that *larger is better*; both decision rules
    therefore use the **lower** CI bound.
    """
    by_inst: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for ep in episodes:
        if bool(ep.get("holdout")):
            continue
        key = f"{ep.get('domain')}:{ep.get('instance')}"
        strat = str(ep.get("strategy", ""))
        by_inst[key][strat] = ep

    paired_rows: list[dict[str, Any]] = []
    for key, m in by_inst.items():
        if policy_strategy not in m or baseline not in m:
            continue
        p = m[policy_strategy]
        b = m[baseline]
        tp = max(1.0, float(p.get("total_tokens_generated") or 1))
        tb = max(1.0, float(b.get("total_tokens_generated") or 1))
        paired_rows.append(
            {
                "instance_key": key,
                "succ_diff": float(bool(p.get("task_success"))) - float(bool(b.get("task_success"))),
                "log_tok_diff": math.log(tb) - math.log(tp),
            }
        )
    if not paired_rows:
        return {"n_pairs": 0, "delta": delta}

    mean_succ = sum(r["succ_diff"] for r in paired_rows) / len(paired_rows)
    mean_log = sum(r["log_tok_diff"] for r in paired_rows) / len(paired_rows)
    succ_boot = cluster_bootstrap(
        paired_rows, lambda rs: sum(r["succ_diff"] for r in rs) / len(rs), n_boot=n_boot, seed=seed
    )
    log_boot = cluster_bootstrap(
        paired_rows, lambda rs: sum(r["log_tok_diff"] for r in rs) / len(rs), n_boot=n_boot, seed=seed
    )
    succ_ci_low = succ_boot["ci_low"]
    log_ci_low = log_boot["ci_low"]
    return {
        "n_pairs": len(paired_rows),
        "delta": delta,
        "mean_success_diff": mean_succ,
        "mean_log_token_diff": mean_log,
        "success_ci_low": succ_ci_low,
        "success_ci_high": succ_boot["ci_high"],
        "log_token_ci_low": log_ci_low,
        "log_token_ci_high": log_boot["ci_high"],
        "non_inferiority_holds": succ_ci_low is not None and succ_ci_low > -delta,
        "token_superiority_holds": log_ci_low is not None and log_ci_low > 0,
    }


def holm(pvals_or_bounds: list[float], family: str = "") -> list[dict[str, Any]]:
    """Holm step-down adjustment (family A–C).

    Adjusted p-value at rank i is ``max(1, ..., i)`` of ``min(1, raw_(rank) * (m - rank))`` --
    the running-maximum ("enforce monotonicity") step is required by Holm (1979) so that
    adjusted p-values never decrease with increasing raw p-value; omitting it (as a prior
    version of this function did) can silently understate the adjustment for a p-value that
    is only slightly larger than the next-smaller one, e.g. raw=[0.01, 0.011, 0.5] with m=3
    must adjust to [0.03, 0.03, 0.5], not [0.03, 0.022, 0.5] (verified against
    ``statsmodels.stats.multitest.multipletests(method="holm")``).
    """
    m = len(pvals_or_bounds)
    order = sorted(range(m), key=lambda i: pvals_or_bounds[i])
    out: list[dict[str, Any]] = [{}] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        step_adj = min(1.0, pvals_or_bounds[idx] * (m - rank))
        running_max = max(running_max, step_adj)
        out[idx] = {
            "index": idx,
            "raw": pvals_or_bounds[idx],
            "adjusted": running_max,
            "family": family,
        }
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


def one_sided_wald_pvalue(coef: float, p_two_sided: float, *, direction: int = -1) -> float:
    """One-sided p-value for a directional GEE/Wald coefficient test (H3's interaction term).

    A two-sided Wald p-value splits evenly across both tails under the standard symmetric-
    statistic assumption; halving it gives the one-sided p-value when the estimate points the
    hypothesized way (``direction=-1`` for "coefficient < 0", the H3 degradation direction), and
    the complement (``1 - p_two_sided/2``) when it points the wrong way -- a coefficient in the
    wrong direction can never support a directional hypothesis, however small its two-sided p.
    """
    half = p_two_sided / 2.0
    points_right_way = coef < 0 if direction < 0 else coef > 0
    return half if points_right_way else (1.0 - half)


def fit_h3_model(
    df: Any,
    *,
    signal: str = "tle",
    domain: str | None = None,
) -> dict[str, Any]:
    """
    GEE: y ~ z_signal * position_norm with exchangeable instance clustering.

    The signal is z-standardized (mean 0, SD 1) *within each compute_stage* (C0/C1/C2), not
    pooled across stages -- raw TLE/VC scales are not commensurable across stages (different
    decoding temperatures and reasoning-token budgets shift each signal's scale independently
    of the construct being measured), mirroring the allocator's stage-wise ECDF normalization.
    See docs/adrs.md ADR-006 for the full argument (incl. why this doesn't affect H3
    significance, only interpretability) and the P0-5 entry in
    ../metacog-thesis/notes/revision_audit_2026-07.md.

    Fallback: returns error dict if statsmodels GEE fails to converge.
    """
    rows = _as_rows(df)
    if domain is not None:
        rows = [r for r in rows if str(r.get("domain")) == domain]
    y: list[int] = []
    z: list[float] = []
    pos: list[float] = []
    groups: list[str] = []
    stages: list[str] = []
    for r in rows:
        if r.get("y_optimal") is None:
            continue
        stage = r.get("compute_stage")
        if not stage:
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
        stages.append(str(stage))
    if len(y) < 20 or len(set(groups)) < 3:
        return {"converged": False, "note": "insufficient data"}
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial

        frame = pd.DataFrame(
            {"y": y, "z": z, "position_norm": pos, "g": groups, "stage": stages}
        )
        stage_mean = frame.groupby("stage")["z"].transform("mean")
        stage_std = frame.groupby("stage")["z"].transform("std")
        if (stage_std == 0).any() or stage_std.isna().any():
            return {
                "converged": False,
                "note": "zero- or undefined-variance signal within a compute_stage group",
                "signal": signal,
                "domain": domain,
            }
        frame["z_c"] = (frame["z"] - stage_mean) / stage_std
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
