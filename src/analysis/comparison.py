"""
Mixed-effects models for Phase 2 strategy comparison.

This module provides lightweight, dependency-minimal comparisons:
- bootstrap confidence intervals
- permutation tests for metric differences

Mixed-effects models can be added later; for now `run_mixed_effects` remains as a stub.
"""
from __future__ import annotations

import random
from typing import Any


def run_mixed_effects(
    data_path: str | None = None,
    formula: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run mixed-effects model (e.g. strategy, domain, instance random effect).
    Stub: returns empty dict until implemented.
    """
    return {}


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _k in range(n):
            s += float(values[rng.randrange(n)])
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(sum(values) / n), "ci_lo": float(lo), "ci_hi": float(hi)}


def bootstrap_diff_in_means_ci(
    a: list[float],
    b: list[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float] | None:
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    na = len(a)
    nb = len(b)
    diffs: list[float] = []
    for _ in range(n_boot):
        ma = sum(float(a[rng.randrange(na)]) for _k in range(na)) / na
        mb = sum(float(b[rng.randrange(nb)]) for _k in range(nb)) / nb
        diffs.append(ma - mb)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[int((1 - alpha / 2) * n_boot) - 1]
    return {"diff_mean": float((sum(a) / na) - (sum(b) / nb)), "ci_lo": float(lo), "ci_hi": float(hi)}


def permutation_test_diff_in_means(
    a: list[float],
    b: list[float],
    *,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict[str, float] | None:
    """
    Two-sided permutation test for difference in means.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    obs = (sum(a) / len(a)) - (sum(b) / len(b))
    pooled = list(a) + list(b)
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        ma = sum(pooled[:na]) / na
        mb = sum(pooled[na:]) / len(b)
        diff = ma - mb
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"diff_mean": float(obs), "p_value": float(p)}


def compare_strategies_phase2(
    episodes: list[dict[str, Any]],
    *,
    metric: str = "task_success",
    group_key: str = "strategy",
    domain_key: str = "domain",
    baseline: str = "always_c2",
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Compare strategies within each domain against a baseline using bootstrap CIs.

    metric:
      - 'task_success' (binary -> mean success rate)
      - 'normalized_compute_cost' (float -> mean cost)
      - 'efficiency' (derived: success_rate / mean_cost; computed per group)
    """
    by_dom: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        d = str(ep.get(domain_key, "unknown"))
        by_dom.setdefault(d, []).append(ep)

    out: dict[str, Any] = {"metric": metric, "baseline": baseline, "by_domain": {}}
    for d, eps in sorted(by_dom.items()):
        groups: dict[str, list[dict[str, Any]]] = {}
        for ep in eps:
            g = str(ep.get(group_key, "unknown"))
            groups.setdefault(g, []).append(ep)

        dom_out: dict[str, Any] = {"groups": {}, "diff_vs_baseline": {}}
        # group summaries
        for g, geps in sorted(groups.items()):
            if metric == "task_success":
                vals = [1.0 if bool(e.get("task_success")) else 0.0 for e in geps]
                dom_out["groups"][g] = {"mean_ci": bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)}
            elif metric == "normalized_compute_cost":
                vals = [float(e.get("normalized_compute_cost") or 0.0) for e in geps]
                dom_out["groups"][g] = {"mean_ci": bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)}
            elif metric == "efficiency":
                succ = [1.0 if bool(e.get("task_success")) else 0.0 for e in geps]
                costs = [float(e.get("normalized_compute_cost") or 0.0) for e in geps]
                ms = (sum(succ) / len(succ)) if succ else 0.0
                mc = (sum(costs) / len(costs)) if costs else 0.0
                eff = (ms / mc) if mc > 0 else None
                dom_out["groups"][g] = {"efficiency": float(eff) if eff is not None else None, "success_rate": ms, "mean_cost": mc}
            else:
                dom_out["groups"][g] = {"error": f"unknown metric: {metric}"}

        # baseline diffs (only for mean-based metrics)
        if baseline in groups and metric in {"task_success", "normalized_compute_cost"}:
            if metric == "task_success":
                base_vals = [1.0 if bool(e.get("task_success")) else 0.0 for e in groups[baseline]]
            else:
                base_vals = [float(e.get("normalized_compute_cost") or 0.0) for e in groups[baseline]]
            for g, geps in sorted(groups.items()):
                if g == baseline:
                    continue
                if metric == "task_success":
                    vals = [1.0 if bool(e.get("task_success")) else 0.0 for e in geps]
                else:
                    vals = [float(e.get("normalized_compute_cost") or 0.0) for e in geps]
                dom_out["diff_vs_baseline"][g] = {
                    "bootstrap_diff_ci": bootstrap_diff_in_means_ci(vals, base_vals, n_boot=n_boot, seed=seed),
                    "perm_test": permutation_test_diff_in_means(vals, base_vals, seed=seed),
                }

        out["by_domain"][d] = dom_out
    return out
