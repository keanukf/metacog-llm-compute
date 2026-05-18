"""
Allocation- and efficiency-oriented metrics (Phase 2 and beyond).

These utilities operate on episode/step rows produced by `src.analysis.datasets`.
They are intentionally dependency-light (no stats libs required).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(sum(values) / n), "ci_lo": float(lo), "ci_hi": float(hi)}


def stage_mix(
    steps_rows: Iterable[dict[str, Any]], *, stage_key: str = "compute_stage"
) -> dict[str, Any]:
    """
    Compute fraction of steps spent in each stage (C0/C1/C2) overall and by domain/strategy.
    """
    total = 0
    counts: dict[str, int] = defaultdict(int)
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_strategy: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        st = r.get(stage_key) or r.get("stage_per_step") or r.get("stage_at_step")
        if st is None:
            continue
        st = str(st)
        total += 1
        counts[st] += 1
        d = str(r.get("domain", "unknown"))
        by_domain[d][st] += 1
        s = str(r.get("strategy", r.get("compute_stage", "unknown")))
        by_strategy[s][st] += 1

    def _frac_map(m: dict[str, int]) -> dict[str, float]:
        denom = sum(m.values())
        return {k: (v / denom if denom else 0.0) for k, v in sorted(m.items())}

    return {
        "n_steps": int(total),
        "overall": _frac_map(counts),
        "by_domain": {k: _frac_map(v) for k, v in sorted(by_domain.items())},
        "by_strategy": {k: _frac_map(v) for k, v in sorted(by_strategy.items())},
    }


def run_health(
    episodes_rows: Iterable[dict[str, Any]], steps_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """
    High-signal diagnostics for whether the run contains the information needed for analysis.
    """
    n_eps = 0
    n_eps_steps_synth = 0
    n_eps_with_logprobs = 0
    n_eps_with_vc_sidecar = 0
    n_eps_with_trace = 0
    for ep in episodes_rows:
        if not isinstance(ep, dict):
            continue
        n_eps += 1
        if bool(ep.get("_steps_detail_synthesized")):
            n_eps_steps_synth += 1
        if ep.get("logprobs_json_path") or ep.get("logprobs_csv_path"):
            n_eps_with_logprobs += 1
        if ep.get("vc_json_path") or ep.get("vc_csv_path"):
            n_eps_with_vc_sidecar += 1
        if ep.get("trace_jsonl_path"):
            n_eps_with_trace += 1

    n_steps = 0
    n_missing_vc = 0
    n_missing_tle = 0
    n_missing_label = 0
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        n_steps += 1
        if r.get("vc") is None:
            n_missing_vc += 1
        if r.get("tle_mean_entropy") is None:
            n_missing_tle += 1
        if r.get("step_correct_optimal") is None:
            n_missing_label += 1

    return {
        "episodes": int(n_eps),
        "steps": int(n_steps),
        "episodes_steps_detail_synthesized_rate": (n_eps_steps_synth / n_eps) if n_eps else 0.0,
        "episodes_with_logprobs_sidecar_rate": (n_eps_with_logprobs / n_eps) if n_eps else 0.0,
        "episodes_with_vc_sidecar_rate": (n_eps_with_vc_sidecar / n_eps) if n_eps else 0.0,
        "episodes_with_trace_rate": (n_eps_with_trace / n_eps) if n_eps else 0.0,
        "missing_vc_rate": (n_missing_vc / n_steps) if n_steps else 0.0,
        "missing_tle_rate": (n_missing_tle / n_steps) if n_steps else 0.0,
        "missing_step_label_rate": (n_missing_label / n_steps) if n_steps else 0.0,
    }


def efficiency_summary(
    episodes_rows: Iterable[dict[str, Any]], *, group_key: str | None = None
) -> list[dict[str, Any]]:
    """
    Group episodes and summarize success_rate, mean costs, and efficiency = success_rate / mean_cost.
    """
    rows = [ep for ep in episodes_rows if isinstance(ep, dict)]
    if not rows:
        return []

    if group_key is None:
        group_key = "strategy" if any("strategy" in ep for ep in rows) else "compute_stage"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in rows:
        g = str(ep.get(group_key, "unknown"))
        groups[g].append(ep)

    out: list[dict[str, Any]] = []
    for g, eps in sorted(groups.items()):
        n = len(eps)
        succ = sum(1 for e in eps if bool(e.get("task_success")))
        success_rate = succ / n if n else 0.0
        costs = [float(e.get("normalized_compute_cost") or 0.0) for e in eps]
        mean_cost = sum(costs) / n if n else 0.0
        eff = (success_rate / mean_cost) if mean_cost > 0 else None
        out.append(
            {
                group_key: g,
                "episodes": int(n),
                "success_rate": float(success_rate),
                "mean_normalized_compute_cost": float(mean_cost),
                "efficiency": float(eff) if eff is not None else None,
                "success_rate_ci": _bootstrap_ci(
                    [1.0 if bool(e.get("task_success")) else 0.0 for e in eps]
                ),
                "cost_ci": _bootstrap_ci(costs) if any(costs) else None,
            }
        )
    return out


def regret_vs_baselines(
    episodes_rows: Iterable[dict[str, Any]],
    *,
    domain_key: str = "domain",
    strategy_key: str = "strategy",
    success_key: str = "task_success",
    cost_key: str = "normalized_compute_cost",
    baseline_strategies: tuple[str, ...] = ("always_c0", "always_c2"),
) -> dict[str, Any]:
    """
    Compute simple regret-style comparisons per domain:
    compare each strategy's (success_rate, mean_cost, efficiency) to baselines.
    """
    rows = [ep for ep in episodes_rows if isinstance(ep, dict)]
    if not rows:
        return {}

    # Aggregate per domain/strategy
    acc: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in rows:
        d = str(ep.get(domain_key, "unknown"))
        s = str(ep.get(strategy_key, ep.get("compute_stage", "unknown")))
        acc[(d, s)].append(ep)

    per: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (d, s), eps in sorted(acc.items()):
        n = len(eps)
        succ = sum(1 for e in eps if bool(e.get(success_key)))
        success_rate = succ / n if n else 0.0
        costs = [float(e.get(cost_key) or 0.0) for e in eps]
        mean_cost = sum(costs) / n if n else 0.0
        eff = (success_rate / mean_cost) if mean_cost > 0 else 0.0
        per[d][s] = {
            "success_rate": float(success_rate),
            "mean_cost": float(mean_cost),
            "efficiency": float(eff),
        }

    out: dict[str, Any] = {}
    for d, strat_map in per.items():
        base = {b: strat_map.get(b) for b in baseline_strategies if b in strat_map}
        out_d: dict[str, Any] = {"baselines": base, "strategies": {}}
        for s, m in strat_map.items():
            entry = dict(m)
            for b, bm in base.items():
                if bm is None:
                    continue
                entry[f"delta_eff_vs_{b}"] = float(m["efficiency"] - bm["efficiency"])
                entry[f"delta_cost_vs_{b}"] = float(m["mean_cost"] - bm["mean_cost"])
                entry[f"delta_success_vs_{b}"] = float(m["success_rate"] - bm["success_rate"])
            out_d["strategies"][s] = entry
        out[d] = out_d
    return out
