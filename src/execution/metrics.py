"""Execution-layer run metrics (descriptive + throughput)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def _episode_mean_tle(ep: dict) -> float | None:
    vals: list[float] = []
    for sd in ep.get("steps_detail") or []:
        tle = sd.get("tle") if isinstance(sd, dict) else None
        if isinstance(tle, dict):
            v = tle.get("mean_entropy")
            if isinstance(v, (int, float)):
                vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_execution_metrics(
    *,
    checkpoint_dir: Path,
    total_wall_time_s: float,
    total_tokens_generated: int,
    max_in_flight_observed: int,
    trajectory_divergence_rate: float | None = None,
) -> dict[str, Any]:
    """Aggregate throughput and per-stage token stats from checkpoint episodes."""
    episodes: list[dict] = []
    for p in sorted(checkpoint_dir.glob("ep_*.json")):
        try:
            import json

            with open(p, encoding="utf-8") as f:
                episodes.append(json.load(f))
        except Exception:
            continue

    by_stage: dict[str, list[int]] = defaultdict(list)
    for ep in episodes:
        stage = str(ep.get("compute_stage") or ep.get("strategy") or "unknown")
        tok = int(ep.get("total_tokens_generated") or ep.get("tokens") or 0)
        by_stage[stage].append(tok)

    avg_tokens_by_stage = {
        stage: (sum(toks) / len(toks) if toks else 0.0) for stage, toks in by_stage.items()
    }
    tokens_per_sec = (
        float(total_tokens_generated) / float(total_wall_time_s) if total_wall_time_s > 0 else 0.0
    )
    out: dict[str, Any] = {
        "tokens_per_sec": tokens_per_sec,
        "avg_tokens_by_stage": avg_tokens_by_stage,
        "max_in_flight_observed": int(max_in_flight_observed),
    }
    if trajectory_divergence_rate is not None:
        out["trajectory_divergence_rate"] = float(trajectory_divergence_rate)
    return out


def trajectory_divergence_rate(
    solo_episodes: list[dict[str, Any]],
    parallel_episodes: list[dict[str, Any]],
) -> float:
    """
    Descriptive: fraction of episode pairs with differing ``task_success`` or step count.

    Not used for GO/NO-GO.
    """
    solo_by_id = {str(e.get("episode_id")): e for e in solo_episodes if e.get("episode_id")}
    par_by_id = {str(e.get("episode_id")): e for e in parallel_episodes if e.get("episode_id")}
    common = set(solo_by_id) & set(par_by_id)
    if not common:
        return 0.0
    diverged = 0
    for eid in common:
        a, b = solo_by_id[eid], par_by_id[eid]
        if bool(a.get("task_success")) != bool(b.get("task_success")):
            diverged += 1
            continue
        if int(a.get("steps") or 0) != int(b.get("steps") or 0):
            diverged += 1
    return diverged / len(common)
