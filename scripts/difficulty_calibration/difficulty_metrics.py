"""Gate D calibration metrics — Cap derivation and success@Cap (no signal analysis)."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

SUCCESS_CORRIDOR = (0.30, 0.50)
PLAUSIBLE_BAND = (0.15, 0.70)
LENGTH_GUIDANCE = (8, 15)


def extract_win_step(result: dict[str, Any]) -> int | None:
    """Step count at episode win; None if the episode did not succeed."""
    if not bool(result.get("task_success")):
        return None
    length = int(result.get("episode_length_steps", result.get("steps", 0)))
    step_results = result.get("step_correctness") or result.get("step_results") or []
    if isinstance(step_results, list):
        for rec in step_results:
            if isinstance(rec, dict) and rec.get("won"):
                idx = int(rec.get("step_index", length - 1))
                return idx + 1
    return length if length > 0 else None


def episode_record(result: dict[str, Any], *, obs_ceiling: int) -> dict[str, Any]:
    length = int(result.get("episode_length_steps", result.get("steps", 0)))
    task_success = bool(result.get("task_success"))
    win_step = extract_win_step(result)
    truncated = length >= obs_ceiling
    return {
        "task_success": task_success,
        "win_step": win_step,
        "episode_length_steps": length,
        "truncated": truncated,
    }


def success_rate_at_cap(episodes: Sequence[dict[str, Any]], cap: int) -> float:
    n = len(episodes)
    if n == 0:
        return 0.0
    hits = sum(
        1
        for ep in episodes
        if ep.get("task_success") and ep.get("win_step") is not None and int(ep["win_step"]) <= cap
    )
    return hits / n


def success_rate_at_obs(episodes: Sequence[dict[str, Any]]) -> float:
    n = len(episodes)
    if n == 0:
        return 0.0
    return sum(1 for ep in episodes if ep.get("task_success")) / n


def _percentile(values: Sequence[float | int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def derive_production_cap(
    win_steps: Sequence[int],
    *,
    margin: int = 2,
) -> int | None:
    """Cap = ceil(p90 of win steps) + margin, at least ceil(median)."""
    if not win_steps:
        return None
    p90 = _percentile(win_steps, 90)
    med = statistics.median(win_steps)
    return int(max(math.ceil(p90) + margin, math.ceil(med)))


def aggregate_length_stats(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    win_steps = [
        int(ep["win_step"])
        for ep in episodes
        if ep.get("task_success") and ep.get("win_step") is not None
    ]
    all_lengths = [int(ep["episode_length_steps"]) for ep in episodes]
    truncated = sum(1 for ep in episodes if ep.get("truncated"))
    n = len(episodes)
    out: dict[str, float | int | None] = {
        "num_instances": n,
        "truncation_rate": (truncated / n) if n else 0.0,
        "median_win_step_success": None,
        "p90_win_step_success": None,
        "mean_episode_length_all": (sum(all_lengths) / n) if n else 0.0,
    }
    if win_steps:
        out["median_win_step_success"] = float(statistics.median(win_steps))
        out["p90_win_step_success"] = float(_percentile(win_steps, 90))
        out["mean_episode_length_success"] = float(sum(win_steps) / len(win_steps))
    return out


def collect_win_steps_from_cells(
    cell_results: Sequence[dict[str, Any]],
    *,
    use_plausible_band: bool = True,
    success_key: str = "success_rate_at_obs",
) -> list[int]:
    steps: list[int] = []
    lo, hi = PLAUSIBLE_BAND
    for cell in cell_results:
        rate = float(cell.get(success_key, 0.0))
        if use_plausible_band and not (lo <= rate <= hi):
            continue
        for ep in cell.get("episodes", []):
            if ep.get("task_success") and ep.get("win_step") is not None:
                steps.append(int(ep["win_step"]))
    return steps


def cap_ceiling_warning(
    *,
    production_cap: int,
    obs_ceiling: int,
    truncation_rate: float,
    p90_win_step: float | None,
) -> list[str]:
    warnings: list[str] = []
    if p90_win_step is not None and p90_win_step >= obs_ceiling - 3:
        warnings.append(
            f"p90_win_step={p90_win_step:.1f} near obs_ceiling={obs_ceiling}; "
            "right tail may be clipped — rerun sweep at higher observation ceiling."
        )
    if production_cap >= obs_ceiling - 1:
        warnings.append(
            f"production_cap={production_cap} ≈ obs_ceiling={obs_ceiling}; "
            "observation ceiling did not extend meaningfully beyond production cap."
        )
    if truncation_rate > 0.05:
        warnings.append(
            f"truncation_rate={truncation_rate:.3f} > 0.05 at obs_ceiling={obs_ceiling}; "
            "p90 may be downward-biased."
        )
    return warnings


def json_combo_key(combo: dict[str, Any]) -> str:
    return (
        f"r{combo.get('num_rooms')}_i{combo.get('num_ingredients')}_"
        f"{combo.get('operations', '')}_o{int(bool(combo.get('open')))}"
    )


def select_corridor_candidates(
    cell_results: Sequence[dict[str, Any]],
    *,
    success_key: str = "success_rate_at_cap",
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Pick up to 3 cells in the success corridor, spread by success rate."""
    lo, hi = SUCCESS_CORRIDOR
    in_corridor = [c for c in cell_results if lo <= float(c.get(success_key, 0.0)) <= hi]
    if not in_corridor:
        ranked = sorted(
            cell_results,
            key=lambda c: abs(float(c.get(success_key, 0.0)) - 0.40),
        )
        return ranked[:max_candidates]

    by_success = sorted(in_corridor, key=lambda c: float(c.get(success_key, 0.0)))
    picks: list[dict[str, Any]] = []
    if by_success:
        picks.append(by_success[len(by_success) // 2])
    if len(by_success) >= 2:
        picks.append(by_success[0])
        picks.append(by_success[-1])
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for pick in picks:
        key = json_combo_key(pick.get("combo", {}))
        if key not in seen:
            seen.add(key)
            unique.append(pick)
    return unique[:max_candidates]


def score_success_first(
    success_rate_at_cap: float,
    median_win_step_success: float | None,
) -> float:
    """Lower is better. Success corridor primary; length guidance secondary."""
    success_center = 0.40
    score = abs(success_rate_at_cap - success_center)
    lo, hi = SUCCESS_CORRIDOR
    if lo <= success_rate_at_cap <= hi and median_win_step_success is not None:
        length_center = 11.5
        score += 0.05 * abs(median_win_step_success - length_center)
    return score
