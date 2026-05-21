"""
Pilot step execution helper functions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.pilot.artifacts import load_episode_jsons, load_json_optional


def prepare_feasibility_inputs(
    output_dir: Path,
    *,
    test1: dict[str, Any],
    test2: dict[str, Any],
    test3: dict[str, Any],
    episodes: list[dict[str, Any]],
    toh: dict[str, Any],
    sanity: dict[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    """
    Merge in-session results with prior JSON artifacts so partial --only runs can compute feasibility.
    """
    t1 = test1 or (load_json_optional(output_dir / "pilot_test1_inference.json") or {})
    t2 = test2 or (load_json_optional(output_dir / "pilot_test2_tle.json") or {})
    t3 = test3 or (load_json_optional(output_dir / "pilot_test3_vc.json") or {})
    eps = episodes if episodes else load_episode_jsons(output_dir, "ep_textworld_*.json")
    th = toh or (load_json_optional(output_dir / "pilot_test5_toh.json") or {})
    san = sanity if sanity is not None else load_json_optional(output_dir / "pilot_sanity.json")
    toh_eps = load_episode_jsons(output_dir, "ep_tower_of_hanoi_*.json")
    return t1, t2, t3, eps, th, san, toh_eps


def episode_vc_tle_rates(
    textworld_episodes: list[dict[str, Any]],
    toh_episodes: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """
    Fraction of env steps with non-null VC and non-null TLE across all pilot episodes.
    """
    all_eps = list(textworld_episodes) + list(toh_episodes)
    total_vc_steps = 0
    nonnull_vc_steps = 0
    total_tle_steps = 0
    nonnull_tle_steps = 0
    for ep in all_eps:
        for value in ep.get("vc_per_step") or []:
            total_vc_steps += 1
            if value is not None:
                nonnull_vc_steps += 1
        for value in ep.get("tle_per_step") or []:
            total_tle_steps += 1
            if value is not None:
                nonnull_tle_steps += 1
    vc_rate = (nonnull_vc_steps / total_vc_steps) if total_vc_steps else None
    tle_rate = (nonnull_tle_steps / total_tle_steps) if total_tle_steps else None
    return vc_rate, tle_rate


def run_mock_inference_speed_benchmark(
    num_prompts: int, tokens_per_call: int = 200
) -> dict[str, Any]:
    """Simulate a benchmark batch and return latency/tokens-per-second statistics."""
    latencies: list[float] = []
    total_tokens = 0
    for _ in range(int(num_prompts)):
        t0 = time.perf_counter()
        time.sleep(0.001)
        total_tokens += int(tokens_per_call)
        latencies.append(time.perf_counter() - t0)
    elapsed = sum(latencies)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    variance = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) if latencies else 0.0
    std_lat = variance**0.5
    return {
        "tokens_per_sec": float(tokens_per_sec),
        "latency_mean": float(mean_lat),
        "latency_std": float(std_lat),
        "total_tokens": int(total_tokens),
        "num_prompts": int(num_prompts),
    }
