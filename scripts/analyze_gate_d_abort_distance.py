#!/usr/bin/env python3
"""
Recover quest Restdistanz at episode end for Gate D sweep v1.

The original sweep did not persist per-episode step records. This script
replays the same grid / seeds / obs ceiling read from sweep_results.json and
records optimal_moves_remaining (quest Restdistanz) at the final step.

Not a calibration rerun — same 216 experimental units, telemetry recovery only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_d_metrics import episode_record
from scripts.sweep_textworld_difficulty import (
    _create_model,
    _generate_combo_games,
    _instance_seed,
    _load_merged_config,
)


def _quest_distance_at_end(result: dict[str, Any]) -> int | None:
    steps = result.get("step_correctness") or []
    if not steps:
        return None
    last = steps[-1]
    if not isinstance(last, dict):
        return None
    dist = last.get("optimal_moves_remaining")
    if dist is None:
        dist = last.get("quest_distance_after")
    if dist is None:
        return None
    try:
        return int(dist)
    except (TypeError, ValueError):
        return None


def _quest_distance_at_start(result: dict[str, Any]) -> int | None:
    steps = result.get("step_correctness") or []
    if not steps:
        return None
    first = steps[0]
    if not isinstance(first, dict):
        return None
    dist = first.get("quest_distance_before")
    if dist is None:
        dist = first.get("optimal_moves_remaining")
    if dist is None:
        return None
    try:
        return int(dist)
    except (TypeError, ValueError):
        return None


def _summarize_distances(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return float(ordered[0])
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(ordered[f])
        return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

    return {
        "n": n,
        "median": float(statistics.median(ordered)),
        "mean": float(sum(ordered) / n),
        "p90": float(pct(0.9)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _ops_from_label(label: str) -> dict[str, bool]:
    if label == "take+cut+cook":
        return {"cut": True, "cook": True}
    if label == "take+cook":
        return {"cut": False, "cook": True}
    return {"cut": False, "cook": False}


def _run_replay(
    *,
    sweep: dict[str, Any],
    config: dict[str, Any],
    use_real: bool,
    obs_ceiling: int,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv

    seed = int(sweep.get("seed", 42))
    instances_per_combo = int(sweep.get("instances_per_combo", 8))
    grid = sweep.get("grid") or {}
    rooms_grid = grid.get("rooms") or [3, 5, 7]
    ingredients_grid = grid.get("ingredients") or [1, 2, 3]
    open_ = bool(grid.get("open", False))
    ops_labels = grid.get("operations") or ["take-only", "take+cook", "take+cut+cook"]

    model = _create_model(config, use_real)
    c0 = get_step_fn("C0")

    import shutil
    import tempfile

    work_root = Path(tempfile.mkdtemp(prefix="gate_d_abort_"))
    cells: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    combo_idx = 0
    try:
        for rooms in rooms_grid:
            for ingredients in ingredients_grid:
                for label in ops_labels:
                    combo_idx += 1
                    ops = _ops_from_label(str(label))
                    combo_name = f"r{rooms}_i{ingredients}_{str(label).replace('+', '_')}"
                    combo_dir = work_root / combo_name
                    combo_seed = _instance_seed(seed, combo_idx)
                    games = _generate_combo_games(
                        base_dir=combo_dir,
                        num_instances=instances_per_combo,
                        seed=combo_seed,
                        num_rooms=int(rooms),
                        num_ingredients=int(ingredients),
                        cut=bool(ops["cut"]),
                        cook=bool(ops["cook"]),
                        open_=open_,
                    )
                    cell_eps: list[dict[str, Any]] = []
                    for game_path in games:
                        env = TextWorldEnv(game_file=str(game_path), max_steps=obs_ceiling)
                        result = run_episode(env, model, "C0", step_fn=c0, max_steps=obs_ceiling)
                        ep = episode_record(result, obs_ceiling=obs_ceiling)
                        end_dist = _quest_distance_at_end(result)
                        start_dist = _quest_distance_at_start(result)
                        row = {
                            **ep,
                            "combo": combo_name,
                            "quest_distance_at_start": start_dist,
                            "quest_distance_at_end": end_dist,
                            "aborted_without_success": bool(
                                not ep["task_success"] and ep["truncated"]
                            ),
                        }
                        cell_eps.append(row)
                        all_episodes.append(row)

                    end_dists = [
                        int(e["quest_distance_at_end"])
                        for e in cell_eps
                        if e.get("quest_distance_at_end") is not None
                    ]
                    abort_dists = [
                        int(e["quest_distance_at_end"])
                        for e in cell_eps
                        if e.get("aborted_without_success")
                        and e.get("quest_distance_at_end") is not None
                    ]
                    cells.append(
                        {
                            "combo": {
                                "num_rooms": rooms,
                                "num_ingredients": ingredients,
                                "operations": label,
                                "open": open_,
                            },
                            "combo_name": combo_name,
                            "episodes": cell_eps,
                            "end_distance_all": _summarize_distances(end_dists),
                            "end_distance_aborted": _summarize_distances(abort_dists),
                            "histogram_aborted": dict(Counter(abort_dists)),
                        }
                    )
                    print(
                        f"[{combo_name}] aborted_n="
                        f"{sum(1 for e in cell_eps if e['aborted_without_success'])} "
                        f"end_dist_abort={_summarize_distances(abort_dists)}"
                    )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    aborted = [e for e in all_episodes if e.get("aborted_without_success")]
    successes = [e for e in all_episodes if e.get("task_success")]
    abort_end = [
        int(e["quest_distance_at_end"])
        for e in aborted
        if e.get("quest_distance_at_end") is not None
    ]
    all_end = [
        int(e["quest_distance_at_end"])
        for e in all_episodes
        if e.get("quest_distance_at_end") is not None
    ]

    near_abort = sum(1 for d in abort_end if d <= 3)
    far_abort = sum(1 for d in abort_end if d >= 8)

    easiest = next((c for c in cells if c["combo_name"] == "r3_i1_take-only"), None)

    return {
        "source_sweep": "sweep_results.json replay (same seeds/grid)",
        "obs_ceiling": obs_ceiling,
        "total_episodes": len(all_episodes),
        "aborted_without_success": len(aborted),
        "successes": len(successes),
        "global_end_distance_all": _summarize_distances(all_end),
        "global_end_distance_aborted": _summarize_distances(abort_end),
        "histogram_aborted": dict(Counter(abort_end)),
        "aborted_near_done_le3": near_abort,
        "aborted_far_ge8": far_abort,
        "fraction_aborted_near_done": (near_abort / len(abort_end) if abort_end else None),
        "fraction_aborted_far": (far_abort / len(abort_end) if abort_end else None),
        "easiest_cell_r3_i1_take_only": easiest,
        "cells": cells,
    }


def _interpret(report: dict[str, Any]) -> str:
    lines = ["## Gate D — Restdistanz bei Abbruch (Replay)\n"]
    g = report.get("global_end_distance_aborted") or {}
    lines.append(
        f"- Abgebrochene Episoden (Cap={report.get('obs_ceiling')}): "
        f"**n={report.get('aborted_without_success')}**"
    )
    if g.get("n"):
        lines.append(
            f"- Restdistanz am End-Step (abort): median **{g.get('median')}**, "
            f"p90 **{g.get('p90')}**, mean **{g.get('mean')}**, "
            f"min/max **{g.get('min')}** / **{g.get('max')}**"
        )
    frac_near = report.get("fraction_aborted_near_done")
    frac_far = report.get("fraction_aborted_far")
    if frac_near is not None:
        lines.append(f"- Anteil abort mit Restdistanz ≤3: **{frac_near:.1%}**")
    if frac_far is not None:
        lines.append(f"- Anteil abort mit Restdistanz ≥8: **{frac_far:.1%}**")

    easiest = report.get("easiest_cell_r3_i1_take_only") or {}
    ed = (easiest.get("end_distance_aborted") or {}) if easiest else {}
    if ed.get("n"):
        lines.append(
            f"- Leichtestes Raster (r3_i1_take-only) abort Restdistanz: "
            f"median **{ed.get('median')}**, p90 **{ed.get('p90')}**"
        )

    lines.append("\n### Lesart (Heuristik)\n")
    if frac_near is not None and frac_near >= 0.35 and (g.get("median") or 99) <= 5:
        lines.append(
            "- Viele Abbrüche **nah am Ziel** → nächster Lauf: **höhere Decke** "
            "(eher **50**, nicht knapp über 25)."
        )
    if frac_far is not None and frac_far >= 0.35:
        lines.append(
            "- Viele Abbrüche **weit vom Ziel** → zusätzlich **leichtere Instanzen** nötig."
        )
    if (report.get("successes") or 0) <= 4 and (frac_far or 0) > 0.5:
        lines.append(
            "- Bei ~1.9% Erfolg und hoher Fern-Restdistanz: **beides** wahrscheinlich "
            "(Decke + leichteres Setup); r3_i1_take-only @ Decke 50 als untere Flanke testen."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Gate D sweep for abort Restdistanz.")
    parser.add_argument(
        "--sweep-results",
        default="data/results/gate_d_calibration/textworld_sweep/sweep_results.json",
    )
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/textworld_abort_distance",
    )
    args = parser.parse_args()

    sweep_path = REPO_ROOT / args.sweep_results
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    obs_ceiling = int(sweep.get("obs_ceiling", 25))
    config = _load_merged_config(REPO_ROOT / args.config)

    report = _run_replay(
        sweep=sweep,
        config=config,
        use_real=bool(args.real),
        obs_ceiling=obs_ceiling,
    )
    report["interpretation_md"] = _interpret(report)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "abort_distance_report.json"
    md_path = out_dir / "abort_distance_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report["interpretation_md"], encoding="utf-8")
    print(report["interpretation_md"])
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
