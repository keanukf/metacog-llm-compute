#!/usr/bin/env python3
"""
Gate D feasibility diagnostic: r3_i1_take-only, C0/C1/C2, no signal analysis.

Reports per-stage success rate, median episode length, truncation rate only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_d_metrics import aggregate_length_stats, episode_record, success_rate_at_obs
from scripts.sweep_textworld_difficulty import (
    _create_model,
    _generate_combo_games,
    _instance_seed,
    _load_merged_config,
)


def _median_length(episodes: list[dict[str, Any]]) -> float | None:
    lengths = [
        int(ep["episode_length_steps"])
        for ep in episodes
        if ep.get("episode_length_steps") is not None
    ]
    if not lengths:
        return None
    return float(statistics.median(lengths))


def _run_stage(
    *,
    game_files: list[Path],
    config: dict[str, Any],
    use_real: bool,
    stage: str,
    obs_ceiling: int,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    model = _create_model(config, use_real)
    step_cfg = resolve_step_fn_kwargs(config, "textworld")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    step_fn = get_step_fn(stage, **step_cfg)
    episodes: list[dict[str, Any]] = []
    try:
        for p in game_files:
            env = TextWorldEnv(game_file=str(p), max_steps=obs_ceiling)
            result = run_episode(
                env, model, stage, step_fn=step_fn, max_steps=obs_ceiling, **history_cfg
            )
            episodes.append(episode_record(result, obs_ceiling=obs_ceiling))
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()

    stats = aggregate_length_stats(episodes)
    return {
        "compute_stage": stage,
        "n_episodes": len(episodes),
        "success_rate": success_rate_at_obs(episodes),
        "median_episode_length": _median_length(episodes),
        "truncation_rate": float(stats.get("truncation_rate") or 0.0),
        "mean_episode_length_all": float(stats.get("mean_episode_length_all") or 0.0),
        "successes": sum(1 for e in episodes if e.get("task_success")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D TW feasibility diagnostic.")
    parser.add_argument("--config", default="configs/dev/gate_d_diagnostic.yaml")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=12)
    parser.add_argument("--obs-ceiling", type=int, default=35)
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_diagnostic/feasibility",
    )
    args = parser.parse_args()

    import shutil
    import tempfile

    config = _load_merged_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_idx = 1  # r3_i1_take-only first cell
    combo_seed = _instance_seed(int(args.seed), combo_idx)
    work = Path(tempfile.mkdtemp(prefix="gate_d_feas_"))
    try:
        games = _generate_combo_games(
            base_dir=work / "r3_i1_take-only",
            num_instances=int(args.num_instances),
            seed=combo_seed,
            num_rooms=3,
            num_ingredients=1,
            cut=False,
            cook=False,
            open_=False,
        )
        stages = ["C0", "C1", "C2"]
        by_stage: dict[str, Any] = {}
        for stage in stages:
            print(f"Running {stage} x{len(games)} (obs_ceiling={args.obs_ceiling})...")
            by_stage[stage] = _run_stage(
                game_files=games,
                config=config,
                use_real=bool(args.real),
                stage=stage,
                obs_ceiling=int(args.obs_ceiling),
            )
            row = by_stage[stage]
            print(
                f"  {stage}: success={row['success_rate']:.1%} "
                f"({row['successes']}/{row['n_episodes']}) "
                f"median_len={row['median_episode_length']} trunc={row['truncation_rate']:.1%}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report = {
        "combo": "r3_i1_take-only",
        "seed": int(args.seed),
        "num_instances": int(args.num_instances),
        "obs_ceiling": int(args.obs_ceiling),
        "config": str(args.config),
        "vocabulary_fix": "experiment_core.yaml static templates (2026-07-15)",
        "by_stage": by_stage,
        "interpretation_rule": (
            "C1/C2 >> C0 (>30% vs floor) => domain viable, C0-specific; "
            "all stages <~10% => domain not tractable for Qwen3-8B in current form"
        ),
    }
    out_path = out_dir / "feasibility_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
