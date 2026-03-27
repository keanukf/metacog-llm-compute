#!/usr/bin/env python3
"""
Sweep TextWorld Cooking generation difficulty and evaluate C0 behavior.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_abs(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _instance_seed(master_seed: int, idx: int) -> int:
    rng = random.Random(master_seed)
    for _ in range(idx + 1):
        value = rng.randint(0, 2**31 - 1)
    return int(value)


def _ops_name(cut: bool, cook: bool) -> str:
    if cut and cook:
        return "take+cut+cook"
    if cook:
        return "take+cook"
    return "take-only"


def _generate_combo_games(
    *,
    base_dir: Path,
    num_instances: int,
    seed: int,
    num_rooms: int,
    num_ingredients: int,
    cut: bool,
    cook: bool,
    open_: bool,
) -> list[Path]:
    from scripts.generate_textworld_games import _run_generate_command, _write_sidecar

    base_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for i in range(num_instances):
        game_seed = _instance_seed(seed, i)
        game_file = base_dir / f"textworld_{i}.ulx"
        sidecar_file = base_dir / f"textworld_{i}.json"
        _run_generate_command(
            output_file=game_file,
            seed=game_seed,
            num_rooms=num_rooms,
            num_ingredients=num_ingredients,
            cut=cut,
            cook=cook,
            open_=open_,
        )
        _write_sidecar(
            sidecar_file=sidecar_file,
            game_file=game_file,
            instance_id=i,
            master_seed=seed,
            game_seed=game_seed,
            num_rooms=num_rooms,
            num_ingredients=num_ingredients,
            cut=cut,
            cook=cook,
            open_=open_,
        )
        files.append(game_file)
    return files


def _run_c0_batch(
    *,
    game_files: list[Path],
    config: dict[str, Any],
    use_real_model: bool,
    max_steps: int,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.experiment_env import create_experiment_model

    model = create_experiment_model(config, use_real_model)
    c0 = get_step_fn("C0")

    episode_lengths: list[int] = []
    successes = 0
    for p in game_files:
        env = TextWorldEnv(game_file=str(p), max_steps=max_steps)
        result = run_episode(env, model, "C0", step_fn=c0, max_steps=max_steps)
        length = int(result.get("episode_length_steps", result.get("steps", 0)))
        episode_lengths.append(length)
        if bool(result.get("task_success")):
            successes += 1
    n = len(game_files)
    dist = Counter(episode_lengths)
    return {
        "num_instances": n,
        "success_rate": (successes / n) if n else 0.0,
        "mean_episode_length": (sum(episode_lengths) / n) if n else 0.0,
        "step_count_distribution": {str(k): v for k, v in sorted(dist.items())},
        "episode_lengths": episode_lengths,
    }


def _score_candidate(success_rate: float, mean_episode_length: float) -> float:
    # Lower is better. Zero means exactly at center of target band.
    success_center = 0.40
    length_center = 11.5
    return abs(success_rate - success_center) + 0.05 * abs(mean_episode_length - length_center)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep TextWorld Cooking difficulty for C0 target window.")
    parser.add_argument("--config", default="configs/experiment_core.yaml", help="Experiment config path")
    parser.add_argument("--output-dir", default="data/tasks/textworld/sweeps", help="Where sweep artifacts are written")
    parser.add_argument("--instances-per-combo", type=int, default=6, help="Small batch size per parameter combination (5-10 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for reproducibility")
    parser.add_argument("--open", action="store_true", dest="open_", help="Enable open/close operations")
    parser.add_argument("--runtime-max-steps", type=int, default=25, help="Agent-loop cap (targeted 20-25)")
    parser.add_argument("--real", action="store_true", help="Use real model backend from config")
    parser.add_argument("--keep-games", action="store_true", help="Keep per-combo generated games")
    args = parser.parse_args()

    out_dir = _to_abs(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _to_abs(args.config)
    with open(cfg_path) as f:
        import yaml

        config = yaml.safe_load(f)

    rooms_grid = [3, 5, 7]
    ingredients_grid = [1, 2, 3]
    ops_grid = [
        {"cut": False, "cook": False, "label": "take-only"},
        {"cut": False, "cook": True, "label": "take+cook"},
        {"cut": True, "cook": True, "label": "take+cut+cook"},
    ]

    results: list[dict[str, Any]] = []
    combo_idx = 0
    for rooms in rooms_grid:
        for ingredients in ingredients_grid:
            for ops in ops_grid:
                combo_idx += 1
                combo_name = f"r{rooms}_i{ingredients}_{ops['label'].replace('+', '_')}"
                combo_dir = out_dir / combo_name
                if combo_dir.exists():
                    shutil.rmtree(combo_dir)
                combo_dir.mkdir(parents=True, exist_ok=True)
                combo_seed = _instance_seed(args.seed, combo_idx)
                games = _generate_combo_games(
                    base_dir=combo_dir,
                    num_instances=args.instances_per_combo,
                    seed=combo_seed,
                    num_rooms=rooms,
                    num_ingredients=ingredients,
                    cut=bool(ops["cut"]),
                    cook=bool(ops["cook"]),
                    open_=bool(args.open_),
                )
                metrics = _run_c0_batch(
                    game_files=games,
                    config=config,
                    use_real_model=bool(args.real),
                    max_steps=int(args.runtime_max_steps),
                )
                success_rate = float(metrics["success_rate"])
                mean_len = float(metrics["mean_episode_length"])
                candidate = {
                    "combo": {
                        "num_rooms": rooms,
                        "num_ingredients": ingredients,
                        "operations": _ops_name(bool(ops["cut"]), bool(ops["cook"])),
                        "open": bool(args.open_),
                    },
                    "metrics": metrics,
                    "target_window": {
                        "success_rate": [0.30, 0.50],
                        "episode_length": [8, 15],
                    },
                    "inside_window": (
                        0.30 <= success_rate <= 0.50 and 8.0 <= mean_len <= 15.0
                    ),
                    "distance_score": _score_candidate(success_rate, mean_len),
                }
                results.append(candidate)
                print(
                    f"[{combo_name}] success={success_rate:.3f} "
                    f"mean_steps={mean_len:.2f} inside={candidate['inside_window']}"
                )
                if not args.keep_games:
                    shutil.rmtree(combo_dir, ignore_errors=True)

    ranked = sorted(results, key=lambda x: float(x["distance_score"]))
    best = ranked[0] if ranked else None
    summary = {
        "seed": int(args.seed),
        "instances_per_combo": int(args.instances_per_combo),
        "grid": {
            "rooms": rooms_grid,
            "ingredients": ingredients_grid,
            "operations": [x["label"] for x in ops_grid],
            "open": bool(args.open_),
        },
        "runtime_max_steps": int(args.runtime_max_steps),
        "use_real_model": bool(args.real),
        "ranked_results": ranked,
        "best_candidate": best,
    }
    out_file = out_dir / "sweep_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTop 5 candidates:")
    for i, row in enumerate(ranked[:5], start=1):
        c = row["combo"]
        m = row["metrics"]
        print(
            f"{i}. rooms={c['num_rooms']} ingredients={c['num_ingredients']} "
            f"ops={c['operations']} success={m['success_rate']:.3f} "
            f"mean_steps={m['mean_episode_length']:.2f} score={row['distance_score']:.4f}"
        )
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
