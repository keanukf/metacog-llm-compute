#!/usr/bin/env python3
"""
Sweep TextWorld Cooking generation difficulty and evaluate C0 behavior (Gate D).

Observation ceiling (default 25) is for length distribution only.
Production cap is derived globally from p90(win_step) + margin; corridor uses success@Cap.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_d_metrics import (  # noqa: E402
    LENGTH_GUIDANCE,
    PLAUSIBLE_BAND,
    SUCCESS_CORRIDOR,
    aggregate_length_stats,
    cap_ceiling_warning,
    collect_win_steps_from_cells,
    derive_production_cap,
    episode_record,
    json_combo_key,
    score_success_first,
    select_corridor_candidates,
    success_rate_at_cap,
    success_rate_at_obs,
)


def _to_abs(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _deep_merge_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge ``overlay`` onto a copy of ``base`` (dict values merge at every depth,
    scalars/lists replace). A shallow ``{**base[key], **overlay[key]}`` merge only handles one
    level of nesting: an overlay like ``domain_prompts.textworld.cot_max_tokens: 8192`` would
    silently wipe out sibling keys (``prefix``, ``action_stop``, ...) under
    ``domain_prompts.textworld`` instead of merging alongside them. Recursing per-key avoids that.
    """
    merged = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_overlay(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_merged_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        return {}
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    base_path = config_path.parent / str(extends)
    if not base_path.is_file():
        base_path = REPO_ROOT / str(extends)
    base = _load_merged_config(base_path)
    return _deep_merge_overlay(base, raw)


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
    from scripts.datasets.generate_textworld_games import _run_generate_command, _write_sidecar

    base_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for i in range(num_instances):
        game_seed = _instance_seed(seed, i)
        game_file = base_dir / f"textworld_{i}.z8"
        sidecar_file = base_dir / f"textworld_{i}.meta.json"
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


def _create_model(config: dict[str, Any], use_real_model: bool) -> Any:
    from src.execution.backend.factory import create_execution_backend

    return create_execution_backend(config, use_real=use_real_model)


def _run_c0_batch(
    *,
    game_files: list[Path],
    config: dict[str, Any],
    use_real_model: bool,
    obs_ceiling: int,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    model = _create_model(config, use_real_model)
    step_cfg = resolve_step_fn_kwargs(config, "textworld")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    c0 = get_step_fn("C0", **step_cfg)

    episodes: list[dict[str, Any]] = []
    for p in game_files:
        env = TextWorldEnv(game_file=str(p), max_steps=obs_ceiling)
        result = run_episode(env, model, "C0", step_fn=c0, max_steps=obs_ceiling, **history_cfg)
        episodes.append(episode_record(result, obs_ceiling=obs_ceiling))

    stats = aggregate_length_stats(episodes)
    stats["success_rate_at_obs"] = success_rate_at_obs(episodes)
    stats["episodes"] = episodes
    stats["step_count_distribution"] = {
        str(k): v
        for k, v in sorted(Counter(int(ep["episode_length_steps"]) for ep in episodes).items())
    }
    return stats


def _finalize_sweep(
    results: list[dict[str, Any]],
    *,
    obs_ceiling: int,
    cap_margin: int,
) -> dict[str, Any]:
    for row in results:
        row["success_rate_at_obs"] = float(row["metrics"]["success_rate_at_obs"])

    win_steps = collect_win_steps_from_cells(results, success_key="success_rate_at_obs")
    production_cap = derive_production_cap(win_steps, margin=cap_margin)
    if production_cap is None:
        production_cap = obs_ceiling

    all_eps = [ep for row in results for ep in row["metrics"]["episodes"]]
    global_stats = aggregate_length_stats(all_eps)
    p90_global = global_stats.get("p90_win_step_success")
    trunc_global = float(global_stats.get("truncation_rate") or 0.0)
    warnings = cap_ceiling_warning(
        production_cap=production_cap,
        obs_ceiling=obs_ceiling,
        truncation_rate=trunc_global,
        p90_win_step=float(p90_global) if p90_global is not None else None,
    )

    for row in results:
        eps = row["metrics"]["episodes"]
        row["success_rate_at_cap"] = success_rate_at_cap(eps, production_cap)
        med = row["metrics"].get("median_win_step_success")
        row["distance_score"] = score_success_first(
            float(row["success_rate_at_cap"]),
            float(med) if med is not None else None,
        )
        lo, hi = SUCCESS_CORRIDOR
        row["inside_success_corridor"] = lo <= row["success_rate_at_cap"] <= hi
        lg_lo, lg_hi = LENGTH_GUIDANCE
        med_val = row["metrics"].get("median_win_step_success")
        row["inside_length_guidance"] = med_val is not None and lg_lo <= float(med_val) <= lg_hi

    ranked = sorted(results, key=lambda x: float(x["distance_score"]))
    candidates = select_corridor_candidates(results)

    return {
        "production_cap": production_cap,
        "cap_derivation": {
            "method": "ceil(p90_win_step) + margin, min ceil(median)",
            "margin": cap_margin,
            "win_steps_sampled": len(win_steps),
            "plausible_band": list(PLAUSIBLE_BAND),
        },
        "obs_ceiling": obs_ceiling,
        "cap_warnings": warnings,
        "global_truncation_rate_at_obs": trunc_global,
        "global_p90_win_step_success": p90_global,
        "ranked_results": ranked,
        "corridor_candidates": candidates,
        "best_candidate": ranked[0] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep TextWorld Cooking difficulty for Gate D (success@Cap)."
    )
    parser.add_argument(
        "--config",
        default="configs/dev/gate_d_calibration.yaml",
        help="Experiment config path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/textworld_sweep",
        help="Where sweep artifacts are written",
    )
    parser.add_argument(
        "--instances-per-combo",
        type=int,
        default=8,
        help="Episodes per grid cell (Gate D default: 8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument(
        "--open", action="store_true", dest="open_", help="Enable open/close operations"
    )
    parser.add_argument(
        "--runtime-max-steps",
        type=int,
        default=25,
        help="Observation ceiling (not production cap)",
    )
    parser.add_argument("--cap-margin", type=int, default=2, help="Added to ceil(p90 win step)")
    parser.add_argument("--real", action="store_true", help="Use real model backend")
    parser.add_argument("--keep-games", action="store_true", help="Keep per-combo generated games")
    args = parser.parse_args()

    out_dir = _to_abs(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_merged_config(_to_abs(args.config))

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
                    obs_ceiling=int(args.runtime_max_steps),
                )
                candidate = {
                    "combo": {
                        "num_rooms": rooms,
                        "num_ingredients": ingredients,
                        "operations": _ops_name(bool(ops["cut"]), bool(ops["cook"])),
                        "open": bool(args.open_),
                    },
                    "combo_key": json_combo_key(
                        {
                            "num_rooms": rooms,
                            "num_ingredients": ingredients,
                            "operations": _ops_name(bool(ops["cut"]), bool(ops["cook"])),
                            "open": bool(args.open_),
                        }
                    ),
                    "metrics": metrics,
                    "target_window": {
                        "success_rate": list(SUCCESS_CORRIDOR),
                        "episode_length_guidance": list(LENGTH_GUIDANCE),
                    },
                }
                results.append(candidate)
                print(
                    f"[{combo_name}] success@obs={metrics['success_rate_at_obs']:.3f} "
                    f"trunc={metrics['truncation_rate']:.2f} "
                    f"p90_win={metrics.get('p90_win_step_success')}"
                )
                if not args.keep_games:
                    shutil.rmtree(combo_dir, ignore_errors=True)

    finalized = _finalize_sweep(
        results,
        obs_ceiling=int(args.runtime_max_steps),
        cap_margin=int(args.cap_margin),
    )

    # Strip raw episodes from JSON output (keep in separate file if needed)
    ranked_export = []
    for row in finalized["ranked_results"]:
        export_row = dict(row)
        export_metrics = dict(export_row["metrics"])
        export_metrics.pop("episodes", None)
        export_row["metrics"] = export_metrics
        ranked_export.append(export_row)

    summary = {
        "seed": int(args.seed),
        "instances_per_combo": int(args.instances_per_combo),
        "grid": {
            "rooms": rooms_grid,
            "ingredients": ingredients_grid,
            "operations": [x["label"] for x in ops_grid],
            "open": bool(args.open_),
        },
        "obs_ceiling": int(args.runtime_max_steps),
        "production_cap": finalized["production_cap"],
        "cap_derivation": finalized["cap_derivation"],
        "cap_warnings": finalized["cap_warnings"],
        "global_truncation_rate_at_obs": finalized["global_truncation_rate_at_obs"],
        "global_p90_win_step_success": finalized["global_p90_win_step_success"],
        "use_real_model": bool(args.real),
        "ranked_results": ranked_export,
        "corridor_candidates": [
            {k: v for k, v in c.items() if k != "metrics" or True}
            for c in finalized["corridor_candidates"]
        ],
        "best_candidate": ranked_export[0] if ranked_export else None,
    }
    for i, cand in enumerate(summary["corridor_candidates"]):
        m = dict(finalized["corridor_candidates"][i]["metrics"])
        m.pop("episodes", None)
        summary["corridor_candidates"][i] = {
            **{k: v for k, v in finalized["corridor_candidates"][i].items() if k != "metrics"},
            "metrics": m,
        }

    out_file = out_dir / "sweep_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nProduction cap (Gate D deliverable): {finalized['production_cap']}")
    for w in finalized["cap_warnings"]:
        print(f"  WARNING: {w}")
    print("\nTop corridor candidates (success@Cap):")
    for i, row in enumerate(summary["corridor_candidates"][:3], start=1):
        c = row["combo"]
        print(
            f"{i}. {row['combo_key']} success@Cap={row.get('success_rate_at_cap', 0):.3f} "
            f"rooms={c['num_rooms']} ing={c['num_ingredients']} ops={c['operations']}"
        )
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
