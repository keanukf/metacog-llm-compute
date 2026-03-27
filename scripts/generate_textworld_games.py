#!/usr/bin/env python3
"""
Generate TextWorld Cooking games and metadata sidecars for thesis experiments.

This script creates one compiled story file (``.z8`` with TextWorld 1.x / Inform 7),
the companion TextWorld ``.json`` game file (``Game.serialize``, required at play time),
plus one ``.meta.json`` experiment sidecar per instance (generation settings, seeds,
walkthrough summary — must not overwrite the ``.json``).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
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


def _num_rooms_to_go(num_rooms: int) -> int:
    """Map ``--num-rooms`` to TextWorld Cooking ``go`` presets (1, 6, 9, or 12)."""
    if num_rooms <= 1:
        return 1
    if num_rooms <= 6:
        return 6
    if num_rooms <= 9:
        return 9
    return 12


def _run_generate_command(
    *,
    output_file: Path,
    seed: int,
    num_rooms: int,
    num_ingredients: int,
    cut: bool,
    cook: bool,
    open_: bool,
) -> None:
    """
    Generate one cooking game via the registered ``tw-cooking`` challenge (TextWorld 1.x).

    Uses the Python API (``tw-make`` wraps the same). Inform 7 builds ``.z8`` story
    files only; there is no ``python -m textworld.challenges.tw_cooking`` runner.
    """
    try:
        import textworld
        from textworld.challenges import CHALLENGES
        from textworld.generator import compile_game
    except ImportError as e:
        raise RuntimeError("textworld is not installed.") from e

    _, make_game, _ = CHALLENGES["tw-cooking"]
    go = _num_rooms_to_go(num_rooms)
    # recipe_seed must stay 0 unless ``take`` > 0 (shuffle-recipe path in tw-cooking).
    settings: dict[str, Any] = {
        "recipe": num_ingredients,
        "take": 0,
        "go": go,
        "open": bool(open_),
        "cook": bool(cook),
        "cut": bool(cut),
        "drop": False,
        "recipe_seed": 0,
        "split": None,
    }
    options = textworld.GameOptions()
    options.seeds = seed
    options.path = str(output_file.resolve())
    options.file_ext = ".z8"
    options.force_recompile = True

    game = make_game(settings=settings, options=options)
    compile_game(game, options)


def _extract_game_metadata(game_file: Path) -> dict[str, Any]:
    """
    Load serialized game JSON written next to the compiled story file (``compile_game``).

    Returns a best-effort metadata dict and never raises.
    """
    data: dict[str, Any] = {
        "walkthrough": [],
        "entities": [],
        "max_score": None,
        "expected_step_count": None,
    }
    try:
        from textworld import Game  # type: ignore

        tw_json = game_file.with_suffix(".json")
        if not tw_json.exists():
            data["metadata_error"] = f"Missing companion Game JSON: {tw_json}"
            return data
        game = Game.load(str(tw_json))
        meta = getattr(game, "metadata", None) or {}
        walkthrough = meta.get("walkthrough") or []
        if not isinstance(walkthrough, list):
            walkthrough = []
        entities = []
        world = getattr(game, "world", None)
        if world is not None:
            for e in getattr(world, "entities", []) or []:
                name = getattr(e, "name", None)
                if isinstance(name, str):
                    entities.append(name)
        max_score = meta.get("max_score")
        if max_score is None:
            max_score = getattr(game, "max_score", None)
        if isinstance(max_score, (int, float)):
            max_score = float(max_score)
        else:
            max_score = None
        data = {
            "walkthrough": walkthrough,
            "entities": sorted(set(entities)),
            "max_score": max_score,
            "expected_step_count": len(walkthrough),
        }
    except Exception as e:
        data["metadata_error"] = f"{type(e).__name__}: {e}"
    return data


def _write_sidecar(
    *,
    sidecar_file: Path,
    game_file: Path,
    instance_id: int,
    master_seed: int,
    game_seed: int,
    num_rooms: int,
    num_ingredients: int,
    cut: bool,
    cook: bool,
    open_: bool,
) -> dict[str, Any]:
    game_meta = _extract_game_metadata(game_file)
    payload: dict[str, Any] = {
        "instance_id": instance_id,
        "game_file": str(game_file),
        "generation_parameters": {
            "num_rooms": int(num_rooms),
            "go_preset": _num_rooms_to_go(num_rooms),
            "num_ingredients": int(num_ingredients),
            "cut": bool(cut),
            "cook": bool(cook),
            "open": bool(open_),
            "max_steps_generation": 50,
        },
        "master_seed": int(master_seed),
        "game_seed": int(game_seed),
        "walkthrough": game_meta.get("walkthrough", []),
        "entities": game_meta.get("entities", []),
        "max_score": game_meta.get("max_score"),
        "expected_step_count": game_meta.get("expected_step_count"),
    }
    if "metadata_error" in game_meta:
        payload["metadata_error"] = game_meta["metadata_error"]
    sidecar_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _difficulty_tier(expected_step_count: Any) -> str:
    if not isinstance(expected_step_count, int):
        return "unknown"
    if expected_step_count <= 8:
        return "easy"
    if expected_step_count <= 15:
        return "medium"
    return "hard"


def _build_manifest(
    *,
    out_dir: Path,
    records: list[dict[str, Any]],
    holdout_count: int,
    holdout_rule: str,
) -> Path:
    if holdout_count < 0:
        holdout_count = 0
    total = len(records)
    holdout_count = min(holdout_count, total)
    # Deterministic holdout selection by sorted instance id.
    sorted_ids = sorted(int(r.get("instance_id", -1)) for r in records)
    holdout_ids = set(sorted_ids[:holdout_count])
    entries = []
    for r in records:
        idx = int(r["instance_id"])
        entry = {
            "instance_id": idx,
            "game_file": r["game_file"],
            "sidecar_file": str(out_dir / f"textworld_{idx}.meta.json"),
            "generation_parameters": r["generation_parameters"],
            "master_seed": r["master_seed"],
            "game_seed": r["game_seed"],
            "expected_step_count": r.get("expected_step_count"),
            "max_score": r.get("max_score"),
            "difficulty_tier": _difficulty_tier(r.get("expected_step_count")),
            "holdout": idx in holdout_ids,
        }
        entries.append(entry)
    manifest = {
        "dataset": "textworld_cooking",
        "num_instances": total,
        "holdout_count": holdout_count,
        "holdout_rule": holdout_rule,
        "entries": sorted(entries, key=lambda x: int(x["instance_id"])),
    }
    path = out_dir / "difficulty_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TextWorld Cooking dataset (.z8 + Game .json + .meta.json sidecars)."
    )
    parser.add_argument("--output-dir", default="data/tasks/textworld", help="Directory for generated game files")
    parser.add_argument("--num-rooms", type=int, required=True, help="Map size (rooms)")
    parser.add_argument("--num-ingredients", type=int, required=True, help="Recipe complexity")
    parser.add_argument("--cut", action="store_true", help="Enable cutting operations")
    parser.add_argument("--cook", action="store_true", help="Enable cooking operations")
    parser.add_argument("--open", action="store_true", dest="open_", help="Enable open/close operations")
    parser.add_argument("--num-instances", type=int, default=10, help="Number of game instances to generate")
    parser.add_argument("--seed", type=int, required=True, help="Master RNG seed")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for file naming")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write difficulty_manifest.json after generation",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=5,
        help="Holdout count used when --write-manifest is set",
    )
    args = parser.parse_args()
    out_dir = _to_abs(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.num_instances <= 0:
        raise ValueError("--num-instances must be >= 1")
    if args.num_rooms <= 0:
        raise ValueError("--num-rooms must be >= 1")
    if args.num_ingredients <= 0:
        raise ValueError("--num-ingredients must be >= 1")

    records: list[dict[str, Any]] = []
    for offset in range(args.num_instances):
        instance_id = args.start_index + offset
        game_seed = _instance_seed(args.seed, instance_id)
        game_file = out_dir / f"textworld_{instance_id}.z8"
        sidecar_file = out_dir / f"textworld_{instance_id}.meta.json"
        try:
            _run_generate_command(
                output_file=game_file,
                seed=game_seed,
                num_rooms=args.num_rooms,
                num_ingredients=args.num_ingredients,
                cut=args.cut,
                cook=args.cook,
                open_=args.open_,
            )
        except Exception as e:
            msg = (
                f"Could not generate {game_file.name}. "
                "Requires textworld with Inform 7 (bundled) and the ``tw-cooking`` challenge. "
                "See ``tw-make tw-cooking --help`` for constraints.\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
        sidecar = _write_sidecar(
            sidecar_file=sidecar_file,
            game_file=game_file,
            instance_id=instance_id,
            master_seed=args.seed,
            game_seed=game_seed,
            num_rooms=args.num_rooms,
            num_ingredients=args.num_ingredients,
            cut=args.cut,
            cook=args.cook,
            open_=args.open_,
        )
        records.append(sidecar)
        print(f"Generated {game_file.name} + {sidecar_file.name}")

    if args.write_manifest:
        manifest_path = _build_manifest(
            out_dir=out_dir,
            records=records,
            holdout_count=args.holdout_count,
            holdout_rule="first-N-by-instance-id",
        )
        print(f"Wrote {manifest_path}")
    print(f"Done. Dataset in {out_dir}")


if __name__ == "__main__":
    main()
