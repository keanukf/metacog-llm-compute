#!/usr/bin/env python3
"""
Build difficulty_manifest.json for final TextWorld dataset artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_abs(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _difficulty_tier(expected_step_count: Any) -> str:
    if not isinstance(expected_step_count, int):
        return "unknown"
    if expected_step_count <= 8:
        return "easy"
    if expected_step_count <= 15:
        return "medium"
    return "hard"


def _load_sidecar(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sidecar JSON: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TextWorld difficulty_manifest.json with fixed holdout split.")
    parser.add_argument(
        "--dataset-dir",
        default="data/tasks/textworld",
        help="Directory containing textworld_*.z8 (or legacy *.ulx) + matching .meta.json sidecars",
    )
    parser.add_argument("--holdout-count", type=int, default=5, help="Number of held-out instances")
    parser.add_argument(
        "--holdout-policy",
        choices=["first-n", "mod-10"],
        default="first-n",
        help="Deterministic holdout policy",
    )
    args = parser.parse_args()

    dataset_dir = _to_abs(args.dataset_dir)
    game_files = sorted(dataset_dir.glob("textworld_*.z8"))
    if not game_files:
        game_files = sorted(dataset_dir.glob("textworld_*.ulx"))
    if not game_files:
        raise FileNotFoundError(
            f"No textworld_*.z8 or textworld_*.ulx files found in {dataset_dir}"
        )

    entries: list[dict[str, Any]] = []
    for game_file in game_files:
        try:
            idx = int(game_file.stem.split("_")[-1])
        except Exception as e:
            raise ValueError(f"Unexpected file naming: {game_file.name}") from e
        meta_path = game_file.parent / f"{game_file.stem}.meta.json"
        if meta_path.exists():
            sidecar_file = meta_path
        else:
            # Legacy layout: experiment metadata was written to ``{stem}.json`` (overwrote Game JSON — avoid).
            legacy = game_file.with_suffix(".json")
            if not legacy.exists():
                raise FileNotFoundError(
                    f"Missing experiment sidecar for {game_file.name}: "
                    f"expected {meta_path.name} (or legacy {legacy.name})"
                )
            with open(legacy) as f:
                probe = json.load(f)
            if not isinstance(probe, dict) or "generation_parameters" not in probe:
                raise FileNotFoundError(
                    f"Missing {meta_path.name} for {game_file.name}. "
                    "If you only have TextWorld's Game .json, regenerate with "
                    "`scripts/generate_textworld_games.py` so .meta.json is written."
                )
            sidecar_file = legacy
        sidecar = _load_sidecar(sidecar_file)
        expected_steps = sidecar.get("expected_step_count")
        entry = {
            "instance_id": idx,
            "game_file": str(game_file),
            "sidecar_file": str(sidecar_file),
            "generation_parameters": sidecar.get("generation_parameters", {}),
            "master_seed": sidecar.get("master_seed"),
            "game_seed": sidecar.get("game_seed"),
            "walkthrough_length": len(sidecar.get("walkthrough", []) or []),
            "expected_step_count": expected_steps,
            "max_score": sidecar.get("max_score"),
            "difficulty_tier": _difficulty_tier(expected_steps),
        }
        entries.append(entry)

    entries = sorted(entries, key=lambda x: int(x["instance_id"]))
    holdout_count = max(0, min(int(args.holdout_count), len(entries)))
    holdout_ids: set[int]
    if args.holdout_policy == "mod-10":
        # Deterministic, spread selection: 0,10,20,... up to holdout_count items.
        mod_candidates = [int(e["instance_id"]) for e in entries if int(e["instance_id"]) % 10 == 0]
        holdout_ids = set(mod_candidates[:holdout_count])
        if len(holdout_ids) < holdout_count:
            # Fill from remaining ids deterministically.
            for e in entries:
                idx = int(e["instance_id"])
                if idx not in holdout_ids:
                    holdout_ids.add(idx)
                if len(holdout_ids) >= holdout_count:
                    break
    else:
        holdout_ids = set(int(e["instance_id"]) for e in entries[:holdout_count])

    for e in entries:
        e["holdout"] = int(e["instance_id"]) in holdout_ids

    manifest = {
        "dataset": "textworld_cooking",
        "num_instances": len(entries),
        "holdout_count": holdout_count,
        "non_holdout_count": len(entries) - holdout_count,
        "holdout_policy": args.holdout_policy,
        "entries": entries,
    }
    out_file = dataset_dir / "difficulty_manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")
    print(f"Holdout: {holdout_count} | Train/Eval: {len(entries) - holdout_count}")


if __name__ == "__main__":
    main()
