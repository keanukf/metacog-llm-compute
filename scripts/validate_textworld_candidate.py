#!/usr/bin/env python3
"""Validate 2–3 TextWorld Gate D candidates with fresh C0 episodes at production Cap."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_d_metrics import (  # noqa: E402
    SUCCESS_CORRIDOR,
    success_rate_at_cap,
)
from scripts.sweep_textworld_difficulty import (  # noqa: E402
    _generate_combo_games,
    _load_merged_config,
    _run_c0_batch,
)


def _instance_seed(master_seed: int, idx: int) -> int:
    rng = random.Random(master_seed)
    for _ in range(idx + 1):
        value = rng.randint(0, 2**31 - 1)
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TW candidate at production Cap.")
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument(
        "--sweep-results", required=True, help="Path to textworld sweep_results.json"
    )
    parser.add_argument("--production-cap", type=int, required=True)
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument(
        "--candidate-index",
        type=int,
        action="append",
        dest="candidate_indices",
        help="Index into corridor_candidates (0-based); repeat for multiple",
    )
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/textworld_validation",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep_results)
    if not sweep_path.is_absolute():
        sweep_path = REPO_ROOT / sweep_path
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    candidates = sweep.get("corridor_candidates") or sweep.get("ranked_results") or []
    if not candidates:
        raise SystemExit("No candidates in sweep results.")

    indices = (
        args.candidate_indices if args.candidate_indices else list(range(min(3, len(candidates))))
    )
    config = _load_merged_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(candidates):
            print(f"Skip invalid candidate index {idx}")
            continue
        cand = candidates[idx]
        combo = cand["combo"]
        combo_dir = out_dir / f"candidate_{idx}"
        if combo_dir.exists():
            shutil.rmtree(combo_dir)
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_seed = _instance_seed(int(args.seed), idx + 1)
        ops = str(combo.get("operations", "take-only"))
        cut = ops == "take+cut+cook"
        cook = ops in {"take+cook", "take+cut+cook"}
        games = _generate_combo_games(
            base_dir=combo_dir,
            num_instances=int(args.num_instances),
            seed=combo_seed,
            num_rooms=int(combo["num_rooms"]),
            num_ingredients=int(combo["num_ingredients"]),
            cut=cut,
            cook=cook,
            open_=bool(combo.get("open")),
        )
        metrics = _run_c0_batch(
            game_files=games,
            config=config,
            use_real_model=bool(args.real),
            obs_ceiling=int(args.production_cap),
        )
        episodes = metrics.pop("episodes")
        rate = success_rate_at_cap(episodes, int(args.production_cap))
        lo, hi = SUCCESS_CORRIDOR
        row = {
            "candidate_index": idx,
            "combo": combo,
            "production_cap": int(args.production_cap),
            "num_instances": int(args.num_instances),
            "success_rate_at_cap": rate,
            "inside_success_corridor": lo <= rate <= hi,
            "metrics": metrics,
        }
        validation_rows.append(row)
        print(
            f"candidate[{idx}] {combo} success@Cap={rate:.3f} "
            f"corridor={row['inside_success_corridor']}"
        )

    summary = {
        "production_cap": int(args.production_cap),
        "sweep_results": str(sweep_path),
        "validations": validation_rows,
        "any_in_corridor": any(r["inside_success_corridor"] for r in validation_rows),
    }
    out_file = out_dir / "validation_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
