#!/usr/bin/env python3
"""Gate D — Tower of Hanoi C0 difficulty sweep (3 vs 4 disks)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_d_metrics import (  # noqa: E402
    SUCCESS_CORRIDOR,
    aggregate_length_stats,
    episode_record,
    success_rate_at_obs,
)
from scripts.sweep_textworld_difficulty import _load_merged_config  # noqa: E402


def _run_toh_c0_batch(
    *,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    use_real_model: bool,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv
    from src.execution.backend.factory import create_execution_backend
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    include_vm = bool(
        (config.get("domain_prompts") or {}).get("tower_of_hanoi", {}).get("include_valid_moves")
    )

    model = create_execution_backend(config, use_real=use_real_model)
    step_cfg = resolve_step_fn_kwargs(config, "tower_of_hanoi")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    c0 = get_step_fn("C0", **step_cfg)

    episodes: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()

    for inst in instances:
        max_steps = int(inst.get("max_steps", 50))
        env = TowerOfHanoiEnv(
            task=inst,
            max_steps=max_steps,
            include_valid_moves=include_vm,
        )
        result = run_episode(env, model, "C0", step_fn=c0, max_steps=max_steps, **history_cfg)
        episodes.append(episode_record(result, obs_ceiling=max_steps))
        for rec in result.get("step_correctness") or []:
            if isinstance(rec, dict):
                corr = rec.get("correctness")
                if isinstance(corr, str):
                    label_counts[corr] += 1

    stats = aggregate_length_stats(episodes)
    stats["success_rate_at_obs"] = success_rate_at_obs(episodes)
    stats["episodes"] = episodes
    total_labels = sum(label_counts.values())
    stats["label_distribution_c0"] = {
        k: {"count": v, "rate": (v / total_labels if total_labels else 0.0)}
        for k, v in sorted(label_counts.items())
    }
    stats["label_note"] = (
        "C0 illegal rate is a coarse proxy for disk choice only; "
        "C2 legal_or_optimal degeneration check belongs in Gate E."
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D ToH C0 sweep (3 vs 4 disks).")
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/toh_sweep",
    )
    parser.add_argument("--instances-per-combo", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    from src.environments.tower_of_hanoi import generate_instances

    config = _load_merged_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    disk_values = [3, 4]
    results: list[dict[str, Any]] = []
    for disks in disk_values:
        instances = generate_instances(
            int(args.instances_per_combo),
            seed=int(args.seed) + disks,
            num_disks_range=(disks, disks),
            partial_start_range=(0, 3),
        )
        metrics = _run_toh_c0_batch(
            instances=instances,
            config=config,
            use_real_model=bool(args.real),
        )
        rate = float(metrics["success_rate_at_obs"])
        lo, hi = SUCCESS_CORRIDOR
        row = {
            "num_disks": disks,
            "partial_start_range": [0, 3],
            "metrics": metrics,
            "success_rate_at_obs": rate,
            "inside_success_corridor": lo <= rate <= hi,
            "cap_mechanism": "instance max_steps = optimal_steps * 3 (not Gate-D-chosen fixed cap)",
        }
        results.append(row)
        labels = metrics.get("label_distribution_c0", {})
        print(
            f"[disks={disks}] success={rate:.3f} corridor={row['inside_success_corridor']} "
            f"labels={ {k: round(v['rate'], 3) for k, v in labels.items()} }"
        )

    ranked = sorted(
        results,
        key=lambda r: abs(float(r["success_rate_at_obs"]) - 0.40),
    )
    summary = {
        "seed": int(args.seed),
        "instances_per_combo": int(args.instances_per_combo),
        "partial_start_range": [0, 3],
        "use_real_model": bool(args.real),
        "results": results,
        "best_candidate": ranked[0] if ranked else None,
    }
    out_file = out_dir / "sweep_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
