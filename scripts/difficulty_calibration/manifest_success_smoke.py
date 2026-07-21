#!/usr/bin/env python3
"""Gate D manifest smoke — reference-stage success@Cap on all manifest instances (Hard-GO)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.difficulty_calibration.difficulty_metrics import (
    SUCCESS_CORRIDOR,
    episode_record,
    success_rate_at_cap,
)
from scripts.difficulty_calibration.sweep_textworld_difficulty import _load_merged_config

# Compute stage whose success@Cap the corridor criterion is judged against, per domain. C0 for
# ToH was found (2026-07) to be structurally near-0% regardless of configuration -- a systematic
# goal-peg-avoidance bias, not a difficulty-tuning problem (docs/consistency_log.md) -- so the
# corridor is calibrated and frozen against C1 for that domain instead.
REFERENCE_STAGE_BY_DOMAIN: dict[str, str] = {
    "textworld": "C0",
    "tower_of_hanoi": "C1",
}


def _run_domain_smoke(
    *,
    domain: str,
    config: dict[str, Any],
    production_cap: int,
    use_real_model: bool,
    reference_stage: str,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.execution.backend.factory import create_execution_backend
    from src.utils.experiment_env import make_experiment_env
    from src.utils.manifest import load_manifest
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    manifest = load_manifest(domain, config, REPO_ROOT)
    if not manifest:
        raise FileNotFoundError(f"No manifest loaded for domain={domain}")

    model = create_execution_backend(config, use_real=use_real_model)
    step_cfg = resolve_step_fn_kwargs(config, domain)
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    step_fn = get_step_fn(reference_stage, **step_cfg)

    episodes: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for iid in sorted(manifest.keys()):
        entry = manifest[iid]
        max_steps = production_cap if domain == "textworld" else production_cap
        env = make_experiment_env(domain, iid, config, max_steps, REPO_ROOT)
        if domain == "tower_of_hanoi":
            max_steps = int(getattr(env, "max_steps", max_steps))
        result = run_episode(
            env, model, reference_stage, step_fn=step_fn, max_steps=max_steps, **history_cfg
        )
        ep = episode_record(result, obs_ceiling=max_steps)
        ep["instance_id"] = iid
        ep["holdout"] = bool(entry.get("holdout"))
        ep["difficulty_tier"] = entry.get("difficulty_tier")
        episodes.append(ep)
        meta_rows.append(
            {
                "instance_id": iid,
                "holdout": ep["holdout"],
                "difficulty_tier": ep["difficulty_tier"],
                "task_success": ep["task_success"],
                "win_step": ep["win_step"],
                "episode_length_steps": ep["episode_length_steps"],
            }
        )

    if domain == "textworld":
        rate = success_rate_at_cap(episodes, production_cap)
    else:
        rate = sum(1 for ep in episodes if ep["task_success"]) / len(episodes) if episodes else 0.0

    lo, hi = SUCCESS_CORRIDOR
    return {
        "domain": domain,
        "reference_stage": reference_stage,
        "num_instances": len(episodes),
        "production_cap": production_cap,
        "success_rate_at_cap": rate,
        "inside_success_corridor": lo <= rate <= hi,
        "instances": meta_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate D manifest smoke (reference-stage success@Cap, all manifest instances)."
    )
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--production-cap", type=int, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/manifest_smoke",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["textworld", "tower_of_hanoi"],
    )
    args = parser.parse_args()

    config = _load_merged_config(REPO_ROOT / args.config)
    config.setdefault("episode", {})["max_steps_per_episode"] = int(args.production_cap)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for domain in args.domains:
        reference_stage = REFERENCE_STAGE_BY_DOMAIN.get(domain, "C0")
        report = _run_domain_smoke(
            domain=domain,
            config=config,
            production_cap=int(args.production_cap),
            use_real_model=bool(args.real),
            reference_stage=reference_stage,
        )
        reports.append(report)
        print(
            f"[{domain}/{reference_stage}] success@Cap={report['success_rate_at_cap']:.3f} "
            f"corridor={report['inside_success_corridor']} n={report['num_instances']}"
        )

    all_go = all(r["inside_success_corridor"] for r in reports)
    summary = {
        "production_cap": int(args.production_cap),
        "hard_go": all_go,
        "domains": reports,
    }
    out_file = out_dir / "manifest_smoke_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nHard-GO: {all_go}")
    print(f"Wrote {out_file}")
    if not all_go:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
