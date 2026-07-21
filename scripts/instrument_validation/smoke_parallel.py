#!/usr/bin/env python3
"""
Parallel execution smoke test (plumbing GO/NO-GO only — not TLE invariance validation).

Runs Phase-1 worklist via EpisodeScheduler against mock or ServerBackend.

Example:
  python scripts/smoke_parallel.py --config configs/dev/smoke.yaml --output-dir data/results/smoke_parallel
  python scripts/smoke_parallel.py --config configs/dev/smoke.yaml --real --output-dir data/results/smoke_parallel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.backend.factory import create_execution_backend
from src.execution.backend.server import verify_enable_thinking
from src.execution.config import ExecutionConfig
from src.execution.episode_runner import Phase1RunContext, run_phase1_job
from src.execution.metrics import build_execution_metrics
from src.execution.scheduler import EpisodeScheduler
from src.execution.worklist import build_phase1_worklist, expected_episode_ids
from src.utils.checkpointing import list_completed_episodes
from src.utils.logging_utils import write_run_metadata
from src.utils.run_output_layout import make_run_subdirectory
from src.utils.run_progress import log


def load_config(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_episode_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and data.get("schema_version") == "episode.v1"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/dev/smoke.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("data/results/smoke_parallel"))
    p.add_argument("--real", action="store_true")
    p.add_argument("--skip-thinking-check", action="store_true")
    args = p.parse_args()

    config_path = (
        (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    )
    config = load_config(config_path)
    exec_cfg = ExecutionConfig.from_config(config, real=bool(args.real))

    out_base = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = make_run_subdirectory(out_base, prefix="smoke_parallel")

    model = create_execution_backend(config, use_real=bool(args.real))

    if args.real and not args.skip_thinking_check:
        from src.execution.backend.server import ServerBackend

        if isinstance(model, ServerBackend):
            ok, detail = verify_enable_thinking(model)
            if not ok:
                print(f"SMOKE_PARALLEL: NO-GO (reason=enable_thinking:{detail})")
                return 1

    phase1 = config.get("phase1", {})
    domains = phase1.get("domains", [])
    instances = int(phase1.get("instances_per_domain", 2))
    runs = int(phase1.get("runs_per_condition", 2))
    stages = 3
    total_planned = len(domains) * instances * stages * runs
    model_cfg = config.get("model", {}) or {}

    write_run_metadata(
        checkpoint_dir,
        config,
        script="smoke_parallel.py",
        config_path=str(config_path),
        pilot_mode="cuda" if args.real else "mock",
        model_name=str(model_cfg.get("name", "mock")),
        model_dtype=str(model_cfg.get("dtype", "unknown")),
        domains=list(domains),
        total_episodes_planned=int(total_planned),
        resumed_from=0,
        repo_root=REPO_ROOT,
    )

    max_steps = int(config.get("episode", {}).get("max_steps_per_episode", 10))
    lg = config.get("logging") or {}
    from src.utils.logprob_sidecar import LogprobSidecarConfig

    run_ctx = Phase1RunContext(
        config=config,
        checkpoint_dir=checkpoint_dir,
        repo_root=REPO_ROOT,
        max_steps=max_steps,
        model_cfg=model_cfg,
        logprob_sidecar=LogprobSidecarConfig.from_logging_config(lg),
        save_vc_distributions=bool(lg.get("save_vc_distributions", False)),
        vc_export_format=str(lg.get("vc_export_format", "json")),
        vc_subdir=str(lg.get("vc_subdir", "vc")),
        save_step_traces=bool(lg.get("save_step_traces", False)),
        allow_history_truncation=False,
        verbose_steps=False,
        tracing_cfg=config.get("tracing"),
        log_fn=log,
    )

    jobs = build_phase1_worklist(config, completed=set(), quarantined=set())
    expected_ids = set(expected_episode_ids(jobs))
    scheduler = EpisodeScheduler(exec_cfg.max_concurrent_episodes)
    t0 = time.perf_counter()
    stats = scheduler.run(
        jobs,
        run_fn=lambda job: run_phase1_job(job, model, run_ctx),
        errors_path=checkpoint_dir / "errors.jsonl",
        checkpoint_dir=checkpoint_dir,
        log_fn=log,
    )
    wall = time.perf_counter() - t0

    completed_ids = list_completed_episodes(checkpoint_dir)
    missing = expected_ids - completed_ids
    extra = completed_ids - expected_ids

    check1 = len(missing) == 0 and all(
        _validate_episode_json(checkpoint_dir / f"{eid}.json") for eid in expected_ids
    )
    check2 = len(extra) == 0 and len(completed_ids) == len(expected_ids)
    check3 = stats.max_in_flight_observed > 1 if exec_cfg.max_concurrent_episodes > 1 else True

    exec_metrics = build_execution_metrics(
        checkpoint_dir=checkpoint_dir,
        total_wall_time_s=wall,
        total_tokens_generated=stats.total_tokens_generated,
        max_in_flight_observed=stats.max_in_flight_observed,
    )
    summary = {
        "expected_episodes": len(expected_ids),
        "completed_episodes": len(completed_ids),
        "max_in_flight_observed": stats.max_in_flight_observed,
        "checks": {"completeness": check1, "uniqueness": check2, "concurrency": check3},
        "execution_metrics": exec_metrics,
    }
    (checkpoint_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if check1 and check2 and check3:
        print("SMOKE_PARALLEL: GO")
        return 0
    reasons = []
    if not check1:
        reasons.append("completeness")
    if not check2:
        reasons.append("uniqueness")
    if not check3:
        reasons.append("concurrency")
    print(f"SMOKE_PARALLEL: NO-GO (reason={','.join(reasons)})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
