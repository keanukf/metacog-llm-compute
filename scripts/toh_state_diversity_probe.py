#!/usr/bin/env python3
"""
ToH state-diversity probe: does C0/C1 track the actual board, or recite a fixed pattern?

Context: two earlier trace episodes (both partial_start=0, i.e. the canonical from-scratch
board) showed C0 producing the same fixed move cycle regardless of legality, diverging from
the true solver path after the first two (canonically-correct) moves. That only proves
"same input -> same output"; it does not test whether the model adapts to a genuinely
different board. This script selects instances whose *required* first move is NOT the
canonical opener, generates traces at C0 and C1, and runs episodes concurrently (ThreadPool
against the shared ServerBackend, which is documented thread-safe for parallel episodes —
see src/execution/backend/server.py) so a real state-diversity probe doesn't cost the ~76
sequential minutes the earlier feasibility run did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_textworld_difficulty import _create_model, _load_merged_config  # noqa: E402
from src.utils.dotenv_loader import load_dotenv_if_present  # noqa: E402

_DOTENV_INFO = load_dotenv_if_present(REPO_ROOT)

CANONICAL_OPENER = ("A", "C")


def _pick_diverse_instances(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Greedily select up to n instances, round-robining across distinct required-first-move
    types and preferring non-canonical openers, so the probe doesn't just re-test the one
    state shape already covered by the existing trace_toh_C0_{0,1} episodes."""
    by_move: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inst in pool:
        sol = inst.get("optimal_solution") or []
        if not sol:
            continue  # already solved by partial_start; not useful for this probe
        first_move = tuple(sol[0])
        by_move.setdefault(first_move, []).append(inst)

    non_canonical_moves = [m for m in by_move if m != CANONICAL_OPENER]
    canonical_moves = [m for m in by_move if m == CANONICAL_OPENER]
    move_order = non_canonical_moves + canonical_moves  # exhaust non-canonical variety first

    selected: list[dict[str, Any]] = []
    idx_per_move = {m: 0 for m in move_order}
    while len(selected) < n and any(idx_per_move[m] < len(by_move[m]) for m in move_order):
        for m in move_order:
            if len(selected) >= n:
                break
            i = idx_per_move[m]
            if i < len(by_move[m]):
                selected.append(by_move[m][i])
                idx_per_move[m] = i + 1
    return selected[:n]


def _run_one_episode(
    *,
    inst: dict[str, Any],
    stage: str,
    step_fn: Any,
    model: Any,
    history_cfg: dict[str, Any],
    trace_dir: Path,
    out_dir: Path,
    ep_index: int,
    tracing_cfg: dict[str, Any] | None = None,
    trace_session_id: str | None = None,
) -> dict[str, Any]:
    from src.agent.base_agent import run_episode
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv
    from src.utils.tracing import build_trace_hook

    # A fresh hook per episode/task, not one shared across ThreadPoolExecutor workers:
    # LangfuseTraceHook keeps its OTel context-manager state as instance attributes, so
    # concurrent threads sharing one instance corrupt each other's context tokens (mirrors
    # src/execution/episode_runner.py::run_phase1_job, which builds its hook per job for the
    # same reason).
    trace_hook = build_trace_hook(tracing_cfg or {})
    ep_id = f"toh_diversity_{stage}_{ep_index}"
    max_steps = int(inst.get("max_steps", 50))
    env = TowerOfHanoiEnv(task=inst, max_steps=max_steps)
    try:
        result = run_episode(
            env,
            model,
            stage,
            step_fn=step_fn,
            max_steps=max_steps,
            save_step_traces=True,
            episode_id=ep_id,
            trace_output_dir=str(trace_dir),
            trace_hook=trace_hook,
            trace_session_id=trace_session_id,
            trace_tags=["gate_d", "toh_state_diversity_probe", stage],
            trace_name=ep_id,
            **history_cfg,
        )
    finally:
        try:
            trace_hook.episode_end()
        except Exception:
            pass
        flush_client = getattr(trace_hook, "_client", None)
        flush = getattr(flush_client, "flush", None)
        if callable(flush):
            flush()
    (out_dir / f"{ep_id}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    required_first_move = tuple((inst.get("optimal_solution") or [(None, None)])[0])
    summary = {
        "ep_id": ep_id,
        "stage": stage,
        "instance_id": inst.get("id"),
        "partial_start_moves": inst.get("partial_start_moves"),
        "required_first_move": list(required_first_move),
        "required_first_move_is_canonical": required_first_move == CANONICAL_OPENER,
        "task_success": bool(result.get("task_success")),
        "episode_length_steps": int(result.get("episode_length_steps", 0)),
        "max_steps": max_steps,
    }
    print(
        f"{ep_id}: required_first={required_first_move} "
        f"success={summary['task_success']} steps={summary['episode_length_steps']}/{max_steps}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToH state-diversity probe (non-canonical starts, C0+C1, concurrent)."
    )
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--num-disks", type=int, default=3, help="Lower bound of disk-count range")
    parser.add_argument(
        "--num-disks-hi",
        type=int,
        default=None,
        help="Upper bound of disk-count range; defaults to --num-disks (single value)",
    )
    parser.add_argument(
        "--partial-start-lo",
        type=int,
        default=1,
        help="Excludes 0 by default -- 0 is the canonical from-scratch board already probed",
    )
    parser.add_argument("--partial-start-hi", type=int, default=6)
    parser.add_argument("--num-selected", type=int, default=8)
    parser.add_argument(
        "--selection",
        choices=["diverse", "random"],
        default="diverse",
        help="'diverse' curates for required-first-move variety (bias diagnosis, NOT a "
        "representative success-rate/corridor sample); 'random' takes an uncurated slice of "
        "the generated pool (the right mode for corridor/calibration estimates).",
    )
    parser.add_argument("--stages", default="C0,C1")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--output-dir", default="data/results/gate_d_calibration/toh_state_diversity_probe"
    )
    args = parser.parse_args()

    from src.environments.tower_of_hanoi import generate_instances
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    num_disks_hi = int(args.num_disks_hi) if args.num_disks_hi is not None else int(args.num_disks)
    pool = generate_instances(
        int(args.pool_size),
        seed=int(args.seed),
        num_disks_range=(int(args.num_disks), num_disks_hi),
        partial_start_range=(int(args.partial_start_lo), int(args.partial_start_hi)),
    )
    if args.selection == "random":
        selected = pool[: int(args.num_selected)]
    else:
        selected = _pick_diverse_instances(pool, int(args.num_selected))
    print(f"Selected {len(selected)} instances from a pool of {len(pool)}:")
    for inst in selected:
        sol = inst.get("optimal_solution") or []
        print(
            f"  {inst['id']}: partial_start={inst['partial_start_moves']} "
            f"required_first_move={sol[0] if sol else None} initial_state={inst['initial_state']}"
        )

    config = _load_merged_config(REPO_ROOT / args.config)
    step_cfg = resolve_step_fn_kwargs(config, "tower_of_hanoi")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    model = _create_model(config, bool(args.real))

    from src.agent.compute_stages import get_step_fn
    from src.utils.tracing import log_langfuse_startup_status

    log_langfuse_startup_status(config, dotenv_info=_DOTENV_INFO)
    tracing_cfg = config.get("tracing")
    trace_session_id = f"toh_diversity_probe_{int(time.time())}"

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    step_fns = {stage: get_step_fn(stage, **step_cfg) for stage in stages}

    tasks = []
    for stage in stages:
        for i, inst in enumerate(selected):
            tasks.append((stage, inst, i))

    summaries: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as pool_exec:
            futures = {
                pool_exec.submit(
                    _run_one_episode,
                    inst=inst,
                    stage=stage,
                    step_fn=step_fns[stage],
                    model=model,
                    history_cfg=history_cfg,
                    trace_dir=trace_dir,
                    out_dir=out_dir,
                    ep_index=i,
                    tracing_cfg=tracing_cfg,
                    trace_session_id=trace_session_id,
                ): (stage, i)
                for stage, inst, i in tasks
            }
            for fut in as_completed(futures):
                summaries.append(fut.result())
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()

    summary_path = out_dir / "diversity_probe_summary.json"
    summary_path.write_text(json.dumps({"episodes": summaries}, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")

    for stage in stages:
        rows = [s for s in summaries if s["stage"] == stage]
        n_succ = sum(1 for r in rows if r["task_success"])
        print(f"{stage}: {n_succ}/{len(rows)} success")


if __name__ == "__main__":
    main()
