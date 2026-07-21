#!/usr/bin/env python3
"""Gate F C1/C2 quality-control probe (HART) — real model, small n, fully parallel, full traces.

Runs a handful of real episodes per (domain, stage) cell (C1 and C2 only) against the frozen
production manifests, under the exact production config (configs/experiment_core.yaml: cot
budget, sidecar policy, calibrated caps) -- via the *actual* production job function
(``src/execution/episode_runner.py::run_phase1_job``) and the *actual* production scheduler
(``src/execution/scheduler.py::EpisodeScheduler``), not a reimplementation. All cells/episodes
are submitted as one flat job list and run concurrently (bounded by ``--max-concurrent``), the
same way a real Phase 1 block does -- this is a small slice of exactly that code path, not a
separate one.

Unlike the large Gate D/E sweeps, this deliberately turns on save_step_traces so every LM call
(including all C2 self-consistency candidates) is captured in full for manual inspection, and
adds automated checks for the three things Gate F flagged as needing verification before
committing to the full Phase 1/2 run:

1. Does C2's majority vote actually pick the majority action? Recomputed independently from the
   raw per-candidate vote keys in call_detail["subcalls"] and cross-checked against the recorded
   winner_index -- this is a real correctness check, not just "did a winner get picked."
2. Do outputs arrive intact -- no truncation? Checked via truncation_reason, whether a step ended
   with no admissible candidates, and whether generated tokens are suspiciously close to the
   cot_max_tokens ceiling (8192) without producing a parseable action.
3. Is nothing mis-parsed -- does the extracted action look like an action, not leaked reasoning?
   Flags any parsed action longer than a generous threshold or containing reasoning markers.

Usage:
    python scripts/run_readiness/c1_c2_quality_probe.py --n-episodes 5 --real --max-concurrent 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.dotenv_loader import load_dotenv_if_present

load_dotenv_if_present(REPO_ROOT)

DOMAINS = ("textworld", "tower_of_hanoi")
STAGES = ("C1", "C2")
# mod-10 holdout is {0,10,20,30,40}; 1..5 is always non-holdout for either domain's frozen manifest.
INSTANCE_IDS = [1, 2, 3, 4, 5]

# A parsed action should be a short command/move, never a reasoning fragment.
MAX_PLAUSIBLE_ACTION_CHARS = 60
LEAK_MARKERS = ("<think", "</think>", "step-by-step", "let me", "i need to", "first,")


def _recompute_c2_majority(subcalls: list[dict[str, Any]]) -> tuple[str | None, dict[str, int]]:
    """Independently recompute the majority vote from raw candidate vote keys."""
    from collections import Counter

    admissible = [s for s in subcalls if s.get("kind") == "sample" and s.get("admissible")]
    counts = Counter(
        str(s.get("action_normalized") or "") for s in admissible if s.get("action_normalized")
    )
    if not counts:
        return None, {}
    max_count = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == max_count)
    # Genuine ties are broken by the production tie-break RNG (src/agent/stages/shared.py);
    # we can't reproduce the exact draw here, so a tie is reported as such, not treated as a
    # single "correct" answer.
    return (winners[0] if len(winners) == 1 else f"TIE[{','.join(winners)}]"), dict(counts)


def _check_step(stage: str, ep_id: str, step_index: int, rec: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    action = str(rec.get("action_parsed") or "")

    if not action:
        problems.append(f"{ep_id} step{step_index}: no action parsed at all")
    elif len(action) > MAX_PLAUSIBLE_ACTION_CHARS:
        problems.append(
            f"{ep_id} step{step_index}: parsed action suspiciously long ({len(action)} chars): {action[:80]!r}"
        )
    low = action.lower()
    if any(m in low for m in LEAK_MARKERS):
        problems.append(
            f"{ep_id} step{step_index}: parsed action looks like leaked reasoning: {action[:80]!r}"
        )

    call_detail = rec.get("call_detail") or {}
    if stage == "C2" and isinstance(call_detail, dict):
        subcalls = call_detail.get("subcalls") or []
        n_samples = call_detail.get("n_samples")
        n_admissible = call_detail.get("n_samples_admissible")
        winner_index = call_detail.get("winner_index")
        truncation_reason = call_detail.get("truncation_reason")

        if n_samples != 3:
            problems.append(f"{ep_id} step{step_index}: n_samples={n_samples}, expected 3")
        if truncation_reason:
            problems.append(f"{ep_id} step{step_index}: truncation_reason={truncation_reason!r}")
        if not n_admissible:
            problems.append(f"{ep_id} step{step_index}: 0 admissible candidates out of {n_samples}")

        recomputed_winner, vote_counts = _recompute_c2_majority(subcalls)
        winner_sample = next((s for s in subcalls if s.get("is_winner")), None)
        winner_key = str(winner_sample.get("action_normalized") or "") if winner_sample else None
        if recomputed_winner is not None and not str(recomputed_winner).startswith("TIE["):
            if winner_key != recomputed_winner:
                problems.append(
                    f"{ep_id} step{step_index}: VOTING MISMATCH -- recorded winner "
                    f"(index={winner_index}, key={winner_key!r}) != independently recomputed "
                    f"majority ({recomputed_winner!r}), vote_counts={vote_counts}"
                )
        for s in subcalls:
            if s.get("kind") != "sample":
                continue
            tok = s.get("tokens_generated") or 0
            if not s.get("admissible") and tok >= 8000:
                problems.append(
                    f"{ep_id} step{step_index}: sample {s.get('sample_index')} rejected "
                    f"({s.get('reject_reason')}) after {tok} tokens -- looks like a "
                    "cot_max_tokens=8192 truncation, not a genuine parse failure"
                )
    elif stage == "C1" and isinstance(call_detail, dict):
        tok = call_detail.get("tokens_generated") or rec.get("tokens_generated") or 0
        if not action and tok >= 8000:
            problems.append(
                f"{ep_id} step{step_index}: no action parsed after {tok} tokens -- looks like "
                "cot_max_tokens=8192 truncation"
            )

    return problems


def build_jobs(n_episodes: int) -> list[Any]:
    from src.execution.worklist import EpisodeJob

    jobs = []
    for domain in DOMAINS:
        for stage in STAGES:
            for iid in INSTANCE_IDS[:n_episodes]:
                jobs.append(
                    EpisodeJob(
                        episode_id=f"qc_{domain}_{iid}_{stage}",
                        domain=domain,
                        instance=iid,
                        run=0,
                        phase="phase1",
                        compute_stage=stage,
                    )
                )
    return jobs


def run_probe(
    n_episodes: int, use_real: bool, max_concurrent: int, out_dir: Path
) -> dict[str, Any]:
    from scripts.difficulty_calibration.sweep_textworld_difficulty import _load_merged_config
    from src.execution.backend.factory import create_execution_backend
    from src.execution.episode_runner import Phase1RunContext, run_phase1_job
    from src.execution.scheduler import EpisodeScheduler
    from src.utils.logprob_sidecar import LogprobSidecarConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_merged_config(REPO_ROOT / "configs/experiment_core.yaml")
    model = create_execution_backend(config, use_real=use_real)
    max_steps = int(config.get("episode", {}).get("max_steps_per_episode", 45))
    logging_cfg = config.get("logging") or {}

    ctx = Phase1RunContext(
        config=config,
        checkpoint_dir=out_dir,
        repo_root=REPO_ROOT,
        max_steps=max_steps,
        model_cfg=config.get("model", {}),
        logprob_sidecar=LogprobSidecarConfig.from_logging_config(logging_cfg),
        save_vc_distributions=False,
        vc_export_format=str(logging_cfg.get("vc_export_format", "json")),
        vc_subdir=str(logging_cfg.get("vc_subdir", "vc")),
        save_step_traces=True,
        allow_history_truncation=False,
        verbose_steps=False,
        tracing_cfg=config.get("tracing"),
        log_fn=print,
    )

    jobs = build_jobs(n_episodes)
    print(
        f"Gate F C1/C2 quality probe -- {len(jobs)} episodes "
        f"({len(DOMAINS)} domains x {len(STAGES)} stages x {n_episodes}/cell), "
        f"max_concurrent={max_concurrent}, real={use_real}\n"
    )

    scheduler = EpisodeScheduler(max_concurrent_episodes=max_concurrent)
    t0 = time.perf_counter()
    stats = scheduler.run(
        jobs,
        run_fn=lambda job: run_phase1_job(job, model, ctx),
        checkpoint_dir=out_dir,
        log_fn=lambda msg: print(f"  {msg}"),
    )
    wall_s = time.perf_counter() - t0
    print(
        f"\nDone in {wall_s:.0f}s: {stats.episodes_completed} completed, "
        f"{stats.episodes_failed} failed, max_in_flight={stats.max_in_flight_observed}"
    )
    return {
        "wall_s": wall_s,
        "episodes_completed": stats.episodes_completed,
        "episodes_failed": stats.episodes_failed,
    }


def analyze(out_dir: Path, n_episodes: int) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, Any]] = {
        (d, s): {"domain": d, "stage": s, "episodes": [], "problems": []}
        for d in DOMAINS
        for s in STAGES
    }
    for domain in DOMAINS:
        for stage in STAGES:
            for iid in INSTANCE_IDS[:n_episodes]:
                ep_id = f"qc_{domain}_{iid}_{stage}"
                ep_path = out_dir / f"{ep_id}.json"
                trace_path = out_dir / f"trace_{ep_id}.jsonl"
                cell = cells[(domain, stage)]
                if not ep_path.exists():
                    cell["problems"].append(f"{ep_id}: episode did not complete (no output file)")
                    continue
                result = json.loads(ep_path.read_text())
                step_problems: list[str] = []
                n_steps = 0
                if trace_path.exists():
                    for line in trace_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        n_steps += 1
                        step_problems.extend(_check_step(stage, ep_id, rec["step_index"], rec))
                else:
                    step_problems.append(f"{ep_id}: no trace file found")
                cell["problems"].extend(step_problems)
                cell["episodes"].append(
                    {
                        "episode_id": ep_id,
                        "instance": iid,
                        "task_success": result.get("task_success"),
                        "steps": result.get("episode_length_steps"),
                        "n_trace_steps": n_steps,
                        "n_problems": len(step_problems),
                    }
                )

    results = []
    for (domain, stage), cell in cells.items():
        results.append({**cell, "n_episodes": len(cell["episodes"]), "pass": not cell["problems"]})
    return {"cells": results, "pass": all(r["pass"] for r in results)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Bounded parallel episodes (<=32, the C-1 batch-invariance frozen ceiling).",
    )
    parser.add_argument("--output-dir", default="data/results/gate_f_c1c2_quality_probe")
    args = parser.parse_args()
    if args.max_concurrent > 32:
        raise SystemExit("--max-concurrent must not exceed 32 (C-1 batch-invariance freeze)")

    out_dir = REPO_ROOT / args.output_dir
    run_stats = run_probe(args.n_episodes, args.real, args.max_concurrent, out_dir)
    analysis = analyze(out_dir, args.n_episodes)

    summary = {"n_episodes_per_cell": args.n_episodes, "real": args.real, **run_stats, **analysis}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    for r in analysis["cells"]:
        status = "PASS" if r["pass"] else f"FAIL ({len(r['problems'])} problems)"
        print(
            f"  {r['domain']}/{r['stage']}: {status} ({r['n_episodes']}/{args.n_episodes} episodes)"
        )
        for p in r["problems"]:
            print(f"    - {p}")
    print(f"\n{'ALL PASS' if analysis['pass'] else 'ISSUES FOUND'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    if not analysis["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
