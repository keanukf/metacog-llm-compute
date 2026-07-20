#!/usr/bin/env python3
"""Gate F C1/C2 quality-control probe (HART) — real model, small n, full step traces.

Runs a handful of real episodes per (domain, stage) cell (C1 and C2 only) against the frozen
production manifests, under the exact production config (configs/experiment_core.yaml: cot
budget, sidecar policy, calibrated caps). Unlike the large Gate D/E sweeps, this deliberately
turns on save_step_traces so every LM call (including all C2 self-consistency candidates) is
captured in full for manual inspection, and adds automated checks for the three things Gate F
flagged as needing verification before committing to the full Phase 1/2 run:

1. Does C2's majority vote actually pick the majority action? Recomputed independently from the
   raw per-candidate vote keys in call_detail["subcalls"] and cross-checked against the recorded
   winner_index -- this is a real correctness check, not just "did a winner get picked."
2. Do outputs arrive intact -- no truncation? Checked via truncation_reason, whether a step ended
   with no admissible candidates, and whether generated tokens are suspiciously close to the
   cot_max_tokens ceiling (8192) without producing a parseable action.
3. Is nothing mis-parsed -- does the extracted action look like an action, not leaked reasoning?
   Flags any parsed action longer than a generous threshold or containing reasoning markers.

Usage:
    python scripts/gate_f_c1c2_quality_probe.py --n-episodes 5 --real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def _check_step(
    domain: str, stage: str, ep_id: str, step_index: int, rec: dict[str, Any]
) -> list[str]:
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
                    f"({s.get('reject_reason')}) after {tok} tokens -- looks like a cot_max_tokens=8192 truncation, "
                    "not a genuine parse failure"
                )
    elif stage == "C1" and isinstance(call_detail, dict):
        tok = call_detail.get("tokens_generated") or rec.get("tokens_generated") or 0
        if not action and tok >= 8000:
            problems.append(
                f"{ep_id} step{step_index}: no action parsed after {tok} tokens -- looks like cot_max_tokens=8192 truncation"
            )

    return problems


def run_cell(
    domain: str, stage: str, n_episodes: int, out_dir: Path, use_real: bool
) -> dict[str, Any]:
    from scripts.sweep_textworld_difficulty import _load_merged_config
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.execution.backend.factory import create_execution_backend
    from src.utils.experiment_env import make_experiment_env
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    config = _load_merged_config(REPO_ROOT / "configs/experiment_core.yaml")
    model = create_execution_backend(config, use_real=use_real)
    step_cfg = resolve_step_fn_kwargs(config, domain)
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    step_fn = get_step_fn(stage, **step_cfg)

    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = out_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    all_problems: list[str] = []
    episode_summaries: list[dict[str, Any]] = []

    for iid in INSTANCE_IDS[:n_episodes]:
        ep_id = f"qc_{domain}_{iid}_{stage}"
        # max_steps: production default; ToH's env overrides internally via its own per-instance
        # cap (src/utils/experiment_env.py), same mechanism the real Phase 1/2 pipeline now uses.
        max_steps = config.get("episode", {}).get("max_steps_per_episode", 45)
        env = make_experiment_env(domain, iid, config, max_steps, REPO_ROOT)
        effective_max_steps = int(getattr(env, "max_steps", max_steps))
        result = run_episode(
            env,
            model,
            stage,
            step_fn=step_fn,
            max_steps=effective_max_steps,
            save_step_traces=True,
            episode_id=ep_id,
            trace_output_dir=str(trace_dir),
            **history_cfg,
        )
        (ep_dir / f"{ep_id}.json").write_text(json.dumps(result, indent=2, default=str))

        trace_file = trace_dir / f"trace_{ep_id}.jsonl"
        step_problems: list[str] = []
        n_steps = 0
        if trace_file.exists():
            for line in trace_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_steps += 1
                step_problems.extend(_check_step(domain, stage, ep_id, rec["step_index"], rec))
        all_problems.extend(step_problems)
        episode_summaries.append(
            {
                "episode_id": ep_id,
                "instance": iid,
                "task_success": result.get("task_success"),
                "steps": result.get("episode_length_steps"),
                "max_steps": effective_max_steps,
                "n_trace_steps": n_steps,
                "n_problems": len(step_problems),
            }
        )
        print(
            f"  [{domain}/{stage}] inst={iid} steps={result.get('episode_length_steps')}/{effective_max_steps} "
            f"success={result.get('task_success')} problems={len(step_problems)}"
        )

    return {
        "domain": domain,
        "stage": stage,
        "n_episodes": len(episode_summaries),
        "episodes": episode_summaries,
        "problems": all_problems,
        "pass": not all_problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output-dir", default="data/results/gate_f_c1c2_quality_probe")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gate F C1/C2 quality probe -- n={args.n_episodes}/cell, real={args.real}\n")
    results = []
    for domain in DOMAINS:
        for stage in STAGES:
            print(f"=== {domain} / {stage} ===")
            cell_out = out_dir / f"{domain}_{stage}"
            results.append(run_cell(domain, stage, args.n_episodes, cell_out, args.real))
            print()

    all_pass = all(r["pass"] for r in results)
    summary = {
        "n_episodes_per_cell": args.n_episodes,
        "real": args.real,
        "cells": results,
        "pass": all_pass,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=== Summary ===")
    for r in results:
        status = "PASS" if r["pass"] else f"FAIL ({len(r['problems'])} problems)"
        print(f"  {r['domain']}/{r['stage']}: {status}")
        for p in r["problems"]:
            print(f"    - {p}")
    print(f"\n{'ALL PASS' if all_pass else 'ISSUES FOUND'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
