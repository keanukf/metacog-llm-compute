#!/usr/bin/env python3
"""
Inspect verbatim last actions for Gate D abort episodes (easiest TW cell).

Replays r3_i1_take-only only (same seeds as sweep) and extracts the final
action tail for episodes ending at quest Restdistanz 2–3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.difficulty_calibration.sweep_textworld_difficulty import (
    _create_model,
    _generate_combo_games,
    _instance_seed,
    _load_merged_config,
)


def _resolve_obs_ceiling(explicit: int | None, sweep: dict[str, Any]) -> int:
    """Cap to replay under: explicit CLI override, else the sweep's own ``obs_ceiling``.

    Without this fallback, an unset ``--obs-ceiling`` silently defaulted to 25
    regardless of what cap the sweep being inspected actually ran at (e.g. the
    cap-70 rerun noted in docs/consistency_log.md 2026-07-17) — replaying the same
    seeds under the wrong cap, unlike analyze_gate_d_abort_distance.py's sibling
    script, which already reads ``obs_ceiling`` from the sweep results.
    """
    if explicit is not None:
        return int(explicit)
    return int(sweep.get("obs_ceiling", 25))


def _step_tail(steps: list[dict[str, Any]], n: int = 4) -> list[dict[str, Any]]:
    tail = steps[-n:] if len(steps) >= n else steps
    out: list[dict[str, Any]] = []
    for s in tail:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "step_index": s.get("step_index"),
                "action_raw": s.get("action_raw"),
                "action_parsed": s.get("action_parsed"),
                "correctness": s.get("correctness"),
                "label_reason": s.get("label_reason"),
                "quest_distance_before": s.get("quest_distance_before"),
                "quest_distance_after": s.get("quest_distance_after"),
                "won": s.get("won"),
            }
        )
    return out


def _end_distance(steps: list[dict[str, Any]]) -> int | None:
    if not steps:
        return None
    last = steps[-1]
    dist = last.get("optimal_moves_remaining")
    if dist is None:
        dist = last.get("quest_distance_after")
    if dist is None:
        return None
    try:
        return int(dist)
    except (TypeError, ValueError):
        return None


def _classify_pattern(tail: list[dict[str, Any]], *, end_dist: int) -> str:
    """Heuristic triage: parsing | looping | cap_progress."""
    if not tail:
        return "unknown"
    actions = [str(s.get("action_parsed") or s.get("action_raw") or "").strip() for s in tail]
    correctness = [str(s.get("correctness") or "") for s in tail]
    dists = [s.get("quest_distance_after") for s in tail]

    if any(c == "illegal" for c in correctness):
        return "pattern_1_parsing_or_illegal"

    normalized = [a.lower() for a in actions if a]
    if len(normalized) >= 3:
        navish = sum(
            1
            for a in normalized
            if a.startswith(("go ", "look", "inventory", "examine", "open ", "close "))
        )
        if navish >= 2 and len(set(normalized)) <= len(normalized) - 1:
            return "pattern_2_looping_or_stuck"

    decreasing = False
    prev = None
    for d in dists:
        if d is None:
            continue
        if prev is not None and int(d) < int(prev):
            decreasing = True
        prev = d
    if decreasing and end_dist <= 3:
        return "pattern_3_cap_while_progressing"
    if end_dist <= 3:
        return "pattern_2_near_goal_no_finish"
    return "pattern_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-results",
        default="data/results/gate_d_calibration/textworld_sweep/sweep_results.json",
    )
    parser.add_argument("--config", default="configs/dev/gate_d_calibration.yaml")
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--obs-ceiling",
        type=int,
        default=None,
        help="Override the observation ceiling; defaults to the sweep's own obs_ceiling "
        "(same field analyze_gate_d_abort_distance.py reads) so this replays under the "
        "same cap the sweep it inspects actually ran at.",
    )
    parser.add_argument("--tail-steps", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="data/results/gate_d_calibration/textworld_abort_action_inspection",
    )
    args = parser.parse_args()

    sweep = json.loads((REPO_ROOT / args.sweep_results).read_text(encoding="utf-8"))
    seed = int(sweep.get("seed", 42))
    instances_per_combo = int(sweep.get("instances_per_combo", 8))
    obs_ceiling = _resolve_obs_ceiling(args.obs_ceiling, sweep)
    config = _load_merged_config(REPO_ROOT / args.config)

    import shutil
    import tempfile

    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.step_config import HISTORY_CFG_KEYS, resolve_step_fn_kwargs

    model = _create_model(config, bool(args.real))
    step_cfg = resolve_step_fn_kwargs(config, "textworld")
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in HISTORY_CFG_KEYS}
    c0 = get_step_fn("C0", **step_cfg)
    work_root = Path(tempfile.mkdtemp(prefix="gate_d_action_insp_"))
    combo_idx = 1  # r3_i1_take-only is first cell in sweep grid
    combo_seed = _instance_seed(seed, combo_idx)

    episodes: list[dict[str, Any]] = []
    try:
        games = _generate_combo_games(
            base_dir=work_root / "r3_i1_take-only",
            num_instances=instances_per_combo,
            seed=combo_seed,
            num_rooms=3,
            num_ingredients=1,
            cut=False,
            cook=False,
            open_=False,
        )
        for inst_idx, game_path in enumerate(games):
            env = TextWorldEnv(game_file=str(game_path), max_steps=obs_ceiling)
            result = run_episode(env, model, "C0", step_fn=c0, max_steps=obs_ceiling, **history_cfg)
            steps = result.get("step_correctness") or []
            end_dist = _end_distance(steps)
            tail = _step_tail(steps, n=int(args.tail_steps))
            ep = {
                "instance_index": inst_idx,
                "game_seed": _instance_seed(combo_seed, inst_idx),
                "task_success": bool(result.get("task_success")),
                "episode_length_steps": int(
                    result.get("episode_length_steps", result.get("steps", 0))
                ),
                "quest_distance_at_end": end_dist,
                "last_actions": tail,
                "pattern_hint": _classify_pattern(tail, end_dist=end_dist or 99),
            }
            episodes.append(ep)
            print(
                f"inst={inst_idx} end_dist={end_dist} success={ep['task_success']} "
                f"pattern={ep['pattern_hint']}"
            )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        close = getattr(model, "close", None)
        if callable(close):
            close()

    target = [e for e in episodes if e.get("quest_distance_at_end") in (2, 3)]
    report = {
        "combo": "r3_i1_take-only",
        "obs_ceiling": obs_ceiling,
        "tail_steps": int(args.tail_steps),
        "episodes_all": episodes,
        "episodes_end_dist_2_or_3": target,
    }

    lines = [
        "## Gate D — letzte Aktionen (r3_i1_take-only, Restdistanz 2–3)\n",
        f"Cap={obs_ceiling}, letzte **{args.tail_steps}** Aktionen pro Episode.\n",
    ]
    for ep in target:
        lines.append(
            f"### Instanz {ep['instance_index']} "
            f"(Restdistanz={ep['quest_distance_at_end']}, "
            f"Länge={ep['episode_length_steps']}, "
            f"Hint={ep['pattern_hint']})\n"
        )
        for s in ep["last_actions"]:
            lines.append(
                f"- step {s.get('step_index')}: "
                f"**`{s.get('action_raw')}`** "
                f"→ parsed `{s.get('action_parsed')}` | "
                f"{s.get('correctness')} | "
                f"dist {s.get('quest_distance_before')}→{s.get('quest_distance_after')}"
            )
        lines.append("")

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "abort_action_inspection.json"
    md_path = out_dir / "abort_action_inspection.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
