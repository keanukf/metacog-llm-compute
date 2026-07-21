#!/usr/bin/env python3
"""
CPU-only audit: command verbs/forms required by tw-cooking grid (27 cells).

Collects walkthrough + policy_commands trajectory + admissible_commands samples
per cell; compares against experiment_core.yaml template list.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.generate_textworld_games import (
    _instance_seed,
    _run_generate_command,
    _write_sidecar,
)

DIRS = frozenset({"north", "south", "east", "west", "up", "down"})
COOK_VERBS = frozenset({"cook", "fry", "roast", "grill", "bbq", "broil", "toast"})
CUT_VERBS = frozenset({"cut", "slice", "chop", "dice"})

CURRENT_PROMPT_TEMPLATES = [
    "go [north|south|east|west|up|down]",
    "take [object]",
    "drop [object]",
    "examine [object]",
    "open [container]",
    "close [container]",
    "put [object] in [container]",
    "cook [object] with [tool]",
    "inventory",
    "look",
]


def generalize(cmd: str) -> str:
    s = " ".join(cmd.strip().split())
    low = s.lower()
    parts = low.split()
    if not parts:
        return s
    verb = parts[0]
    if verb == "go" and len(parts) == 2 and parts[1] in DIRS:
        return "go [north|south|east|west|up|down]"
    if verb == "take" and len(parts) >= 2:
        return "take [object]"
    if verb == "drop" and len(parts) >= 2:
        return "drop [object]"
    if verb == "examine" and len(parts) >= 2:
        return "examine [object]"
    if verb == "open" and len(parts) >= 2:
        return "open [container]"
    if verb == "close" and len(parts) >= 2:
        return "close [container]"
    if verb == "put" and " in " in low:
        return "put [object] in [container]"
    if verb in COOK_VERBS and " with " in low:
        return f"{verb} [object] with [tool]"
    if verb in CUT_VERBS and " with " in low:
        return f"{verb} [object] with [tool]"
    if low in {"inventory", "look", "prepare meal", "eat meal"}:
        return low
    if verb == "look" and len(parts) >= 2:
        return "look [object]"
    return s


def normalize_for_prompt_diff(pattern: str) -> str:
    """Map observed cook-verb variants to prompt's single cook template for diff only."""
    for v in COOK_VERBS:
        if pattern.startswith(f"{v} "):
            return pattern.replace(v, "cook", 1)
    return pattern


def collect_policy_via_gym(
    game_file: Path, max_steps: int = 200
) -> tuple[list[str], list[list[str]], str | None]:
    from src.environments.textworld_env import TextWorldEnv

    env = TextWorldEnv(game_file=str(game_file), max_steps=max_steps)
    if not env._use_real or env._gym_env is None:
        return [], [], "textworld_unavailable"

    gym = env._gym_env
    result = gym.reset()
    info: dict[str, Any] = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
    trajectory: list[str] = []
    adm_snapshots: list[list[str]] = []
    err: str | None = None

    for _ in range(max_steps):
        adm = info.get("admissible_commands") or []
        if isinstance(adm, list):
            adm_snapshots.append(list(adm))
        if info.get("won") or info.get("game_won"):
            break
        policy = info.get("policy_commands") or []
        if not policy:
            err = "empty_policy_before_win"
            break
        cmd = str(policy[0])
        trajectory.append(cmd)
        step_out = gym.step(cmd)
        if len(step_out) == 5:
            _, _, term, trunc, info = step_out
            done = bool(term) or bool(trunc)
        else:
            _, _, done, info = step_out[0], step_out[1], step_out[2], step_out[3]
        if info.get("won") or info.get("game_won"):
            break
        if done:
            err = "done_without_win"
            break
    else:
        err = "max_steps"

    return trajectory, adm_snapshots, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/results/gate_d_calibration/textworld_vocab_audit.json"
    )
    args = parser.parse_args()

    rooms_grid = [3, 5, 7]
    ingredients_grid = [1, 2, 3]
    ops_grid = [
        {"cut": False, "cook": False, "label": "take-only"},
        {"cut": False, "cook": True, "label": "take+cook"},
        {"cut": True, "cook": True, "label": "take+cut+cook"},
    ]

    work = Path(tempfile.mkdtemp(prefix="tw_vocab_audit_"))
    all_patterns: Counter[str] = Counter()
    pattern_examples: dict[str, set[str]] = defaultdict(set)
    by_cell: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    try:
        combo_idx = 0
        for rooms in rooms_grid:
            for ingredients in ingredients_grid:
                for ops in ops_grid:
                    combo_idx += 1
                    ops_label = str(ops["label"]).replace("+", "_")
                    label = f"r{rooms}_i{ingredients}_{ops_label}"
                    combo_seed = _instance_seed(int(args.seed), combo_idx)
                    game_seed = _instance_seed(combo_seed, 0)
                    game_file = work / f"{label}.z8"
                    sidecar = work / f"{label}.meta.json"
                    _run_generate_command(
                        output_file=game_file,
                        seed=game_seed,
                        num_rooms=rooms,
                        num_ingredients=ingredients,
                        cut=bool(ops["cut"]),
                        cook=bool(ops["cook"]),
                        open_=False,
                    )
                    _write_sidecar(
                        sidecar_file=sidecar,
                        game_file=game_file,
                        instance_id=0,
                        master_seed=combo_seed,
                        game_seed=game_seed,
                        num_rooms=rooms,
                        num_ingredients=ingredients,
                        cut=bool(ops["cut"]),
                        cook=bool(ops["cook"]),
                        open_=False,
                    )
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    walkthrough = [str(x) for x in (meta.get("walkthrough") or [])]
                    traj, adm_snaps, err = collect_policy_via_gym(game_file)
                    if err:
                        errors.append(
                            {
                                "cell": label,
                                "err": err,
                                "walkthrough_len": len(walkthrough),
                                "policy_traj_len": len(traj),
                            }
                        )

                    adm_all: set[str] = set()
                    for snap in adm_snaps:
                        adm_all.update(str(c) for c in snap)
                    for idx in {0, len(adm_snaps) // 2, len(adm_snaps) - 1} if adm_snaps else []:
                        if 0 <= idx < len(adm_snaps):
                            adm_all.update(str(c) for c in adm_snaps[idx])

                    combined = set(walkthrough) | set(traj) | adm_all
                    for cmd in combined:
                        c_norm = " ".join(str(cmd).split())
                        pat = generalize(c_norm)
                        all_patterns[pat] += 1
                        if len(pattern_examples[pat]) < 5:
                            pattern_examples[pat].add(c_norm)

                    by_cell[label] = {
                        "cut": ops["cut"],
                        "cook": ops["cook"],
                        "walkthrough_len": len(walkthrough),
                        "policy_traj_len": len(traj),
                        "policy_err": err,
                        "walkthrough_patterns": sorted({generalize(c) for c in walkthrough}),
                        "policy_patterns": sorted({generalize(c) for c in traj}),
                        "admissible_pattern_count": len({generalize(c) for c in adm_all}),
                    }
    finally:
        shutil.rmtree(work, ignore_errors=True)

    observed = set(all_patterns.keys())
    observed_norm = {normalize_for_prompt_diff(p) for p in observed}
    current_set = set(CURRENT_PROMPT_TEMPLATES)
    missing = sorted(observed_norm - current_set)
    extra = sorted(current_set - observed_norm)

    report = {
        "source": "27 tw-cooking cells, 1 instance each, seed=42 grid",
        "cells_audited": len(by_cell),
        "policy_trajectory_errors": errors,
        "pattern_counts": dict(sorted(all_patterns.items(), key=lambda x: (-x[1], x[0]))),
        "pattern_examples": {k: sorted(v) for k, v in sorted(pattern_examples.items())},
        "current_prompt_templates": CURRENT_PROMPT_TEMPLATES,
        "missing_from_prompt": missing,
        "extra_in_prompt_not_observed_in_audit": extra,
        "by_cell": by_cell,
    }

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
