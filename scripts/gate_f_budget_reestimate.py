#!/usr/bin/env python3
"""Gate F budget re-estimate — Phase 1+2 GPU hours from real local data, no pod needed.

The original ~48h estimate (docs/instrument_validation_session.md, Gate F budget section) used
the C-5 pilot run (`instrument_validation/phase1_20260714_105004`, 72 episodes at the pre-Gate-D
max_steps_per_episode=20 cap) with a flat `wall_hours = total_episodes / measured_ep_per_hour`
formula. Since then, difficulty calibration lengthened both domains' real episode-length corridors
substantially (TextWorld cap 20->45; ToH's per-instance cap averages ~30 with the frozen 4-disk
manifest), which a flat ep/h figure can't reflect -- C1/C2 cost scales with *step count* (fresh
reasoning generation every step), not just episode count.

This instead builds a token-based model:

    hours = sum_over(domain, stage) [ n_episodes * tokens_per_episode(domain, stage) ] / tokens_per_sec / 3600

- ``tokens_per_sec`` (aggregate GPU+batching-engine throughput at N=32) is treated as a hardware
  constant, taken from the same C-5 run -- this shouldn't have changed (same GPU, model, engine).
- ``tokens_per_episode`` is read from the most current real local data available per (domain,
  stage); where no post-calibration real data exists yet (flagged explicitly), the C-5 pilot's
  figure is used as a documented placeholder, not silently assumed identical.

Phase 2 strategy stage-mix (src/agent/allocator.py ``STRATEGIES``): always_c0/always_c2 are pure;
random is a real uniform 1/3 C0+C1+C2 draw *per step* (not a proxy); adaptive_tle/adaptive_vc/
eager_style depend on Phase-1-derived thresholds unknown pre-hoc, so they're modeled the same as
`random` as a stated, direction-flagged approximation (an effective adaptive policy should cost
less on average by skipping C2 when unnecessary, so this likely overestimates Phase 2 cost).

Usage: python scripts/gate_f_budget_reestimate.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

TOKENS_PER_SEC = 1188.4780965202558  # C-5 105004, execution_metrics.tokens_per_sec, N=32
PILOT_RUN_DIR = REPO_ROOT / "data/results/instrument_validation/phase1_20260714_105004"
TW_CONFIRMATION = (
    REPO_ROOT
    / "data/results/gate_d_calibration/textworld_candidate_confirmation/validation_results.json"
)
TOH_CORRIDOR_DIR = REPO_ROOT / "data/results/gate_d_calibration/toh_corridor_scramble_n30"

PHASE1_EPISODES_PER_DOMAIN_STAGE = 250  # 50 instances x 5 runs
PHASE2_EPISODES_PER_DOMAIN_STRATEGY = 250  # 50 instances x 5 runs
PHASE2_STRATEGIES = (
    "always_c0",
    "always_c2",
    "random",
    "adaptive_tle",
    "adaptive_vc",
    "eager_style",
)


def _pilot_stage_stats() -> dict[tuple[str, str], dict[str, float]]:
    """Real per-(domain, stage) tokens/episode and steps/episode from the C-5 pilot (72 ep,
    old cap=20, num_disks_range [3,4] mixed). Used as the fallback anchor where no newer,
    post-calibration real data exists yet."""
    from collections import defaultdict

    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "tokens": 0.0, "steps": 0.0}
    )
    for f in PILOT_RUN_DIR.glob("ep_*.json"):
        d = json.loads(f.read_text())
        k = (d["domain"], d["compute_stage"])
        agg[k]["n"] += 1
        agg[k]["tokens"] += d["total_tokens_generated"]
        agg[k]["steps"] += d["episode_length_steps"]
    out = {}
    for k, v in agg.items():
        out[k] = {
            "tokens_per_ep": v["tokens"] / v["n"],
            "steps_per_ep": v["steps"] / v["n"],
            "n": v["n"],
        }
    return out


def _textworld_c0_current() -> dict[str, float]:
    """Real current C0 data at the frozen corridor (r5_i1_take+cook, cap 45, n=16)."""
    data = json.loads(TW_CONFIRMATION.read_text())
    for v in data["validations"]:
        if v["combo"]["operations"] == "take+cook":
            m = v["metrics"]
            return {
                "steps_per_ep": m["mean_episode_length_all"],
                "n": m["num_instances"],
                "source": "real, cap=45, frozen corridor",
            }
    raise RuntimeError("take+cook cell not found in textworld_candidate_confirmation")


def _toh_4disk_stage_stats() -> dict[str, dict[str, float]]:
    """Real current C0/C1 data at the frozen 4-disk corridor (per-instance 3x optimal_steps cap,
    same mechanism build_toh_manifest.py / generate_instances() actually implements)."""
    from collections import defaultdict

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"steps": [], "tokens": []})
    for f in TOH_CORRIDOR_DIR.glob("toh_diversity_*.json"):
        m = re.search(r"toh_diversity_(C\d)_", f.name)
        if not m:
            continue
        stage = m.group(1)
        d = json.loads(f.read_text())
        nd = None
        for step in d.get("steps_detail") or []:
            text = step.get("vc_prompt") or step.get("reason_prompt") or ""
            dm = re.search(r"holds all (\d+) disks", text)
            if dm:
                nd = int(dm.group(1))
                break
        if nd != 4:
            continue
        agg[stage]["steps"].append(d["episode_length_steps"])
        agg[stage]["tokens"].append(d["total_tokens_generated"])
    out = {}
    for stage, v in agg.items():
        n = len(v["steps"])
        out[stage] = {
            "steps_per_ep": sum(v["steps"]) / n,
            "tokens_per_ep": sum(v["tokens"]) / n,
            "n": n,
        }
    return out


def build_tokens_per_episode_table() -> dict[tuple[str, str], dict[str, Any]]:
    pilot = _pilot_stage_stats()
    tw_c0 = _textworld_c0_current()
    toh = _toh_4disk_stage_stats()

    table: dict[tuple[str, str], dict[str, Any]] = {}

    # TextWorld C0: real current tokens/step (pilot) x real current steps/ep (confirmation run).
    tw_c0_tokens_per_step = (
        pilot[("textworld", "C0")]["tokens_per_ep"] / pilot[("textworld", "C0")]["steps_per_ep"]
    )
    table[("textworld", "C0")] = {
        "tokens_per_ep": tw_c0_tokens_per_step * tw_c0["steps_per_ep"],
        "steps_per_ep": tw_c0["steps_per_ep"],
        "basis": "real steps (n=16, cap=45, frozen corridor) x real tokens/step (C-5 pilot)",
    }
    # TextWorld C1/C2: NO real post-calibration data exists (Gate D only swept C0, the reference
    # stage). Flagged assumption: same step-count as the real C0 anchor above (neutral, unverified
    # -- C1/C2 success rates are known to differ substantially from C0's, so real behavior could
    # be shorter (finishing before the cap) or similarly cap-bound; direction is genuinely unknown).
    for stage in ("C1", "C2"):
        tokens_per_step = (
            pilot[("textworld", stage)]["tokens_per_ep"]
            / pilot[("textworld", stage)]["steps_per_ep"]
        )
        table[("textworld", stage)] = {
            "tokens_per_ep": tokens_per_step * tw_c0["steps_per_ep"],
            "steps_per_ep": tw_c0["steps_per_ep"],
            "basis": f"ASSUMED steps=C0's real value (no real {stage} data at cap=45 yet) x real tokens/step (C-5 pilot)",
        }

    # ToH C0/C1: real current data at the frozen 4-disk corridor, per-instance 3x optimal cap.
    for stage in ("C0", "C1"):
        table[("tower_of_hanoi", stage)] = {
            "tokens_per_ep": toh[stage]["tokens_per_ep"],
            "steps_per_ep": toh[stage]["steps_per_ep"],
            "basis": f"real (n={toh[stage]['n']}, 4-disk, per-instance 3x-optimal cap, same n=30 run that informed the freeze decision)",
        }
    # ToH C2: no real 4-disk data. Flagged assumption: scale the C-5 pilot's C2 figure by the
    # ratio of real-current to pilot ToH episode length (pilot used num_disks [3,4] mixed at the
    # old flat cap=20; current uses 4-disk only at the real, longer per-instance cap).
    pilot_toh_c2 = pilot[("tower_of_hanoi", "C2")]
    length_ratio = toh["C1"]["steps_per_ep"] / pilot_toh_c2["steps_per_ep"]
    table[("tower_of_hanoi", "C2")] = {
        "tokens_per_ep": pilot_toh_c2["tokens_per_ep"] * length_ratio,
        "steps_per_ep": toh["C1"]["steps_per_ep"],
        "basis": (
            f"ASSUMED: no real 4-disk C2 data -- C-5 pilot's C2 tokens/ep ({pilot_toh_c2['tokens_per_ep']:.0f}) "
            f"scaled by the real C1 vs. pilot-C2 step-length ratio ({length_ratio:.2f}x)"
        ),
    }
    return table


def phase1_episode_counts() -> dict[tuple[str, str], int]:
    counts = {}
    for domain in ("textworld", "tower_of_hanoi"):
        for stage in ("C0", "C1", "C2"):
            counts[(domain, stage)] = PHASE1_EPISODES_PER_DOMAIN_STAGE
    return counts


def phase2_episode_counts() -> dict[tuple[str, str], float]:
    """Effective (domain, stage) episode counts under the stage-mix model described in the
    module docstring."""
    counts: dict[tuple[str, str], float] = {
        (d, s): 0.0 for d in ("textworld", "tower_of_hanoi") for s in ("C0", "C1", "C2")
    }
    for domain in ("textworld", "tower_of_hanoi"):
        for strategy in PHASE2_STRATEGIES:
            n = PHASE2_EPISODES_PER_DOMAIN_STRATEGY
            if strategy == "always_c0":
                counts[(domain, "C0")] += n
            elif strategy == "always_c2":
                counts[(domain, "C2")] += n
            else:  # random, adaptive_tle, adaptive_vc, eager_style -- uniform-mix proxy
                counts[(domain, "C0")] += n / 3
                counts[(domain, "C1")] += n / 3
                counts[(domain, "C2")] += n / 3
    return counts


def main() -> None:
    table = build_tokens_per_episode_table()

    print("Tokens/episode table (real where noted, flagged assumption otherwise):\n")
    for (domain, stage), row in sorted(table.items()):
        flag = "REAL" if "real" in row["basis"] and "ASSUMED" not in row["basis"] else "ASSUMED"
        print(
            f"  [{flag}] {domain}/{stage}: {row['tokens_per_ep']:,.0f} tok/ep, {row['steps_per_ep']:.1f} steps/ep"
        )
        print(f"           basis: {row['basis']}")

    def total_tokens(counts: dict[tuple[str, str], float]) -> float:
        return sum(n * table[k]["tokens_per_ep"] for k, n in counts.items())

    p1_counts = phase1_episode_counts()
    p2_counts = phase2_episode_counts()
    p1_tokens = total_tokens(p1_counts)
    p2_tokens = total_tokens(p2_counts)
    p1_hours = p1_tokens / TOKENS_PER_SEC / 3600
    p2_hours = p2_tokens / TOKENS_PER_SEC / 3600

    print(f"\nPhase 1: {p1_tokens:,.0f} tokens -> {p1_hours:.1f} h")
    print(f"Phase 2: {p2_tokens:,.0f} tokens -> {p2_hours:.1f} h")
    print(f"Total:   {p1_tokens + p2_tokens:,.0f} tokens -> {p1_hours + p2_hours:.1f} h")
    print(
        f"\n(tokens_per_sec={TOKENS_PER_SEC:.1f}, from C-5 105004, N=32 -- treated as a hardware constant)"
    )

    out = {
        "tokens_per_sec": TOKENS_PER_SEC,
        "tokens_per_episode_table": {f"{k[0]}/{k[1]}": v for k, v in table.items()},
        "phase1_tokens": p1_tokens,
        "phase2_tokens": p2_tokens,
        "phase1_hours": p1_hours,
        "phase2_hours": p2_hours,
        "total_hours": p1_hours + p2_hours,
    }
    out_path = REPO_ROOT / "data/results/gate_f_budget_reestimate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
