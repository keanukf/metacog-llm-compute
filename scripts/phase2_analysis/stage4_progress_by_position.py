#!/usr/bin/env python3
"""Within-episode progress by position and allocation policy, for both domains.

The Phase 1 counterpart (`scripts/phase1_analysis/progress_by_position.py`, thesis Figure 6.1)
shows what the fixed compute stages do to progress within an episode. This script is its Phase 2
analogue: the same two-row view with one curve per allocation policy, so the adaptive policies can
be read against the Always-C2 baseline as trajectories rather than as the endpoint summaries of
Table 7.1.

Metric. Phase 2 recorded episode files but no step traces, so the exact Tower of Hanoi
breadth-first distance the Phase 1 figure uses is not reconstructible here. Both domains are
therefore reported on the quantity that *is* available in both, the cumulative count of steps coded
optimal, against the median reference solution length of the included instances. That has an
incidental benefit over the Phase 1 figure: the two panels carry the same unit, so the domains may
be compared directly.

Aggregation follows the Phase 1 figure exactly. Each curve is a population mean over all episodes
of its policy with the final value carried forward once an episode ends, so the denominator is
constant across the axis and a policy that solves early is not silently dropped from the average.

Instance set. The curves are drawn on the same episodes the H2 test of Section 7.1 runs on, so the
figure and Table 7.1 describe one population: non-holdout instances carrying data for all three
arms. For TextWorld that applies `TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES`, the union of the true
holdout with the four instances the deployed thresholds were mistakenly fitted on, leaving 41
instances. For the Tower of Hanoi the Always-C2 collection was aborted at 12 of 50 instances, so
that arm bounds the paired set at 12 and the domain is exploratory throughout Chapter 7.

Run from repo root:
  python scripts/phase2_analysis/stage4_progress_by_position.py
"""
from __future__ import annotations

import collections
import glob
import json
import re
import statistics as stats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES,
)

EP_DIR = REPO_ROOT / "data/results/phase2/phase2_stage1_20260805_UTC"
TW_MANIFEST = REPO_ROOT / "data/tasks/textworld/difficulty_manifest.json"
TOH_MANIFEST = REPO_ROOT / "data/tasks/tower_of_hanoi/difficulty_manifest.json"
OUT_JSON = REPO_ROOT / "data/results/phase2_analysis/progress_by_position_phase2.json"
OUT_FIG = REPO_ROOT / "data/results/phase2_analysis/figures/progress_by_position_phase2.png"

DOMAINS = ("tower_of_hanoi", "textworld")
POLICIES = ("always_c2", "adaptive_tle", "adaptive_vc")
EP_RE = re.compile(r"ep_(textworld|tower_of_hanoi)_(\d+)_(.+)_(\d+)\.json$")


def load_episodes():
    """Every Phase 2 episode, keyed by domain, grouped by policy, with instance and holdout kept."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(str(EP_DIR / "ep_*.json"))):
        m = EP_RE.search(path)
        if not m:
            continue
        domain, instance, policy = m.group(1), int(m.group(2)), m.group(3)
        if policy not in POLICIES:
            continue
        episode = json.load(open(path))
        holdout = bool(episode.get("holdout"))
        # The TextWorld holdout field encodes the wrong split; the confirmatory exclusion set is
        # the authority, exactly as in scripts/phase2_analysis/stage1_h2_adaptive_allocation.py.
        if domain == "textworld":
            holdout = instance in TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES
        out[domain][policy].append(
            {
                "instance": instance,
                "holdout": holdout,
                "solved": bool(episode.get("task_success")),
                "steps": episode.get("steps_detail") or [],
            }
        )
    return out


def paired_instances(by_policy):
    """Non-holdout instances carrying episodes for all three arms, as the H2 test pairs them."""
    sets = []
    for policy in POLICIES:
        sets.append({e["instance"] for e in by_policy[policy] if not e["holdout"]})
    return set.intersection(*sets)


def cumulative_optimal(episode):
    """Running count of steps coded optimal, one entry per step."""
    running, series = 0, []
    for step in episode["steps"]:
        running += 1 if step.get("correctness") == "optimal" else 0
        series.append(running)
    return series


def summarise(episodes, n_points):
    """Population mean of the carried-forward series and the cumulative solved share."""
    means, solved_share = [], []
    n = len(episodes)
    for k in range(n_points):
        vals = [e["series"][k] if k < len(e["series"]) else e["series"][-1] for e in episodes]
        means.append(sum(vals) / n)
        solved_share.append(
            sum(1 for e in episodes if e["solve_step"] is not None and e["solve_step"] <= k + 1) / n
        )
    return means, solved_share


def main():
    raw = load_episodes()
    tw_ref = {e["instance_id"]: e["walkthrough_length"]
              for e in json.load(open(TW_MANIFEST))["entries"]}
    toh_ref = {e["instance_id"]: e["optimal_steps"]
               for e in json.load(open(TOH_MANIFEST))["entries"]}
    reference = {"textworld": tw_ref, "tower_of_hanoi": toh_ref}

    report = {
        "metric": (
            "Cumulative count of steps coded optimal, by step index. Phase 2 wrote no step traces, "
            "so the exact Tower of Hanoi goal distance of the Phase 1 figure is unavailable and "
            "both domains use the one quantity present in both."
        ),
        "aggregation": (
            "Population mean with each episode's final value carried forward once it ends, so the "
            "denominator is constant across the axis."
        ),
        "instance_set": (
            "Non-holdout instances carrying all three arms, the set the H2 test of Section 7.1 "
            "pairs on."
        ),
        "domains": {},
    }

    print("Phase 2 within-episode progress\n")
    for domain in DOMAINS:
        keep = paired_instances(raw[domain])
        median_reference = stats.median(reference[domain][i] for i in sorted(keep))
        series = {}
        for policy in POLICIES:
            eps = []
            for e in raw[domain][policy]:
                if e["instance"] not in keep:
                    continue
                s = cumulative_optimal(e)
                if not s:
                    continue
                eps.append({"series": s, "solve_step": len(s) if e["solved"] else None})
            series[policy] = eps
        n_points = max(len(e["series"]) for eps in series.values() for e in eps)

        entry = {
            "n_instances": len(keep),
            "median_reference_solution": median_reference,
            "policies": {},
        }
        print(f"  {domain}: {len(keep)} instances, median reference solution "
              f"{median_reference:g} steps")
        for policy in POLICIES:
            means, solved = summarise(series[policy], n_points)
            entry["policies"][policy] = {
                "n_episodes": len(series[policy]),
                "mean_cumulative_optimal": means,
                "cumulative_solved_share": solved,
            }
            print(f"    {policy:14s} n={len(series[policy]):4d}  "
                  f"optimal actions {means[-1]:5.2f}  solved {solved[-1]:.3f}")
        report["domains"][domain] = entry

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; JSON written, figure skipped")
        return 0

    fig, axes = plt.subplots(
        2, 2, figsize=(9.6, 5.6), sharex="col", gridspec_kw={"height_ratios": [3, 2]}
    )
    styles = {"always_c2": ("#b03a2e", "--"), "adaptive_tle": ("#1f5fa8", "-"),
              "adaptive_vc": ("#444444", ":")}
    labels = {"always_c2": "Always-C2  baseline", "adaptive_tle": r"$\pi_{TLE}$  entropy policy",
              "adaptive_vc": r"$\pi_{VC}$  confidence policy"}
    titles = {"tower_of_hanoi": "Tower of Hanoi", "textworld": "TextWorld"}

    for col, domain in enumerate(DOMAINS):
        entry = report["domains"][domain]
        top, bottom = axes[0][col], axes[1][col]
        n_points = len(entry["policies"]["always_c2"]["mean_cumulative_optimal"])
        xs = list(range(n_points))
        for policy in POLICIES:
            colour, ls = styles[policy]
            p = entry["policies"][policy]
            top.plot(xs, p["mean_cumulative_optimal"], color=colour, linestyle=ls, linewidth=1.7,
                     label=labels[policy] if col == 0 else None)
            bottom.plot(xs, p["cumulative_solved_share"], color=colour, linestyle=ls,
                        linewidth=1.4)
        ref = entry["median_reference_solution"]
        top.axhline(ref, color="#bbbbbb", linewidth=0.8)
        ref_label = f"{ref:g}"  # 7.5 must not print as 8
        top.annotate(f"median reference solution, {ref_label} steps",
                     xy=(0.02, ref), xycoords=("axes fraction", "data"),
                     ha="left", va="bottom", fontsize=7.5, color="#777777")
        top.set_title(f"{titles[domain]}  ({entry['n_instances']} instances)",
                      fontsize=10, loc="left")
        top.set_ylim(0, ref + 2.0)
        bottom.set_xlabel("Step index within episode")
        for a in (top, bottom):
            a.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Mean cumulative optimal actions")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[1][0].set_ylabel("Share solved", fontsize=8)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
