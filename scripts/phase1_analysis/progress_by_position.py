#!/usr/bin/env python3
"""Within-episode progress by position and compute stage, for both domains.

Chapters 6 and 7 report what the compute stages do to episode success and to signal quality, but
nothing in the thesis shows what they do to *progress within* an episode. This script supplies that
view, one curve per compute stage, both domains side by side.

Tower of Hanoi admits an exact progress metric. With four disks the puzzle has 81 reachable
configurations, so a breadth-first search from the goal gives the true minimum number of remaining
moves for every state the agent can occupy. The peg configuration is parsed out of each step's
`observation_before`; empty pegs render as "(empty)" rather than "[]", which is why the pattern
accepts both.

Two design decisions matter for how the curves may be read.

*Carry-forward, not survivor means.* A naive mean over the episodes still running at step k is
badly confounded here, because the step budget is instance-specific: `tower_of_hanoi.py` sets it to
three times the instance's optimal solution length, so easy instances (short solutions, states that
start close to the goal) exhaust their budget first and drop out of the average early. Measured on
this dataset, the mean *starting* distance of the surviving episodes climbs from 10.20 at step 0 to
15.00 at step 44. A survivor mean therefore rises with step index even when every episode is moving
toward the goal, which reverses the qualitative reading of the figure. This script instead keeps
every episode in the average for the whole x range and carries its final value forward: an episode
that solves contributes 0 from then on, an episode that exhausts its budget contributes the
distance it stopped at. The denominator is a constant 250 episodes per stage at every step index,
so the curves are population means and the three stages start from an identical mean distance of
10.20.

*The two domains are not put on one metric.* TextWorld records a reference walkthrough length per
instance, so remaining distance looks recoverable as walkthrough length minus the number of steps
the judge coded optimal. It is not: in 119 of 750 episodes (16 %) the optimal count exceeds the
walkthrough length outright. The judge scores each step against a shortest path *from the state the
agent is currently in*, so a detour followed by a recovery accumulates optimal steps without net
progress along the reference solution. That is the right analogue of a Hanoi optimal move, but it
does not compose into a distance. TextWorld is therefore reported as the cumulative count of
optimal actions taken, carried forward on the same rule, against the median walkthrough length as a
reference level. Both panels read "further along is better"; the axes are deliberately labelled as
different quantities.

The lower row gives the cumulative share of episodes solved by each step index, which is what makes
the stage separation concrete: in TextWorld the C0 curve stops rising at .24 around step 25 while
C1 and C2 continue to .67 and .70.

Run from repo root:
  python scripts/phase1_analysis/progress_by_position.py
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

TOH_DIR = REPO_ROOT / "data/results/phase1/phase1_20260722_091125"
TW_DIR = REPO_ROOT / "data/results/phase1/textworld_regen_20260724"
TW_MANIFEST = REPO_ROOT / "data/tasks/textworld/difficulty_manifest.json"
OUT_JSON = REPO_ROOT / "data/results/phase1_analysis/progress_by_position.json"
OUT_FIG = REPO_ROOT / "data/results/phase1_analysis/figures/progress_by_position.png"
STAGES = ("C0", "C1", "C2")
N_DISKS = 4
GOAL = (2,) * N_DISKS  # every disk on peg C

PEG_RE = re.compile(r"Peg ([ABC]): (?:\[([^\]]*)\]|\(empty\))")
STAGE_RE = re.compile(r"_(C[012])_")
TW_INSTANCE_RE = re.compile(r"ep_textworld_(\d+)_")


def parse_state(observation):
    """Peg listing -> tuple giving each disk's peg index, or None if not parseable."""
    pegs = {}
    for m in PEG_RE.finditer(observation or ""):
        pegs[m.group(1)] = tuple(int(x) for x in (m.group(2) or "").split(",") if x.strip())
    if len(pegs) != 3:
        return None
    loc = {}
    for peg_index, peg in enumerate("ABC"):
        for disk in pegs[peg]:
            loc[disk] = peg_index
    if len(loc) != N_DISKS:
        return None
    return tuple(loc[d] for d in range(1, N_DISKS + 1))


def legal_moves(state):
    pegs = [[] for _ in range(3)]
    for disk in range(N_DISKS, 0, -1):
        pegs[state[disk - 1]].append(disk)
    out = []
    for src in range(3):
        if not pegs[src]:
            continue
        top = pegs[src][-1]
        for dst in range(3):
            if src == dst:
                continue
            if pegs[dst] and pegs[dst][-1] < top:
                continue
            nxt = list(state)
            nxt[top - 1] = dst
            out.append(tuple(nxt))
    return out


def goal_distances():
    dist = {GOAL: 0}
    queue = collections.deque([GOAL])
    while queue:
        state = queue.popleft()
        for nxt in legal_moves(state):
            if nxt not in dist:
                dist[nxt] = dist[state] + 1
                queue.append(nxt)
    return dist


def episode_json(trace_path):
    return json.load(open(trace_path.replace("trace_ep_", "ep_").replace(".jsonl", ".json")))


def read_trace(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def carry_forward(series, n_points):
    """Value at each step index, holding the last observed value once the episode has ended."""
    return [series[k] if k < len(series) else series[-1] for k in range(n_points)]


def collect_hanoi(dist):
    """Per-episode distance series, plus solve step, keyed by compute stage."""
    out = {s: [] for s in STAGES}
    unparsed = 0
    for path in sorted(glob.glob(str(TOH_DIR / "trace_ep_tower_of_hanoi_*.jsonl"))):
        m = STAGE_RE.search(path)
        if not m:
            continue
        rows = read_trace(path)
        series = []
        for row in rows:
            state = parse_state(row.get("observation_before"))
            if state is None:
                unparsed += 1
                continue
            series.append(dist[state])
        if not series:
            continue
        solved = bool(episode_json(path)["task_success"])
        # The distance after the final move is 0 exactly when the episode solved, which the trace
        # records only in `observation_after`; appending it keeps the terminal state in the series.
        series.append(0 if solved else series[-1])
        out[m.group(1)].append({"series": series, "solve_step": len(rows) if solved else None})
    return out, unparsed


def collect_textworld(walkthrough):
    """Per-episode cumulative optimal-action counts, plus solve step, keyed by compute stage."""
    out = {s: [] for s in STAGES}
    for path in sorted(glob.glob(str(TW_DIR / "trace_ep_textworld_*.jsonl"))):
        m = STAGE_RE.search(path)
        if not m:
            continue
        rows = read_trace(path)
        if not rows:
            continue
        running, series = 0, []
        for row in rows:
            running += 1 if row.get("correctness") == "optimal" else 0
            series.append(running)
        solved = bool(episode_json(path)["task_success"])
        out[m.group(1)].append({"series": series, "solve_step": len(rows) if solved else None})
    return out


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
    dist = goal_distances()
    max_distance = max(dist.values())
    print(f"BFS over the state space: {len(dist)} configurations, maximum distance {max_distance}")

    hanoi, unparsed = collect_hanoi(dist)
    entries = json.load(open(TW_MANIFEST))["entries"]
    walkthrough = {e["instance_id"]: e["walkthrough_length"] for e in entries}
    median_walkthrough = stats.median(walkthrough.values())
    textworld = collect_textworld(walkthrough)

    n_h = max(len(e["series"]) for s in STAGES for e in hanoi[s])
    n_t = max(len(e["series"]) for s in STAGES for e in textworld[s])

    report = {
        "n_disks": N_DISKS,
        "max_goal_distance": max_distance,
        "unparsed_steps": unparsed,
        "median_walkthrough_length": median_walkthrough,
        "aggregation": (
            "Population mean with the final value carried forward, so all episodes of a stage stay "
            "in the denominator at every step index. A survivor mean is confounded here because "
            "the Hanoi step budget is three times the instance optimal length, which retires easy "
            "instances first."
        ),
        "textworld_metric_note": (
            "Cumulative optimal actions, not remaining distance: the judge scores optimality from "
            "the agent's current state, so the count exceeds the reference walkthrough length in "
            "16 % of episodes and does not compose into a distance."
        ),
        "series_tower_of_hanoi": {},
        "series_textworld": {},
    }

    for stage in STAGES:
        means, solved = summarise(hanoi[stage], n_h)
        report["series_tower_of_hanoi"][stage] = {
            "n_episodes": len(hanoi[stage]),
            "mean_remaining_moves": means,
            "cumulative_solved_share": solved,
        }
        means, solved = summarise(textworld[stage], n_t)
        report["series_textworld"][stage] = {
            "n_episodes": len(textworld[stage]),
            "mean_cumulative_optimal": means,
            "cumulative_solved_share": solved,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {OUT_JSON}  ({unparsed} steps unparseable)")

    for stage in STAGES:
        h = report["series_tower_of_hanoi"][stage]
        t = report["series_textworld"][stage]
        print(
            f"  {stage}: Hanoi {h['mean_remaining_moves'][0]:.2f} -> "
            f"{h['mean_remaining_moves'][-1]:.2f} moves remaining, solved "
            f"{h['cumulative_solved_share'][-1]:.2f}   |   TextWorld "
            f"{t['mean_cumulative_optimal'][-1]:.2f} optimal actions, solved "
            f"{t['cumulative_solved_share'][-1]:.2f}"
        )

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
    (ax_h, ax_t), (sx_h, sx_t) = axes
    styles = {"C0": ("#444444", ":"), "C1": ("#1f5fa8", "-"), "C2": ("#b03a2e", "--")}
    labels = {"C0": "C0  direct inference", "C1": "C1  single reasoning pass",
              "C2": "C2  self-consistency"}

    xs_h, xs_t = list(range(n_h)), list(range(n_t))
    for stage in STAGES:
        colour, ls = styles[stage]
        h = report["series_tower_of_hanoi"][stage]
        t = report["series_textworld"][stage]
        ax_h.plot(xs_h, h["mean_remaining_moves"], color=colour, linestyle=ls, linewidth=1.7,
                  label=labels[stage])
        ax_t.plot(xs_t, t["mean_cumulative_optimal"], color=colour, linestyle=ls, linewidth=1.7)
        sx_h.plot(xs_h, h["cumulative_solved_share"], color=colour, linestyle=ls, linewidth=1.4)
        sx_t.plot(xs_t, t["cumulative_solved_share"], color=colour, linestyle=ls, linewidth=1.4)

    ax_h.set_title("Tower of Hanoi", fontsize=10, loc="left")
    ax_h.set_ylabel("Mean remaining moves to goal")
    ax_h.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax_t.set_title("TextWorld", fontsize=10, loc="left")
    ax_t.set_ylabel("Mean cumulative optimal actions")
    ax_t.axhline(median_walkthrough, color="#bbbbbb", linewidth=0.8)
    ax_t.set_ylim(0, median_walkthrough + 1.4)
    ax_t.annotate(
        f"median reference solution, {median_walkthrough:.0f} actions",
        xy=(0.02, median_walkthrough), xycoords=("axes fraction", "data"),
        ha="left", va="bottom", fontsize=7.5, color="#777777",
    )
    for a in (sx_h, sx_t):
        a.set_ylabel("Share solved", fontsize=8)
        a.set_ylim(0, 0.75)
        a.set_xlabel("Step index within episode")
    for a in (ax_h, ax_t, sx_h, sx_t):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
