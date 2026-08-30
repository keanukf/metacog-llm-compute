#!/usr/bin/env python3
"""Within-episode progress by position and compute stage, for both domains.

Chapters 6 and 7 report what the compute stages do to episode success and to signal quality, but
nothing in the thesis shows what they do to *progress within* an episode. This script supplies that
view. The planning domain admits an exact definition: with four disks the puzzle has 81 reachable
configurations, so a breadth-first search from the goal gives the true minimum number of remaining
moves for every state the agent can be in. TextWorld has no such metric -- its environment score is
sparse and effectively terminal, awarded in the last tenth of an episode in the cases where it is
awarded at all -- so the analogous quantity there is the share of steps the judge codes optimal.
Both are averaged over episodes at each step index, giving one curve per compute stage.

Method:
  1. Parse the peg configuration out of each step's `observation_before`. Empty pegs render as
     "(empty)" rather than "[]", which is why the pattern accepts both.
  2. Precompute exact distances by BFS from the goal state over all 81 configurations.
  3. Average by (compute stage, step index), and record how many episodes are still running at
     each index so the survival structure is visible.

Caveat carried into the figure: episodes that solve terminate, so the population at later step
indices is increasingly the set of episodes that have not solved. The n curve is reported alongside
the means for exactly this reason; in this domain the effect is small, because most departures are
step-cap terminations rather than solutions.

Run from repo root:
  python scripts/phase1_analysis/toh_distance_to_goal.py
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOH_DIR = REPO_ROOT / "data/results/phase1/phase1_20260722_091125"
TW_DIR = REPO_ROOT / "data/results/phase1/textworld_regen_20260724"
OUT_JSON = REPO_ROOT / "data/results/phase1_analysis/progress_by_position.json"
OUT_FIG = REPO_ROOT / "data/results/phase1_analysis/figures/progress_by_position.png"
STAGES = ("C0", "C1", "C2")
N_DISKS = 4
GOAL = (2,) * N_DISKS  # every disk on peg C

PEG_RE = re.compile(r"Peg ([ABC]): (?:\[([^\]]*)\]|\(empty\))")


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


def main():
    dist = goal_distances()
    print(f"BFS over the state space: {len(dist)} configurations, maximum distance {max(dist.values())}")

    acc = {s: collections.defaultdict(list) for s in STAGES}
    unparsed = 0
    for path in sorted(glob.glob(str(TOH_DIR / "trace_ep_tower_of_hanoi_*.jsonl"))):
        m = re.search(r"_(C[012])_", path)
        if not m:
            continue
        stage = m.group(1)
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = parse_state(row.get("observation_before"))
                if state is None:
                    unparsed += 1
                    continue
                acc[stage][row["step_index"]].append(dist[state])

    tw = {s: collections.defaultdict(list) for s in STAGES}
    for path in sorted(glob.glob(str(TW_DIR / "trace_ep_textworld_*.jsonl"))):
        m = re.search(r"_(C[012])_", path)
        if not m:
            continue
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("correctness")
                if label is None:
                    continue
                tw[m.group(1)][row["step_index"]].append(1.0 if label == "optimal" else 0.0)

    max_step = max(k for s in STAGES for k in acc[s])
    report = {
        "n_disks": N_DISKS,
        "optimal_solution_length": max(dist.values()),
        "unparsed_steps": unparsed,
        "note": (
            "Mean minimum remaining moves to the goal, by compute stage and step index. Episodes "
            "that solve terminate, so n falls with step index and the later population is "
            "increasingly the unsolved one."
        ),
        "series_tower_of_hanoi": {},
        "series_textworld": {},
    }
    tw_max = max(k for s in STAGES for k in tw[s])
    for stage in STAGES:
        means, ns = [], []
        for k in range(max_step + 1):
            vals = acc[stage].get(k, [])
            means.append(sum(vals) / len(vals) if vals else None)
            ns.append(len(vals))
        report["series_tower_of_hanoi"][stage] = {"mean_distance": means, "n_episodes": ns}
        rates, tns = [], []
        for k in range(tw_max + 1):
            vals = tw[stage].get(k, [])
            rates.append(sum(vals) / len(vals) if vals else None)
            tns.append(len(vals))
        report["series_textworld"][stage] = {"optimal_rate": rates, "n_episodes": tns}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {OUT_JSON}  ({unparsed} steps unparseable)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; JSON written, figure skipped")
        return 0

    fig, axes = plt.subplots(
        2, 2, figsize=(9.6, 5.4), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
    )
    (ax_h, ax_t), (nx_h, nx_t) = axes
    styles = {"C0": ("#444444", ":"), "C1": ("#1f5fa8", "-"), "C2": ("#b03a2e", "--")}
    labels = {"C0": "C0  direct inference", "C1": "C1  single reasoning pass",
              "C2": "C2  self-consistency"}

    xs_h = list(range(max_step + 1))
    xs_t = list(range(tw_max + 1))
    for stage in STAGES:
        colour, ls = styles[stage]
        h = report["series_tower_of_hanoi"][stage]
        t = report["series_textworld"][stage]
        ax_h.plot(xs_h, h["mean_distance"], color=colour, linestyle=ls, linewidth=1.6,
                  label=labels[stage])
        nx_h.plot(xs_h, h["n_episodes"], color=colour, linestyle=ls, linewidth=1.1)
        ax_t.plot(xs_t, t["optimal_rate"], color=colour, linestyle=ls, linewidth=1.6)
        nx_t.plot(xs_t, t["n_episodes"], color=colour, linestyle=ls, linewidth=1.1)

    ax_h.axhline(max(dist.values()), color="#bbbbbb", linewidth=0.8)
    ax_h.set_title("Tower of Hanoi", fontsize=10, loc="left")
    ax_h.set_ylabel("Mean remaining moves to goal\n(grey line: the maximum, 15)")
    ax_h.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_t.set_title("TextWorld", fontsize=10, loc="left")
    ax_t.set_ylabel("Share of steps coded optimal")
    ax_t.set_ylim(0, 0.8)
    for a in (ax_h, ax_t, nx_h, nx_t):
        a.spines[["top", "right"]].set_visible(False)
    nx_h.set_ylabel("Episodes\nstill running", fontsize=8)
    for a in (nx_h, nx_t):
        a.set_xlabel("Step index within episode")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
