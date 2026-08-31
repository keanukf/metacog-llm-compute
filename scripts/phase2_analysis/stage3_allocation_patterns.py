#!/usr/bin/env python3
"""Allocation-pattern analysis for thesis §7.4 (deep-check C1), plus the total-token axis (C4).

Chapter 7 and Chapter 8 both state that the compute-stage mix the allocator actually chose cannot
be reported. It can: every Phase 2 adaptive episode logs ``stage_per_step`` and, per step,
``allocator_uncertainty_score`` with the deployed ``allocator_theta1``/``allocator_theta2``
(docs/artifact_schema.md, recorded expressly for this analysis). This script reads those fields
directly from the collected episode records and reports, per policy and domain:

  1. the realised C0/C1/C2 share (the "when does the agent choose to deliberate" question);
  2. step-level correctness conditional on the stage the allocator chose;
  3. the escalation rate across normalised episode position;
  4. the distribution of the allocator's uncertainty score against its deployed thresholds;
  5. total tokens (prompt + generated), the secondary cost axis promised in §5.3 (C4).

What genuinely remains uncollected is only the EAGer-Style / Random-Alloc comparison, which needs
baseline conditions that were never run.

Read-only, no GPU, no bootstrap. Run from repo root:
  python scripts/phase2_analysis/stage3_allocation_patterns.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES,
)

STAGES = ("C0", "C1", "C2")
POLICIES = ("adaptive_tle", "adaptive_vc")
DOMS = ("textworld", "tower_of_hanoi")
def _keep(e: dict) -> bool:
    """Non-holdout filter matching the canonical pipeline: the Tower of Hanoi uses each episode's
    embedded ``holdout`` flag, while TextWorld uses the explicit confirmatory exclusion set,
    because it is exactly that embedded flag which the holdout-labelling incident corrupted."""
    dom = str(e.get("domain"))
    if dom == "textworld":
        return int(e.get("instance", -1)) not in TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES
    return not bool(e.get("holdout"))


def _load(checkpoint: Path) -> list[dict]:
    eps = []
    for f in sorted(checkpoint.rglob("*.json")):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        if not isinstance(o, dict) or o.get("domain") is None:
            continue
        eps.append(o)
    return eps


def _adaptive(eps: list[dict], dom: str, pol: str) -> list[dict]:
    return [
        e
        for e in eps
        if str(e.get("domain")) == dom and str(e.get("strategy")) == pol and _keep(e)
    ]


def stage_mix_and_outcomes(eps: list[dict]) -> None:
    print("\n===== 1+2  Realised stage mix and correctness by chosen stage =====")
    for dom in DOMS:
        for pol in POLICIES:
            rows = _adaptive(eps, dom, pol)
            if not rows:
                print(f"  {dom}/{pol}: no episodes")
                continue
            mix = Counter()
            correct = defaultdict(lambda: [0, 0])  # stage -> [n_optimal, n_total]
            for e in rows:
                for sd in e.get("steps_detail", []):
                    s = str(sd.get("compute_stage"))
                    if s not in STAGES:
                        continue
                    mix[s] += 1
                    c = sd.get("correctness")
                    if c in ("optimal", "legal", "illegal"):
                        correct[s][1] += 1
                        if c == "optimal":
                            correct[s][0] += 1
            tot = sum(mix.values())
            inst = len({e.get("instance") for e in rows})
            share = "  ".join(f"{s}={mix[s] / tot:.3f}" for s in STAGES)
            print(f"\n  {dom}/{pol}: episodes={len(rows)} instances={inst} steps={tot}")
            print(f"    realised stage share:  {share}")
            for s in STAGES:
                n_opt, n_tot = correct[s]
                if n_tot:
                    print(
                        f"      {s}: n={n_tot:6d}  optimal_rate={n_opt / n_tot:.3f}"
                    )


def escalation_by_position(eps: list[dict], n_bins: int = 4) -> None:
    print("\n===== 3  Escalation rate across normalised episode position =====")
    for dom in DOMS:
        for pol in POLICIES:
            rows = _adaptive(eps, dom, pol)
            if not rows:
                continue
            bins = defaultdict(lambda: [0, 0])  # bin -> [n_escalated, n_total]
            for e in rows:
                sds = e.get("steps_detail", [])
                n = len(sds)
                if n < 2:
                    continue
                for sd in sds:
                    i = int(sd.get("step_index", 0))
                    pos = i / max(n - 1, 1)
                    b = min(int(pos * n_bins), n_bins - 1)
                    s = str(sd.get("compute_stage"))
                    if s not in STAGES:
                        continue
                    bins[b][1] += 1
                    if s in ("C1", "C2"):
                        bins[b][0] += 1
            out = "  ".join(
                f"Q{b + 1}={bins[b][0] / bins[b][1]:.3f}" for b in range(n_bins) if bins[b][1]
            )
            print(f"  {dom:15s} {pol:13s} escalation rate by position quartile:  {out}")


def uncertainty_scores(eps: list[dict]) -> None:
    print("\n===== 4  Allocator uncertainty score vs deployed thresholds =====")
    for dom in DOMS:
        for pol in POLICIES:
            rows = _adaptive(eps, dom, pol)
            if not rows:
                continue
            scores, th1, th2 = [], set(), set()
            for e in rows:
                for sd in e.get("steps_detail", []):
                    v = sd.get("allocator_uncertainty_score")
                    if v is None:
                        continue
                    scores.append(float(v))
                    if sd.get("allocator_theta1") is not None:
                        th1.add(round(float(sd["allocator_theta1"]), 3))
                        th2.add(round(float(sd["allocator_theta2"]), 3))
            if not scores:
                continue
            scores.sort()
            q = lambda p: scores[min(int(p * len(scores)), len(scores) - 1)]  # noqa: E731
            print(
                f"  {dom:15s} {pol:13s} n={len(scores):6d}  theta1={sorted(th1)} theta2={sorted(th2)}"
            )
            print(
                f"      mean={st.mean(scores):.3f}  median={q(0.50):.3f}  "
                f"p10={q(0.10):.3f}  p90={q(0.90):.3f}"
            )


def total_tokens(eps: list[dict]) -> None:
    """Secondary cost axis (C4), on the SAME paired instance set Table 7.2 uses: instances with
    data for both the adaptive policy and Always-C2 in that domain."""
    print("\n===== 5  Total tokens (prompt + generated), secondary cost axis (C4) =====")
    print("     [paired instance set per policy: instances present for both policy and Always-C2]")
    for dom in DOMS:
        dom_eps = [e for e in eps if str(e.get("domain")) == dom and _keep(e)]
        by_inst: dict[int, set[str]] = defaultdict(set)
        for e in dom_eps:
            by_inst[int(e.get("instance", -1))].add(str(e.get("strategy")))
        for pol in POLICIES:
            paired = {i for i, s in by_inst.items() if pol in s and "always_c2" in s}
            if not paired:
                continue
            print(f"\n  {dom} / {pol}  (paired instances={len(paired)})")
            means = {}
            for arm in (pol, "always_c2"):
                rows = [
                    e
                    for e in dom_eps
                    if str(e.get("strategy")) == arm
                    and int(e.get("instance", -1)) in paired
                ]
                # instance-level averaging: mean within instance, then mean over instances --
                # the convention Tables 7.2/7.3 use (verified to reproduce them exactly).
                by_i: dict[int, list[tuple[float, float]]] = defaultdict(list)
                for e in rows:
                    by_i[int(e["instance"])].append(
                        (
                            float(e.get("total_tokens_generated") or 0),
                            float(e.get("total_prompt_tokens") or 0),
                        )
                    )
                g = st.mean([st.mean([x[0] for x in v]) for v in by_i.values()])
                p = st.mean([st.mean([x[1] for x in v]) for v in by_i.values()])
                means[arm] = (g, p, g + p)
                print(
                    f"    {arm:13s} eps={len(rows):4d}  mean_output={g:10,.0f}  "
                    f"mean_prompt={p:11,.0f}  mean_total={g + p:11,.0f}"
                )
            ao, _, at = means[pol]
            bo, _, bt = means["always_c2"]
            print(
                f"    -> Always-C2 / policy ratio:  output={bo / ao:6.1f}x   total={bt / at:5.1f}x"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir",
        default="data/results/phase2/phase2_stage1_20260805_UTC",
    )
    args = ap.parse_args()
    cp = Path(args.checkpoint_dir)
    if not cp.is_absolute():
        cp = REPO_ROOT / cp
    eps = _load(cp)
    print(f"loaded {len(eps)} Phase 2 episode records from {cp}")
    stage_mix_and_outcomes(eps)
    escalation_by_position(eps)
    uncertainty_scores(eps)
    total_tokens(eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
