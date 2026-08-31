#!/usr/bin/env python3
"""Per-signal, within-stage AUROC for TLE and VC (Chapter 8 §8.1 premise).

The confirmatory H1a analysis reports only the *difference* between the two signals'
discrimination, stage-stratified and pooled, plus a stage-only floor. Chapter 8 opens its
interpretation by asserting that each signal separates correct from incorrect steps above chance
within every compute stage, and nothing in the thesis reported that quantity: a difference of
zero is equally consistent with both signals discriminating well and with neither discriminating
at all. This script supplies the missing per-signal estimates so the claim is checkable.

Estimand: AUROC of one signal against `y_optimal`, computed within a single compute stage, per
domain. Orientation follows `delta_auroc` -- the TLE score is the negated mean action-token
entropy, the VC score is the raw report -- so that under both signals a higher score means a more
likely correct step and .5 is chance. Intervals come from the same instance-clustered bootstrap
harness as every other confirmatory statistic (NB = 5000, SEED = 20260703), resampling instances
within the domain and recomputing the within-stage AUROC on each replicate.

This is a descriptive, disclosed addition. It changes no preregistered analysis and carries no
hypothesis test; the interval is reported so a reader can see whether it clears .5.

Run from repo root:
  python scripts/phase1_analysis/per_signal_stage_auroc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compute_auroc  # noqa: E402
from src.analysis.inference import cluster_bootstrap  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

DOMS = ("tower_of_hanoi", "textworld")
STG = ("C0", "C1", "C2")
SIGNALS = ("tle", "vc")
NB = 5000
SEED = 20260703
OUT = REPO_ROOT / "data/results/phase1_analysis/per_signal_stage_auroc.json"


def _score(row, signal):
    """Orientation as in `delta_auroc`: higher score = more likely correct."""
    if signal == "tle":
        v = row.get("tle_mean_entropy")
        return None if v is None else -float(v)
    v = row.get("vc")
    return None if v is None else float(v)


def stage_signal_auroc(rows, stage, signal):
    """AUROC of one signal within one compute stage, on steps carrying both signals.

    Restricted to steps with both signals present so that every cell of the table is computed on
    the identical step set, which is what makes the two signals comparable within a stage.
    """
    sc, lb = [], []
    for r in rows:
        if str(r.get("compute_stage")) != stage:
            continue
        if (
            r.get("y_optimal") is None
            or r.get("tle_mean_entropy") is None
            or r.get("vc") is None
        ):
            continue
        sc.append(_score(r, signal))
        lb.append(int(r["y_optimal"]))
    if len(set(lb)) < 2:
        return float("nan")
    return compute_auroc(sc, lb)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=NB)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    n_boot = args.n_boot
    out_path = Path(args.out)

    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    steps = ds.steps

    report = {
        "purpose": (
            "Per-signal within-stage AUROC for TLE and VC. Supplies the quantity Chapter 8 needs, "
            "namely whether each signal clears chance within each compute stage, which the "
            "difference-based H1a analysis does not report."
        ),
        "n_boot": n_boot,
        "seed": SEED,
        "orientation": "TLE score = -mean action-token entropy; VC score = raw report; .5 = chance",
        "step_set": "steps carrying y_optimal and both signals",
        "results": {},
    }

    print(f"Per-signal within-stage AUROC  (NB={n_boot}, seed={SEED})\n")
    for dom in DOMS:
        dr = [r for r in steps if str(r.get("domain")) == dom]
        print(f"  {dom}")
        for stage in STG:
            n = sum(
                1
                for r in dr
                if str(r.get("compute_stage")) == stage
                and r.get("y_optimal") is not None
                and r.get("tle_mean_entropy") is not None
                and r.get("vc") is not None
            )
            line = f"    {stage}  n={n:6d}"
            for signal in SIGNALS:
                b = cluster_bootstrap(
                    dr,
                    lambda rows, s=stage, g=signal: stage_signal_auroc(rows, s, g),
                    n_boot=n_boot,
                    seed=SEED,
                )
                lo, hi = b["ci_low"], b["ci_high"]
                above = lo is not None and lo > 0.5
                report["results"][f"{dom}/{stage}/{signal}"] = {
                    "n_steps": n,
                    "auroc": b["point"],
                    "ci_low": lo,
                    "ci_high": hi,
                    "above_chance": bool(above),
                }
                line += (
                    f"   {signal.upper()}={b['point']:.3f} [{lo:.3f}, {hi:.3f}]"
                    f"{'*' if above else ' '}"
                )
            print(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")

    n_cells = len(report["results"])
    n_above = sum(1 for v in report["results"].values() if v["above_chance"])
    print(f"Above chance (lower bound > .5) in {n_above} of {n_cells} cells.  * marks those.")


if __name__ == "__main__":
    main()
