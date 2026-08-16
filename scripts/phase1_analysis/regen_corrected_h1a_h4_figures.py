#!/usr/bin/env python3
"""Regenerate the H1a and H4 discrimination figures under the corrected estimand.

The original ``h1a_auroc_comparison.png`` and ``bootstrap_dist_h4.png`` show the
stage-pooled discrimination result, which the compute-stage-confound correction
(see ``stage_stratified_sensitivity.py`` and thesis Sec 6.1/6.4) supersedes. This
script re-renders both as an explicit pooled-vs-stage-stratified comparison, so the
figure the thesis references matches the corrected tables: for the Tower of Hanoi the
pooled contrast favours TLE while the stage-stratified contrast reverses sign and
favours VC. It reuses the same cluster-bootstrap harness and the same helper
statistics as the sensitivity re-analysis, with identical NB/SEED, so the numbers
match the committed sensitivity output to 4 dp.

Run from repo root:
  python scripts/phase1_analysis/regen_corrected_h1a_h4_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# reuse the sensitivity helpers (strat_dauroc, stage_only_auroc, h4) verbatim
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.inference import (  # noqa: E402
    cluster_bootstrap,
    cluster_bootstrap_stratified,
    delta_auroc,
)
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)
from stage_stratified_sensitivity import h4, stage_only_auroc, strat_dauroc  # noqa: E402

DOMS = ("tower_of_hanoi", "textworld")
DOM_LABEL = {"tower_of_hanoi": "Tower of Hanoi", "textworld": "TextWorld"}
NB = 5000
SEED = 20260703


def _load_steps():
    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    return ds.steps


def _fig_h1a(steps, out_dir: Path) -> Path:
    """Per-domain pooled vs stage-stratified ΔAUROC(TLE−VC) with 90% CIs."""
    rows = {}
    for dom in DOMS:
        dr = [r for r in steps if str(r.get("domain")) == dom]
        pooled = cluster_bootstrap(dr, delta_auroc, n_boot=NB, seed=SEED)
        strat = cluster_bootstrap(dr, strat_dauroc, n_boot=NB, seed=SEED)
        floor = cluster_bootstrap(dr, stage_only_auroc, n_boot=NB, seed=SEED)
        rows[dom] = {"pooled": pooled, "strat": strat, "floor": floor}

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = list(range(len(DOMS)))
    width = 0.34
    colors = {"pooled": "tab:gray", "strat": "tab:blue"}
    for off, key, label in (
        (-width / 2, "pooled", "Pooled (confounded reference)"),
        (width / 2, "strat", "Stage-stratified (confirmatory)"),
    ):
        pts = [rows[d][key]["point"] for d in DOMS]
        los = [rows[d][key]["point"] - rows[d][key]["ci_low"] for d in DOMS]
        his = [rows[d][key]["ci_high"] - rows[d][key]["point"] for d in DOMS]
        ax.errorbar(
            [i + off for i in x],
            pts,
            yerr=[los, his],
            fmt="o",
            color=colors[key],
            capsize=4,
            markersize=7,
            linewidth=1.6,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    for i, d in enumerate(DOMS):
        fl = rows[d]["floor"]["point"]
        ax.annotate(
            f"stage-only\nAUROC={fl:.2f}",
            (i, -0.135),
            ha="center",
            va="top",
            fontsize=7,
            color="dimgray",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DOM_LABEL[d] for d in DOMS])
    ax.set_ylim(-0.16, 0.16)
    ax.set_title("H1a: discrimination advantage, pooled vs stage-stratified")
    ax.set_ylabel("ΔAUROC (TLE − VC), 90% CI\n(above 0: TLE ranks better; below 0: VC)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    p = out_dir / "h1a_auroc_comparison.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _fig_h4(steps, out_dir: Path) -> Path:
    """Overlaid bootstrap distributions of the H4 DiD, pooled vs stage-stratified."""
    pooled = cluster_bootstrap_stratified(steps, h4(delta_auroc), n_boot=NB, seed=SEED)
    strat = cluster_bootstrap_stratified(steps, h4(strat_dauroc), n_boot=NB, seed=SEED)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.hist(pooled["reps"], bins=50, color="tab:gray", alpha=0.55, label=None)
    ax.hist(strat["reps"], bins=50, color="tab:blue", alpha=0.6, label=None)
    for res, color, name in (
        (pooled, "dimgray", "Pooled (confounded reference)"),
        (strat, "tab:blue", "Stage-stratified (confirmatory)"),
    ):
        ax.axvline(
            res["point"],
            color=color,
            linewidth=1.8,
            label=f"{name}: DiD={res['point']:+.3f} [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]",
        )
        ax.axvline(res["ci_low"], color=color, linestyle="--", linewidth=1.0)
        ax.axvline(res["ci_high"], color=color, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.2, label="null = 0")
    ax.set_xlabel("Difference-in-differences (Tower of Hanoi − TextWorld)")
    ax.set_ylabel("Count")
    ax.set_title("H4: domain modulation of discrimination, pooled vs stage-stratified")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", fontsize=7.5, frameon=False)
    fig.tight_layout()
    p = out_dir / "bootstrap_dist_h4.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "data/results/phase1_analysis/corrected_figures"),
        help="output directory for the regenerated figures",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _load_steps()
    p1 = _fig_h1a(steps, out_dir)
    p2 = _fig_h4(steps, out_dir)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
