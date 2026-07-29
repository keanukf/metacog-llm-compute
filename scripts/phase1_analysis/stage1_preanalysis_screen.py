#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 1: preanalysis screen.

Reads the Stage 0 canonical manifest, loads the underlying episode/step rows, and runs the full
preanalysis screen (src/analysis/preanalysis_screen.py::run_preanalysis_screen) -- signal
variance/VC degeneration, cluster counts, class balance by domain AND by position_norm bin (with
an explicit empty-cell flag), episode-length distribution (full quartile spread, not just mean),
and real ICC estimation (src/analysis/icc.py, lifted out of the H3 power simulation so it can
finally be run against actual data) -- before any confirmatory hypothesis test runs.

Also renders a full per-variable descriptive codebook (src/analysis/descriptive_stats.py, APA-7
styled Markdown tables) and distribution/whisker-plot figures (src/analysis/visualization.py) --
the "describe every variable properly" artifacts, not just the pass/fail data-quality gates above.

This stage does not hard-fail on a bad screen result (the preregistered plan frames these checks
as diagnostic, not selective -- see thesis §5.8/§5.9); it prints a clear warning banner for any
empty position x correctness cells or near-degenerate signal variance so a human reads it before
trusting Stage 2+ numbers.

Usage:
  python scripts/phase1_analysis/stage1_preanalysis_screen.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage1/preanalysis_screen.json
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

from src.analysis.descriptive_stats import (  # noqa: E402
    compute_variable_codebook,
    render_apa_codebook_markdown,
)
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402
from src.analysis.preanalysis_screen import run_preanalysis_screen  # noqa: E402
from src.analysis.visualization import (  # noqa: E402
    plot_episode_length_boxplot,
    plot_signal_boxplots,
    plot_signal_histograms,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument(
        "--output", default="data/results/phase1_analysis/stage1/preanalysis_screen.json"
    )
    parser.add_argument(
        "--codebook-output", default="data/results/phase1_analysis/stage1/variable_codebook.md"
    )
    parser.add_argument(
        "--figures-output", default="data/results/phase1_analysis/stage1/figures"
    )
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"Stage 1 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.",
            file=sys.stderr,
        )
        return 1

    ds = load_canonical_dataset_from_manifest(manifest_path)
    screen = run_preanalysis_screen(ds.steps, ds.episodes)

    codebook = compute_variable_codebook(ds.steps, ds.episodes)
    screen["variable_codebook"] = codebook

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(screen, indent=2), encoding="utf-8")
    print(f"Stage 1 OK -- screen written to {out_path}")

    codebook_md = render_apa_codebook_markdown(codebook)
    codebook_path = (
        REPO_ROOT / args.codebook_output
        if not Path(args.codebook_output).is_absolute()
        else Path(args.codebook_output)
    )
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_path.write_text(codebook_md, encoding="utf-8")
    print(f"Stage 1 OK -- variable codebook written to {codebook_path}")

    figures_dir = (
        REPO_ROOT / args.figures_output
        if not Path(args.figures_output).is_absolute()
        else Path(args.figures_output)
    )
    written_figures: dict[str, str] = {}
    written_figures.update(plot_signal_histograms(ds.steps, figures_dir))
    written_figures.update(plot_signal_boxplots(ds.steps, figures_dir))
    written_figures.update(plot_episode_length_boxplot(ds.episodes, figures_dir))
    if written_figures:
        print(f"Stage 1 OK -- {len(written_figures)} figure(s) written to {figures_dir}")
        (figures_dir / "figures_manifest.json").write_text(
            json.dumps(written_figures, indent=2), encoding="utf-8"
        )

    warned = False
    for dom, report in screen.get("by_domain", {}).items():
        n_empty = report.get("position_correctness", {}).get("n_empty_cells", 0)
        if n_empty:
            warned = True
            print(f"  WARNING [{dom}]: {n_empty} empty position x correctness cell(s)")
        icc = report.get("icc", {})
        print(
            f"  {dom}: n_steps={report.get('n_steps')} n_clusters={report.get('n_clusters')} "
            f"icc_gee={icc.get('icc_gee')} vc_missing_rate={report.get('vc_missing_rate')}"
        )
    if warned:
        print("Stage 1: see WARNING lines above -- diagnostic only, does not block later stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
