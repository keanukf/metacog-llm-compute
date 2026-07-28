#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 6: visualizations.

Reads the Stage 2 (H1a) and Stage 4 (H3) result JSON and renders the figures
docs/phase1_analysis_report.md embeds: TLE-vs-VC AUROC comparison bars per domain, and the
H3 signal x position marginal-effect curves per domain/signal. Purely a plotting pass over
already-computed stage output -- no new statistics, so it's safe to re-run anytime after
Stages 2 and 4.

Usage:
  python scripts/phase1_analysis/stage6_visualizations.py \
      --stage2 data/results/phase1_analysis/stage2/h1a_discrimination.json \
      --stage4 data/results/phase1_analysis/stage4/h3_temporal.json \
      --output-dir data/results/phase1_analysis/stage6/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.visualization import (  # noqa: E402
    plot_auroc_comparison_bars,
    plot_h3_marginal_effect,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage2", default="data/results/phase1_analysis/stage2/h1a_discrimination.json")
    parser.add_argument("--stage4", default="data/results/phase1_analysis/stage4/h3_temporal.json")
    parser.add_argument("--output-dir", default="data/results/phase1_analysis/stage6/figures")
    args = parser.parse_args()

    stage2_path = REPO_ROOT / args.stage2 if not Path(args.stage2).is_absolute() else Path(args.stage2)
    stage4_path = REPO_ROOT / args.stage4 if not Path(args.stage4).is_absolute() else Path(args.stage4)
    for p, stage in ((stage2_path, "Stage 2"), (stage4_path, "Stage 4")):
        if not p.exists():
            print(f"Stage 6 FAILED -- {stage} output not found at {p}; run it first.", file=sys.stderr)
            return 1

    h1a_results = json.loads(stage2_path.read_text(encoding="utf-8"))
    h3_results = json.loads(stage4_path.read_text(encoding="utf-8"))

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    written: dict[str, str] = {}
    written.update(plot_auroc_comparison_bars(h1a_results, output_dir))
    written.update(plot_h3_marginal_effect(h3_results, output_dir))

    if not written:
        print(
            "Stage 6 WARNING -- no figures written (matplotlib unavailable, or no converged fits)",
            file=sys.stderr,
        )

    manifest_path = output_dir / "figures_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(written, indent=2), encoding="utf-8")

    print(f"Stage 6 OK -- {len(written)} figure(s) written to {output_dir}")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
