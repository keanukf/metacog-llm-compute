#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- thin sequential orchestrator.

Chains Stages 0-7 via subprocess calls, stopping immediately on the first non-zero exit. Each
stage is independently runnable (see the stageN_*.py scripts directly for that), but this is the
one-shot "reproduce everything" entry point referenced by docs/phase1_analysis_report.md.

Usage:
  python scripts/phase1_analysis/run_all.py [--seed 20260703] [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_DIR = Path(__file__).resolve().parent

STAGE_DATA_ROOT = "data/results/phase1_analysis"


def _stage_cmd(script: str, *extra: str) -> list[str]:
    return [sys.executable, str(STAGE_DIR / script), *extra]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    boot_args = ["--seed", str(args.seed), "--n-boot", str(args.n_boot)]

    stages = [
        ("Stage 0 (canonical dataset)", _stage_cmd("stage0_build_canonical_dataset.py")),
        ("Stage 1 (preanalysis screen)", _stage_cmd("stage1_preanalysis_screen.py")),
        ("Stage 2 (H1a discrimination)", _stage_cmd("stage2_h1a_discrimination.py", *boot_args)),
        ("Stage 3 (H1b calibration)", _stage_cmd("stage3_h1b_calibration.py", *boot_args)),
        ("Stage 4 (H3 temporal)", _stage_cmd("stage4_h3_temporal.py")),
        (
            "Stage 5 (H4 domain modulation)",
            _stage_cmd("stage5_h4_domain_modulation.py", *boot_args),
        ),
        ("Stage 6 (visualizations)", _stage_cmd("stage6_visualizations.py")),
        ("Stage 7 (report generation)", _stage_cmd("stage7_generate_report.py")),
    ]

    for label, cmd in stages:
        print(f"=== {label} ===", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"run_all FAILED at {label} (exit {result.returncode})", file=sys.stderr)
            return result.returncode

    print("run_all OK -- all stages completed, see docs/phase1_analysis_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
