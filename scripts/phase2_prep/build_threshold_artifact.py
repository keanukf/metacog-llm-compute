#!/usr/bin/env python3
"""
Build the frozen Phase 1 -> Phase 2 threshold/policy artifact (thesis §5.4).

Grid-searches TLE/VC allocator thresholds (theta1 < theta2, `step_level_proxy_v1` objective)
against the real Phase 1 holdout data (5 of 50 instances per domain, frozen after Gate D),
producing the policy artifact `adaptive_tle`/`adaptive_vc`/`eager_style` require
(`POLICY_REQUIRED_STRATEGIES`, `src/agent/allocator.py`). This is the one concrete blocker
between "Phase 1 analysis done" and "Phase 2 collection can start" -- see
docs/consistency_log.md, 2026-08-04 entry, for the full context.

Reads through the Stage 0 canonical manifest (`scripts/phase1_analysis/stage0_build_canonical_
dataset.py`) rather than re-deriving the domain/directory selection -- same "one source of truth
for what real Phase 1 data actually is" reasoning as the H1a-H4 pipeline.

Usage:
  python scripts/phase2_prep/build_threshold_artifact.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1/threshold_artifact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)
from src.analysis.thresholds import write_threshold_artifact  # noqa: E402
from src.utils.logging_utils import try_git_commit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument("--output", default="data/results/phase1/threshold_artifact.json")
    args = parser.parse_args()

    manifest_path = (
        REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    )
    if not manifest_path.exists():
        print(
            f"FAILED -- manifest not found at {manifest_path}; run "
            "scripts/phase1_analysis/stage0_build_canonical_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    ds = load_canonical_dataset_from_manifest(manifest_path)
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    holdout_steps = [r for r in ds.steps if bool(r.get("holdout"))]
    if not holdout_steps:
        print(
            "FAILED -- no holdout steps found in the canonical dataset; the Gate D holdout flags "
            "should be present on every real Phase 1 step. Not writing a fallback quantile "
            "artifact silently -- that would be a legacy-pilot artifact, not a real Phase 1 one.",
            file=sys.stderr,
        )
        return 1

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    artifact_path = write_threshold_artifact(
        out_path,
        ds.steps,
        git_commit=try_git_commit(REPO_ROOT),
    )

    print(f"OK -- threshold artifact written to {artifact_path}")
    obj = json.loads(artifact_path.read_text(encoding="utf-8"))
    for domain, by_signal in sorted(obj.get("by_domain", {}).items()):
        for signal, block in sorted(by_signal.items()):
            theta1 = block.get("theta1")
            theta2 = block.get("theta2")
            direction = block.get("direction")
            brier = block.get("brier_eval_non_holdout")
            print(
                f"  {domain}/{signal}: theta1={theta1:.4f} theta2={theta2:.4f} "
                f"direction={direction} brier_eval_non_holdout={brier}"
            )
            # theta1 == theta2 (or both at a grid extreme) means the grid search collapsed to a
            # degenerate single-stage policy -- the exact failure mode the Gate E rehearsal pilot
            # flagged as a real risk on a small holdout (docs/gate_e_rehearsal.md). Not fatal
            # (the artifact is still written), but must not pass silently.
            if theta1 is not None and theta2 is not None and theta1 >= theta2:
                print(
                    f"  WARNING [{domain}/{signal}]: theta1 >= theta2 -- degenerate policy "
                    "(collapses to at most 2 effective stages). Inspect before using for real "
                    "Phase 2 collection.",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
