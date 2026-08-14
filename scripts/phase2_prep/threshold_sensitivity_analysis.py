#!/usr/bin/env python3
"""
Phase 1 -> Phase 2: TextWorld threshold-sensitivity analysis (thesis §5.9).

Thesis §5.9 flags the five-instance threshold holdout as a source of variance in the selected
(theta1, theta2) pair and promises "a reported sensitivity analysis around the selected
thresholds" as mitigation. That analysis was never delivered until now. The TextWorld
holdout-labeling bug documented in docs/consistency_log.md (2026-08-14 entry) -- the deployed
Phase 2 allocator was fit on a non-preregistered five-instance split ({0,1,2,3,4}) instead of the
manifest's mod-10 holdout ({0,10,20,30,40}) -- is a real, already-occurred instantiation of exactly
that named risk, not a hypothetical one, and gives a concrete pair of threshold configurations to
compare: "deployed" (fit on the wrong five) vs. "corrected" (fit on the true five).

This script reconstructs both configurations from the raw Phase 1 data via
``src.analysis.thresholds.grid_search_thresholds`` -- the same function
``scripts/phase2_prep/build_threshold_artifact.py`` uses for the real artifact -- so nothing here
is hand-copied from a prior run. It reports two things, both derived from data already collected
(no re-collection, no additional GPU cost):

1. Proxy-objective gap: where the deployed pair lands on the *correctly evaluated* TextWorld
   Pareto front (both theta pairs sit on the same 0.1-0.9 grid, so the deployed pair is looked up
   directly in the true-holdout grid_table rather than re-searched).
2. Realized routing gap: replaying the actual, already-observed Phase 2 TextWorld signal values
   (from the real adaptive_tle/adaptive_vc episodes) through both policies and comparing the
   resulting C0/C1/C2 stage distributions -- how differently the two configurations would actually
   have behaved, not just how their fitted numbers differ.

``TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL`` (src/analysis/phase1_canonical.py) is
used only here, to reconstruct "what was actually deployed" from code; no other analysis in this
repository is allowed to depend on it.

Usage:
  python scripts/phase2_prep/threshold_sensitivity_analysis.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --phase2-checkpoint-dir data/results/phase2/phase2_stage1_20260805_UTC \
      --output data/results/phase2_analysis/threshold_sensitivity/textworld_sensitivity.json
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

from src.agent.allocation_policy import FrozenPolicy  # noqa: E402
from src.analysis.datasets import load_run_dataset  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL,
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)
from src.analysis.thresholds import grid_search_thresholds  # noqa: E402

DOMAIN = "textworld"
SIGNALS = ("tle_mean_entropy", "vc")
STRATEGY_BY_SIGNAL = {"tle_mean_entropy": "adaptive_tle", "vc": "adaptive_vc"}


def _fit_scenario(all_steps: list[dict[str, Any]], *, signal: str) -> dict[str, Any]:
    """Grid-search a (theta1, theta2) pair for ``signal`` against whatever ``holdout`` flags are
    already set on ``all_steps`` -- the caller picks the scenario by which correction it applied
    beforehand."""
    dom_steps = [r for r in all_steps if str(r.get("domain")) == DOMAIN]
    holdout_steps = [r for r in dom_steps if bool(r.get("holdout"))]
    return grid_search_thresholds(holdout_steps, dom_steps, signal=signal, label_key="y_optimal")


def _lookup_grid_entry(
    grid_table: list[dict[str, Any]], *, theta1: float, theta2: float
) -> dict[str, Any] | None:
    for entry in grid_table:
        if abs(entry["theta1"] - theta1) < 1e-9 and abs(entry["theta2"] - theta2) < 1e-9:
            return entry
    return None


def _routing_distribution(
    policy: FrozenPolicy, observed: list[tuple[float, str]]
) -> dict[str, Any]:
    counts = {"C0": 0, "C1": 0, "C2": 0}
    for raw, source_stage in observed:
        counts[policy.stage(raw, source_stage=source_stage)] += 1
    total = sum(counts.values())
    return {
        "n": total,
        "C0": counts["C0"],
        "C1": counts["C1"],
        "C2": counts["C2"],
        "fraction": {k: (v / total if total else None) for k, v in counts.items()},
    }


def run(manifest_path: Path, phase2_checkpoint_dir: Path) -> dict[str, Any]:
    ds_true = load_canonical_dataset_from_manifest(manifest_path)
    apply_textworld_holdout_correction(ds_true.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)

    ds_deployed = load_canonical_dataset_from_manifest(manifest_path)
    apply_textworld_holdout_correction(
        ds_deployed.steps, TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL
    )

    p2_ds = load_run_dataset(phase2_checkpoint_dir)
    p2_textworld_steps = [s for s in p2_ds.steps if str(s.get("domain")) == DOMAIN]

    by_signal: dict[str, Any] = {}
    for signal in SIGNALS:
        true_fit = _fit_scenario(ds_true.steps, signal=signal)
        deployed_fit = _fit_scenario(ds_deployed.steps, signal=signal)

        deployed_on_true_grid = None
        if true_fit.get("theta1") is not None and deployed_fit.get("theta1") is not None:
            deployed_on_true_grid = _lookup_grid_entry(
                true_fit["grid_table"],
                theta1=deployed_fit["theta1"],
                theta2=deployed_fit["theta2"],
            )
        selected_true = next(
            (
                e
                for e in true_fit["grid_table"]
                if e["theta1"] == true_fit["theta1"] and e["theta2"] == true_fit["theta2"]
            ),
            None,
        )

        true_policy = FrozenPolicy(
            signal=signal,
            domain=DOMAIN,
            ecdf_by_stage=true_fit["ecdf_by_stage"],
            theta1=true_fit["theta1"],
            theta2=true_fit["theta2"],
            direction=true_fit["direction"],
        )
        deployed_policy = FrozenPolicy(
            signal=signal,
            domain=DOMAIN,
            ecdf_by_stage=deployed_fit["ecdf_by_stage"],
            theta1=deployed_fit["theta1"],
            theta2=deployed_fit["theta2"],
            direction=deployed_fit["direction"],
        )

        strategy = STRATEGY_BY_SIGNAL[signal]
        observed: list[tuple[float, str]] = []
        for s in p2_textworld_steps:
            if str(s.get("strategy")) != strategy:
                continue
            raw = s.get(signal)
            source_stage = s.get("compute_stage")
            if raw is None or source_stage is None:
                continue
            observed.append((float(raw), str(source_stage)))

        by_signal[signal] = {
            "true_holdout_fit": {
                "theta1": true_fit["theta1"],
                "theta2": true_fit["theta2"],
                "selection_rule": true_fit.get("selection_rule"),
                "selected_grid_entry": selected_true,
            },
            "deployed_reconstructed_fit": {
                "theta1": deployed_fit["theta1"],
                "theta2": deployed_fit["theta2"],
                "note": (
                    "reconstructed by re-running grid_search_thresholds against the historical "
                    "wrong holdout instance set; not read from any saved artifact"
                ),
            },
            "deployed_pair_scored_on_true_holdout_grid": deployed_on_true_grid,
            "pareto_front_size_true_holdout": len(true_fit.get("pareto_front", [])),
            "realized_routing": {
                "true_holdout_policy": _routing_distribution(true_policy, observed),
                "deployed_policy": _routing_distribution(deployed_policy, observed),
                "n_observed_steps": len(observed),
                "source": (
                    f"real Phase 2 {strategy}/{DOMAIN} episodes, {phase2_checkpoint_dir.name}"
                ),
            },
        }

    return {
        "domain": DOMAIN,
        "purpose": (
            "thesis Section 5.9 threshold-sensitivity analysis, delivered via the real "
            "holdout-labeling incident (docs/consistency_log.md, 2026-08-14 entry)"
        ),
        "true_holdout_instances": sorted(TEXTWORLD_TRUE_HOLDOUT_INSTANCES),
        "deployed_wrong_holdout_instances_historical": sorted(
            TEXTWORLD_DEPLOYED_WRONG_HOLDOUT_INSTANCES_HISTORICAL
        ),
        "by_signal": by_signal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument(
        "--phase2-checkpoint-dir", default="data/results/phase2/phase2_stage1_20260805_UTC"
    )
    parser.add_argument(
        "--output",
        default="data/results/phase2_analysis/threshold_sensitivity/textworld_sensitivity.json",
    )
    args = parser.parse_args()

    manifest_path = (
        REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    )
    checkpoint_dir = (
        REPO_ROOT / args.phase2_checkpoint_dir
        if not Path(args.phase2_checkpoint_dir).is_absolute()
        else Path(args.phase2_checkpoint_dir)
    )
    if not manifest_path.exists():
        print(f"FAILED -- manifest not found at {manifest_path}", file=sys.stderr)
        return 1
    if not checkpoint_dir.exists():
        print(f"FAILED -- Phase 2 checkpoint dir not found at {checkpoint_dir}", file=sys.stderr)
        return 1

    result = run(manifest_path, checkpoint_dir)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"OK -- threshold sensitivity analysis written to {out_path}")
    for signal, block in result["by_signal"].items():
        tf = block["true_holdout_fit"]
        df = block["deployed_reconstructed_fit"]
        print(f"[{signal}]")
        print(f"  true-holdout fit:      theta1={tf['theta1']:.2f} theta2={tf['theta2']:.2f}")
        print(f"  deployed (wrong-fit):  theta1={df['theta1']:.2f} theta2={df['theta2']:.2f}")
        dp = block["deployed_pair_scored_on_true_holdout_grid"]
        se = tf["selected_grid_entry"]
        if dp is not None and se is not None:
            print(
                f"  scored on true holdout grid: deployed success_proxy={dp['success_proxy']:.4f} "
                f"token_proxy={dp['token_proxy']:.1f}  vs. selected "
                f"success_proxy={se['success_proxy']:.4f} token_proxy={se['token_proxy']:.1f}"
            )
        rr = block["realized_routing"]
        n = rr["n_observed_steps"]
        tp = rr["true_holdout_policy"]["fraction"]
        dpol = rr["deployed_policy"]["fraction"]
        print(f"  realized routing over n={n} observed Phase 2 steps:")
        print(
            "    true-holdout policy:  "
            + " ".join(f"{k}={v:.3f}" if v is not None else f"{k}=NA" for k, v in tp.items())
        )
        print(
            "    deployed policy:      "
            + " ".join(f"{k}={v:.3f}" if v is not None else f"{k}=NA" for k, v in dpol.items())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
