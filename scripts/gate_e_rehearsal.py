#!/usr/bin/env python3
"""
Gate E — analysis-pipeline rehearsal (see blueprints/gate_p1_readiness.md, Gate E).

Runs the analysis-side half of the confirmatory chain end-to-end on an existing pilot/C-5
run directory, before any real Phase-1 data exists:

    episode JSONs -> step table (src/analysis/datasets.py)
                  -> grid_search_thresholds / write_threshold_artifact (src/analysis/thresholds.py)
                  -> load_policy sanity check (src/agent/allocation_policy.py)
                  -> cluster_bootstrap on a ΔAUROC(TLE, VC) statistic (src/analysis/inference.py)

The run directory used for the rehearsal (default: the Gate-C signal-smoke run) was NOT
generated with Gate D's difficulty_manifest.json (holdout / difficulty_tier fields), so this
script imposes an artificial "first N instances per domain = holdout" split purely for the
rehearsal. This is a documented workaround, not the real Gate D manifest — see
docs/gate_e_rehearsal.md for the full write-up.

The Phase-2 mock smoke (run_phase2.py --config configs/dev/gate_e_rehearsal.yaml) is a
separate CLI invocation that consumes the policy artifact this script writes; it is not
run from here (see docs/gate_e_rehearsal.md for the exact command).

Usage:
  python scripts/gate_e_rehearsal.py \
      --run-dir data/results/instrument_validation/phase1_20260714_105004 \
      --holdout-instances 3 \
      --artifact-out data/results/gate_e_rehearsal/policy_artifact.json \
      --report-out data/results/gate_e_rehearsal/rehearsal_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _apply_artificial_holdout_split(steps: list[dict[str, Any]], *, holdout_instances: int) -> None:
    """Overwrite ``holdout`` on every step row: instance < holdout_instances -> True.

    In-place, uniform across domains. Overrides any pre-existing ``holdout`` value (e.g. the
    Tower-of-Hanoi episodes in the 105004 run already carry a ``holdout`` field from an interim,
    not-yet-frozen manifest) so the rehearsal exercises a single, domain-symmetric split it fully
    controls, rather than a mix of "real interim manifest" (ToH) and "no manifest at all"
    (TextWorld).
    """
    for row in steps:
        inst = row.get("instance")
        row["holdout"] = bool(inst is not None and int(inst) < holdout_instances)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        default="data/results/instrument_validation/phase1_20260714_105004",
        help="Episode-JSON run directory to rehearse on.",
    )
    parser.add_argument(
        "--holdout-instances",
        type=int,
        default=3,
        help="Instances 0..N-1 (per domain) are marked holdout=True; rest are pool.",
    )
    parser.add_argument(
        "--artifact-out",
        default="data/results/gate_e_rehearsal/policy_artifact.json",
        help="Where to write the threshold/policy artifact (write_threshold_artifact).",
    )
    parser.add_argument(
        "--report-out",
        default="data/results/gate_e_rehearsal/rehearsal_report.json",
        help="Where to write the JSON rehearsal report (evidence for docs/gate_e_rehearsal.md).",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    artifact_out = Path(args.artifact_out)
    if not artifact_out.is_absolute():
        artifact_out = REPO_ROOT / artifact_out
    report_out = Path(args.report_out)
    if not report_out.is_absolute():
        report_out = REPO_ROOT / report_out

    from src.agent.allocation_policy import load_policy
    from src.analysis.datasets import load_run_dataset, validate_analysis_schema
    from src.analysis.inference import cluster_bootstrap, delta_auroc
    from src.analysis.thresholds import write_threshold_artifact

    # --- Step 1: episode JSONs -> step table ---
    ds = load_run_dataset(run_dir)
    print(
        f"[1/4] load_run_dataset({run_dir}) -> {len(ds.episodes)} episodes, {len(ds.steps)} steps"
    )
    health = validate_analysis_schema(ds.steps)
    print(f"      validate_analysis_schema: {health}")

    # --- Step 2: artificial holdout/pool split (rehearsal workaround, see docstring) ---
    _apply_artificial_holdout_split(ds.steps, holdout_instances=args.holdout_instances)
    split_counts = Counter((r.get("domain"), bool(r.get("holdout"))) for r in ds.steps)
    print(
        f"      artificial holdout split (instance < {args.holdout_instances}): {dict(split_counts)}"
    )

    # --- Step 3: grid_search_thresholds (via write_threshold_artifact) -> policy artifact ---
    artifact_path = write_threshold_artifact(artifact_out, ds.steps)
    artifact_body = json.loads(artifact_path.read_text(encoding="utf-8"))
    print(f"[2/4] write_threshold_artifact -> {artifact_path}")
    thresholds_summary: dict[str, Any] = {}
    for domain, by_signal in artifact_body.get("by_domain", {}).items():
        thresholds_summary[domain] = {}
        for signal, block in by_signal.items():
            thresholds_summary[domain][signal] = {
                "theta1": block.get("theta1"),
                "theta2": block.get("theta2"),
                "objective_definition": block.get("objective_definition"),
                "n_grid_candidates": len(block.get("grid_table") or []),
                "n_holdout_by_stage": {
                    k: len(v) for k, v in (block.get("ecdf_by_stage") or {}).items()
                },
            }
            print(
                f"      {domain}/{signal}: theta1={block.get('theta1')} theta2={block.get('theta2')}"
            )

    # --- Step 4: load_policy sanity check ---
    policy_summary: dict[str, Any] = {}
    for domain in thresholds_summary:
        pol = load_policy(artifact_path, domain=domain, signal="tle_mean_entropy")
        stage_with_ref = max(pol.ecdf_by_stage.items(), key=lambda kv: len(kv[1]))
        stage_name, ref = stage_with_ref
        low, mid, high = ref[0], ref[len(ref) // 2], ref[-1]
        stages_seen = [
            pol.stage(low, source_stage=stage_name),
            pol.stage(mid, source_stage=stage_name),
            pol.stage(high, source_stage=stage_name),
        ]
        policy_summary[domain] = {
            "theta1": pol.theta1,
            "theta2": pol.theta2,
            "direction": pol.direction,
            "ecdf_by_stage_lens": {k: len(v) for k, v in pol.ecdf_by_stage.items()},
            "low_mid_high_stage_probe": stages_seen,
        }
        print(f"[3/4] load_policy({domain}, tle_mean_entropy) -> low/mid/high stage: {stages_seen}")

    # --- Step 5: cluster_bootstrap on a ΔAUROC(TLE, VC) statistic ---
    # Confirmatory-style split: fit thresholds on holdout, evaluate signal quality on the
    # non-holdout ("pool") rows, mirroring build_policy_artifact's own
    # platt_eval=non_holdout_confirmatory convention.
    pool_rows = [
        r for r in ds.steps if not bool(r.get("holdout")) and r.get("y_optimal") is not None
    ]
    bootstrap_summary: dict[str, Any] = {}
    for label, rows in (
        ("pooled", pool_rows),
        ("textworld", [r for r in pool_rows if r.get("domain") == "textworld"]),
        ("tower_of_hanoi", [r for r in pool_rows if r.get("domain") == "tower_of_hanoi"]),
    ):
        res = cluster_bootstrap(
            rows,
            delta_auroc,
            cluster_col="instance_key",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        bootstrap_summary[label] = {"n_steps": len(rows), **res}
        print(
            f"[4/4] cluster_bootstrap(delta_auroc) {label}: n={len(rows)} "
            f"point={res.get('point')} ci=({res.get('ci_low')}, {res.get('ci_high')})"
        )

    report = {
        "run_dir": str(run_dir),
        "holdout_instances": args.holdout_instances,
        "n_episodes": len(ds.episodes),
        "n_steps": len(ds.steps),
        "schema_health": health,
        "split_counts": {f"{d}:{h}": n for (d, h), n in split_counts.items()},
        "artifact_path": str(artifact_path),
        "thresholds_summary": thresholds_summary,
        "policy_summary": policy_summary,
        "bootstrap_delta_auroc": bootstrap_summary,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_out}")


if __name__ == "__main__":
    main()
