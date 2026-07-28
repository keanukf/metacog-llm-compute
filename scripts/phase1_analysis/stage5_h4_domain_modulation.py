#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 5: H4 (domain modulation).

Single confirmatory test (Family C, 1 test -- Holm reduces to the raw p-value, kept for
uniformity with the other stages): h4_diff_in_diff = [AUROC_TLE - AUROC_VC]_ToH -
[AUROC_TLE - AUROC_VC]_TextWorld, bootstrapped via cluster_bootstrap_stratified (thesis
Ch.5 §5.8, verbatim: "estimated by the cluster bootstrap with instances resampled within
each domain" -- NOT the flat/pooled cluster_bootstrap used by Stages 2-4, since a domain-level
contrast pooling instance clusters across domains could draw a domain-imbalanced resample by
chance). Decision: lower CI bound > 0 (thesis Table 5.2).

Usage:
  python scripts/phase1_analysis/stage5_h4_domain_modulation.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage5/h4_domain_modulation.json
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

from src.analysis.inference import (  # noqa: E402
    cluster_bootstrap_stratified,
    h4_diff_in_diff,
    holm,
    one_sided_bootstrap_pvalue,
)
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402


def run_h4(steps: list[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    boot = cluster_bootstrap_stratified(
        steps, lambda rs: h4_diff_in_diff(rs), strata_col="domain", n_boot=n_boot, seed=seed
    )
    p = one_sided_bootstrap_pvalue(boot["reps"], null_value=0.0)
    holm_result = holm([p], family="C")[0]
    result = {k: v for k, v in boot.items() if k != "reps"}
    result["one_sided_pvalue"] = p
    result["holm"] = holm_result
    result["decision_holds"] = boot["ci_low"] is not None and boot["ci_low"] > 0 and holm_result["adjusted"] < 0.05
    return {"family": "C", "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json")
    parser.add_argument("--output", default="data/results/phase1_analysis/stage5/h4_domain_modulation.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_path.exists():
        print(f"Stage 5 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.", file=sys.stderr)
        return 1

    ds = load_canonical_dataset_from_manifest(manifest_path)
    result = run_h4(ds.steps, n_boot=args.n_boot, seed=args.seed)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    r = result["result"]
    print(f"Stage 5 OK -- H4 written to {out_path}")
    if r.get("point") is None:
        print("  H4: could not compute (insufficient clusters per stratum)")
        return 0
    print(
        f"  H4: diff-in-diff={r['point']:.4f} CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
        f"holds={r['decision_holds']} holm_adjusted={r['holm']['adjusted']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
