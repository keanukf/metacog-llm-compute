#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 4: H3 (temporal degradation).

Per domain, per signal: fit_h3_model(steps, signal=sig, domain=dom) -- the GEE clustered-logistic
signal x position_norm interaction (already stage-conditionally z-standardized per ADR-006, no
changes needed here). Decision: interaction coefficient significantly negative (thesis Table 5.2).

TextWorld is the confirmatory domain (Family E = 2 tests, TLE + VC, Holm-corrected). Tower of
Hanoi is exploratory only (short episodes, little positional resolution -- matches
scripts/analysis_rehearsal/h3_power_simulation.py's own framing) and is reported descriptively,
not Holm-corrected against the confirmatory family.

Usage:
  python scripts/phase1_analysis/stage4_h3_temporal.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage4/h3_temporal.json
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

from src.analysis.inference import fit_h3_model, holm, one_sided_wald_pvalue  # noqa: E402
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402

SIGNALS = ("tle", "vc")
CONFIRMATORY_DOMAIN = "textworld"
EXPLORATORY_DOMAIN = "tower_of_hanoi"


def run_h3(steps: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {CONFIRMATORY_DOMAIN: {}, EXPLORATORY_DOMAIN: {}}

    for dom in (CONFIRMATORY_DOMAIN, EXPLORATORY_DOMAIN):
        for sig in SIGNALS:
            fit = fit_h3_model(steps, signal=sig, domain=dom)
            if fit.get("converged"):
                coef = fit["params"].get("interaction")
                p_two = fit["pvalues"].get("interaction")
                fit["interaction_coef"] = coef
                fit["one_sided_pvalue_degradation"] = one_sided_wald_pvalue(
                    coef, p_two, direction=-1
                )
            results[dom][sig] = fit

    # Family E: TextWorld TLE + VC, Holm-corrected -- the confirmatory decision.
    pvalues = [
        results[CONFIRMATORY_DOMAIN][sig].get("one_sided_pvalue_degradation", 1.0)
        for sig in SIGNALS
    ]
    holm_result = holm(pvalues, family="E")
    for i, sig in enumerate(SIGNALS):
        r = results[CONFIRMATORY_DOMAIN][sig]
        r["holm"] = holm_result[i]
        coef = r.get("interaction_coef")
        r["decision_holds"] = (
            r.get("converged", False)
            and coef is not None
            and coef < 0
            and holm_result[i]["adjusted"] < 0.05
        )

    # Exploratory ToH: no Holm correction against the confirmatory family, report raw.
    for sig in SIGNALS:
        r = results[EXPLORATORY_DOMAIN][sig]
        r["note"] = (
            "exploratory only (short episodes, little positional resolution) -- "
            "not corrected against the confirmatory family"
        )

    return {
        "family": "E",
        "confirmatory_domain": CONFIRMATORY_DOMAIN,
        "exploratory_domain": EXPLORATORY_DOMAIN,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument("--output", default="data/results/phase1_analysis/stage4/h3_temporal.json")
    args = parser.parse_args()

    manifest_path = (
        REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    )
    if not manifest_path.exists():
        print(
            f"Stage 4 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.",
            file=sys.stderr,
        )
        return 1

    ds = load_canonical_dataset_from_manifest(manifest_path)
    result = run_h3(ds.steps)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Stage 4 OK -- H3 written to {out_path}")
    for dom, tag in ((CONFIRMATORY_DOMAIN, "confirmatory"), (EXPLORATORY_DOMAIN, "exploratory")):
        for sig in SIGNALS:
            r = result["results"][dom][sig]
            if not r.get("converged"):
                print(f"  [{tag}] {dom}/{sig}: did not converge ({r.get('note')})")
                continue
            extra = f" holds={r['decision_holds']}" if "decision_holds" in r else ""
            print(
                f"  [{tag}] {dom}/{sig}: interaction={r['interaction_coef']:.4f} "
                f"p_one_sided={r['one_sided_pvalue_degradation']:.4f}{extra}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
