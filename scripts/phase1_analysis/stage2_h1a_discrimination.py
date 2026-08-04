#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 2: H1a (discrimination).

Per domain: AUROC(TLE) vs AUROC(VC), decided on the lower bound of the cluster-bootstrapped
DeltaAUROC = AUROC(TLE) - AUROC(VC) CI (thesis Table 5.2: "Bound of DeltaAUROC (TLE-VC) > 0, per
domain"). Family A = the two domains, Holm-corrected.

Also runs the independent descriptive layer (src/analysis/calibration.py::compare_signal_
calibration -- point-estimate AUROC + Cohen's d via a completely separate code path, no
clustering/bootstrap) as an internal consistency sanity check: if the confirmatory and
descriptive point estimates disagree by a lot, that's worth noticing before trusting either.

Usage:
  python scripts/phase1_analysis/stage2_h1a_discrimination.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage2/h1a_discrimination.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compare_signal_calibration  # noqa: E402
from src.analysis.inference import (  # noqa: E402
    cluster_bootstrap,
    delta_auroc,
    holm,
    one_sided_bootstrap_pvalue,
)
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402

DOMAINS = ("tower_of_hanoi", "textworld")


def run_h1a(
    steps: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    on_bootstrap: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """``on_bootstrap(domain, boot)`` fires once per domain with the raw bootstrap dict
    (including ``reps``) before it's stripped for the returned/persisted result -- lets
    ``main()`` plot the replicate distribution without recomputing the bootstrap or changing
    this function's own no-``reps``-in-the-output contract."""
    by_domain: dict[str, dict[str, Any]] = {}
    pvalues: list[float] = []
    for dom in DOMAINS:
        dom_steps = [r for r in steps if str(r.get("domain")) == dom]
        boot = cluster_bootstrap(dom_steps, lambda rs: delta_auroc(rs), n_boot=n_boot, seed=seed)
        if on_bootstrap is not None:
            on_bootstrap(dom, boot)
        holds = boot["ci_low"] is not None and boot["ci_low"] > 0
        p = one_sided_bootstrap_pvalue(boot["reps"], null_value=0.0)
        by_domain[dom] = {k: v for k, v in boot.items() if k != "reps"}
        by_domain[dom]["decision_holds"] = holds
        by_domain[dom]["one_sided_pvalue"] = p
        pvalues.append(p)

    holm_result = holm(pvalues, family="A")
    for i, dom in enumerate(DOMAINS):
        by_domain[dom]["holm"] = holm_result[i]

    # Per-domain, not pooled -- compare_signal_calibration has no domain filter of its own, and
    # a pooled call would blend e.g. a strong tower_of_hanoi signal with a near-null textworld
    # one, making the "descriptive vs. confirmatory" comparison meaningless (apples to a fruit
    # salad). Filter episodes to each domain first, matching how preanalysis_screen.py's
    # _discrimination_for_domain already does this for the same reason.
    descriptive = {
        dom: compare_signal_calibration([e for e in episodes if str(e.get("domain")) == dom])
        for dom in DOMAINS
    }

    return {"family": "A", "by_domain": by_domain, "descriptive_cross_check": descriptive}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json")
    parser.add_argument("--output", default="data/results/phase1_analysis/stage2/h1a_discrimination.json")
    parser.add_argument("--figures-output", default="data/results/phase1_analysis/stage2/figures")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_path.exists():
        print(f"Stage 2 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.", file=sys.stderr)
        return 1

    from src.analysis.visualization import plot_bootstrap_distribution

    figures_dir = (
        REPO_ROOT / args.figures_output
        if not Path(args.figures_output).is_absolute()
        else Path(args.figures_output)
    )
    written_figures: dict[str, str] = {}

    def _on_bootstrap(dom: str, boot: dict[str, Any]) -> None:
        written_figures.update(
            plot_bootstrap_distribution(
                boot["reps"],
                figures_dir,
                name=f"h1a_{dom}",
                point=boot.get("point"),
                ci_low=boot.get("ci_low"),
                ci_high=boot.get("ci_high"),
                null_value=0.0,
                title=f"H1a bootstrap distribution ({dom}): DeltaAUROC(TLE,VC)",
            )
        )

    ds = load_canonical_dataset_from_manifest(manifest_path)
    result = run_h1a(ds.steps, ds.episodes, n_boot=args.n_boot, seed=args.seed, on_bootstrap=_on_bootstrap)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if written_figures:
        (figures_dir / "figures_manifest.json").write_text(json.dumps(written_figures, indent=2), encoding="utf-8")

    print(f"Stage 2 OK -- H1a written to {out_path}")
    for dom in DOMAINS:
        d = result["by_domain"][dom]
        desc = result["descriptive_cross_check"][dom]["optimal_only"]
        desc_delta = (desc["tle"]["auroc"] or 0.0) - (desc["vc"]["auroc"] or 0.0)
        print(
            f"  {dom}: DeltaAUROC={d['point']:.4f} CI=[{d['ci_low']:.4f}, {d['ci_high']:.4f}] "
            f"holds={d['decision_holds']} holm_adjusted={d['holm']['adjusted']:.4f} "
            f"| descriptive_delta={desc_delta:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
