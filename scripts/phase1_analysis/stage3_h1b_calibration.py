#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 3: H1b (calibration).

Per domain: fit a logistic TLE->probability calibrator on that domain's HOLDOUT steps (pooled
across runs and compute stages -- thesis Ch.5 §5.9, deliberately not stage-conditional like H3),
then evaluate the confirmatory DeltaBrier = Brier(TLE-mapped) - Brier(VC/100) on that domain's
NON-HOLDOUT steps via cluster_bootstrap, decided on the CI upper bound being fully below 0
(thesis Table 5.2: "Bound of DeltaBrier (TLE-VC) < 0, per domain"). Family D = the two domains,
Holm-corrected.

Note: src.analysis.inference.delta_brier_after_mapping computes only Brier(TLE-mapped) despite
its name -- not a difference. The actual DeltaBrier vs. VC/100 is computed here directly (pairwise
VC-missing exclusion, per thesis §5.2.2/§2.5 of the stats plan: a step needs both TLE and VC
present to enter this specific comparison).

Usage:
  python scripts/phase1_analysis/stage3_h1b_calibration.py \
      --manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --output data/results/phase1_analysis/stage3/h1b_calibration.json
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

from src.analysis.calibration import compute_brier, fit_tle_calibrator, vc_to_prob  # noqa: E402
from src.analysis.inference import cluster_bootstrap, holm, one_sided_bootstrap_pvalue  # noqa: E402
from src.analysis.phase1_canonical import load_canonical_dataset_from_manifest  # noqa: E402

DOMAINS = ("tower_of_hanoi", "textworld")


def _predictions_for_domain(
    rows: list[dict[str, Any]], calibrator: Any
) -> tuple[list[int], list[float], list[float]]:
    """Pairwise VC-missing exclusion (thesis §5.2.2/§2.5): a step needs both TLE and VC present to
    enter this comparison. Shared by the confirmatory Brier-delta stat_fn and the reliability
    diagrams so both use the exact same evaluation subset."""
    ys: list[int] = []
    ps_tle: list[float] = []
    ps_vc: list[float] = []
    for r in rows:
        y = r.get("y_optimal")
        tle = r.get("tle_mean_entropy")
        vc = r.get("vc")
        if y is None or tle is None or vc is None:
            continue
        ys.append(int(y))
        ps_tle.append(calibrator.predict_proba(float(tle)))
        ps_vc.append(vc_to_prob(float(vc)))
    return ys, ps_tle, ps_vc


def _brier_delta_stat_fn(rows: list[dict[str, Any]], calibrator: Any) -> float:
    ys, ps_tle, ps_vc = _predictions_for_domain(rows, calibrator)
    if not ys:
        return float("nan")
    return compute_brier(ps_tle, ys) - compute_brier(ps_vc, ys)


def run_h1b(
    steps: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    on_bootstrap: Callable[[str, dict[str, Any]], None] | None = None,
    on_calibrator_fit: Callable[[str, Any, list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """``on_bootstrap(domain, boot)`` mirrors stage2's hook (raw bootstrap dict incl. ``reps``
    before stripping). ``on_calibrator_fit(domain, calibrator, non_holdout_steps)`` fires once a
    domain's calibrator has converged, before the bootstrap -- lets ``main()`` render reliability
    diagrams against the exact same non-holdout evaluation subset without duplicating the
    fit/filter logic."""
    by_domain: dict[str, dict[str, Any]] = {}
    pvalues: list[float] = []
    for dom in DOMAINS:
        dom_steps = [r for r in steps if str(r.get("domain")) == dom]
        holdout_steps = [r for r in dom_steps if bool(r.get("holdout"))]
        non_holdout_steps = [r for r in dom_steps if not bool(r.get("holdout"))]

        calibrator = fit_tle_calibrator(holdout_steps)
        if isinstance(calibrator, dict):  # fit failed -- record and skip this domain's decision
            by_domain[dom] = {
                "calibrator_converged": False,
                "calibrator_note": calibrator.get("note"),
                "point": None,
                "ci_low": None,
                "ci_high": None,
                "decision_holds": False,
                "one_sided_pvalue": 1.0,
            }
            pvalues.append(1.0)
            continue

        if on_calibrator_fit is not None:
            on_calibrator_fit(dom, calibrator, non_holdout_steps)

        boot = cluster_bootstrap(
            non_holdout_steps,
            lambda rs: _brier_delta_stat_fn(rs, calibrator),
            n_boot=n_boot,
            seed=seed,
        )
        if on_bootstrap is not None:
            on_bootstrap(dom, boot)
        holds = boot["ci_high"] is not None and boot["ci_high"] < 0
        # one-sided test is "DeltaBrier < 0"; one_sided_bootstrap_pvalue tests ">", so negate.
        p = one_sided_bootstrap_pvalue([-r for r in boot["reps"]], null_value=0.0)
        by_domain[dom] = {k: v for k, v in boot.items() if k != "reps"}
        by_domain[dom]["calibrator_converged"] = True
        by_domain[dom]["calibrator_intercept"] = calibrator.intercept
        by_domain[dom]["calibrator_slope"] = calibrator.slope
        by_domain[dom]["n_holdout_steps"] = len(holdout_steps)
        by_domain[dom]["decision_holds"] = holds
        by_domain[dom]["one_sided_pvalue"] = p
        pvalues.append(p)

    holm_result = holm(pvalues, family="D")
    for i, dom in enumerate(DOMAINS):
        by_domain[dom]["holm"] = holm_result[i]

    return {"family": "D", "by_domain": by_domain}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument(
        "--output", default="data/results/phase1_analysis/stage3/h1b_calibration.json"
    )
    parser.add_argument("--figures-output", default="data/results/phase1_analysis/stage3/figures")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest_path = (
        REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    )
    if not manifest_path.exists():
        print(
            f"Stage 3 FAILED -- manifest not found at {manifest_path}; run Stage 0 first.",
            file=sys.stderr,
        )
        return 1

    from src.analysis.visualization import plot_bootstrap_distribution, reliability_diagram

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
                name=f"h1b_{dom}",
                point=boot.get("point"),
                ci_low=boot.get("ci_low"),
                ci_high=boot.get("ci_high"),
                null_value=0.0,
                title=f"H1b bootstrap distribution ({dom}): DeltaBrier(TLE-mapped, VC)",
            )
        )

    def _on_calibrator_fit(
        dom: str, calibrator: Any, non_holdout_steps: list[dict[str, Any]]
    ) -> None:
        # Reliability diagrams evaluated on the exact same non-holdout subset ΔBrier is computed
        # on (via _predictions_for_domain's pairwise VC-missing exclusion) -- not the full,
        # unfiltered step table, so the plot matches what the confirmatory number is actually
        # about.
        figures_dir.mkdir(parents=True, exist_ok=True)
        ys, ps_tle, ps_vc = _predictions_for_domain(non_holdout_steps, calibrator)
        if not ys:
            return
        tle_path = figures_dir / f"reliability_tle_mapped_{dom}.png"
        reliability_diagram(
            ps_tle, ys, save_path=str(tle_path), title=f"Reliability: TLE-mapped ({dom})"
        )
        written_figures[f"reliability_tle_mapped_{dom}"] = str(tle_path)
        vc_path = figures_dir / f"reliability_vc_{dom}.png"
        reliability_diagram(ps_vc, ys, save_path=str(vc_path), title=f"Reliability: VC/100 ({dom})")
        written_figures[f"reliability_vc_{dom}"] = str(vc_path)

    ds = load_canonical_dataset_from_manifest(manifest_path)
    result = run_h1b(
        ds.steps,
        n_boot=args.n_boot,
        seed=args.seed,
        on_bootstrap=_on_bootstrap,
        on_calibrator_fit=_on_calibrator_fit,
    )

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if written_figures:
        (figures_dir / "figures_manifest.json").write_text(
            json.dumps(written_figures, indent=2), encoding="utf-8"
        )

    print(f"Stage 3 OK -- H1b written to {out_path}")
    for dom in DOMAINS:
        d = result["by_domain"][dom]
        if not d.get("calibrator_converged"):
            print(f"  {dom}: calibrator did not converge ({d.get('calibrator_note')})")
            continue
        print(
            f"  {dom}: DeltaBrier={d['point']:.4f} CI=[{d['ci_low']:.4f}, {d['ci_high']:.4f}] "
            f"holds={d['decision_holds']} holm_adjusted={d['holm']['adjusted']:.4f} "
            f"(n_holdout_steps={d['n_holdout_steps']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
