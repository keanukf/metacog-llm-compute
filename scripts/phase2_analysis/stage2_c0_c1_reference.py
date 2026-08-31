#!/usr/bin/env python3
"""
Phase 2 real-data analysis -- Stage 2: Always-C0 / Always-C1 exploratory reference points.

Positions the two adaptive policies against the full C0-C1-C2 compute spectrum, not just against
Always-C2. Always-C0 and Always-C1 are *not* re-collected in Phase 2 (docs/consistency_log.md,
2026-08-05 entry): the same frozen manifest/model/backend means Phase 1's fixed-stage cells (5
runs/instance) already are the Always-C0/Always-C1 arms, and reuse is unproblematic for these two
specifically because neither is part of the confirmatory H2 pairing -- unlike Always-C2, where the
same reuse was tried and rejected (a fixed 5-run baseline caps power at ~50-65% however many
adaptive runs are added; both arms of the *confirmatory* pair needed to scale together, which is
why Always-C2 alone was re-collected at n=15 in Phase 2 Stage 1).

This script is deliberately EXPLORATORY, not confirmatory: no delta-based non-inferiority decision,
no Holm correction, no held/not-held verdict. It reports the same paired success/token statistics
Stage 1 reports for Always-C2, for context, and a single five-arm Pareto plot per domain (C0, C1,
adaptive_tle, adaptive_vc, C2) so the two adaptive policies can be read against the full spectrum
rather than just its two endpoints.

Usage:
  python scripts/phase2_analysis/stage2_c0_c1_reference.py \
      --phase1-manifest data/results/phase1_analysis/stage0/canonical_manifest.json \
      --phase2-checkpoint-dir data/results/phase2/phase2_stage1_20260805_UTC \
      --output data/results/phase2_analysis/stage2/c0_c1_reference.json
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

from src.analysis.datasets import load_run_dataset  # noqa: E402
from src.analysis.inference import cluster_bootstrap, h2_paired  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

ADAPTIVE_POLICIES = ("adaptive_tle", "adaptive_vc")
REFERENCE_ARMS = ("always_c0", "always_c1")  # synthesized from Phase 1 compute_stage
SPECTRUM_ORDER = ("always_c0", "always_c1", "adaptive_tle", "adaptive_vc", "always_c2")
DOMAINS = ("textworld", "tower_of_hanoi")


def _phase1_as_always_c_episodes(phase1_episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy Phase 1 fixed-stage episodes, synthesizing a `strategy` field (`always_c0`/
    `always_c1`) from `compute_stage` so they can be pooled with Phase 2's strategy-keyed
    episodes and pass through the same pairing logic (`h2_paired`, `_arm_absolute_stats`)
    unmodified. C2 is dropped here -- Phase 2's freshly collected always_c2 is used for that arm,
    not Phase 1's, for the power reasons in the module docstring."""
    out: list[dict[str, Any]] = []
    for ep in phase1_episodes:
        stage = ep.get("compute_stage")
        if stage not in ("C0", "C1"):
            continue
        synthetic = dict(ep)
        synthetic["strategy"] = f"always_{str(stage).lower()}"
        out.append(synthetic)
    return out


def _arm_absolute_stats(
    episodes: list[dict[str, Any]],
    *,
    domain: str,
    arms: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """Absolute per-arm success/token means with cluster-bootstrap CIs, over instances that have
    data for *every* arm in ``arms`` (non-holdout only) -- so all points on one Pareto plot are
    drawn from the same instance set."""
    by_inst: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ep in episodes:
        if str(ep.get("domain")) != domain or bool(ep.get("holdout")):
            continue
        inst = str(ep.get("instance"))
        strat = str(ep.get("strategy", ""))
        by_inst.setdefault(inst, {}).setdefault(strat, []).append(ep)

    common_insts = [inst for inst, m in by_inst.items() if all(a in m for a in arms)]

    def _rows_for(strategy: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for inst in common_insts:
            runs = by_inst[inst][strategy]
            succ = sum(1 for r in runs if bool(r.get("task_success"))) / len(runs)
            toks = sum(max(1.0, float(r.get("total_tokens_generated") or 1)) for r in runs) / len(
                runs
            )
            rows.append({"instance_key": inst, "success": succ, "tokens": toks})
        return rows

    out: dict[str, dict[str, float | None]] = {"_n_common_instances": len(common_insts)}  # type: ignore[dict-item]
    for strategy in arms:
        rows = _rows_for(strategy)
        succ_boot = cluster_bootstrap(
            rows, lambda rs: sum(r["success"] for r in rs) / len(rs), n_boot=n_boot, seed=seed
        )
        tok_boot = cluster_bootstrap(
            rows, lambda rs: sum(r["tokens"] for r in rs) / len(rs), n_boot=n_boot, seed=seed
        )
        out[strategy] = {
            "n_instances": len(rows),
            "success_mean": succ_boot["point"],
            "success_ci_low": succ_boot["ci_low"],
            "success_ci_high": succ_boot["ci_high"],
            "tokens_mean": tok_boot["point"],
            "tokens_ci_low": tok_boot["ci_low"],
            "tokens_ci_high": tok_boot["ci_high"],
        }
    return out


def run(
    phase1_episodes: list[dict[str, Any]],
    phase2_episodes: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    combined = _phase1_as_always_c_episodes(phase1_episodes) + phase2_episodes

    by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        dom_episodes = [e for e in combined if str(e.get("domain")) == domain]

        # Exploratory paired contrasts: each adaptive policy vs. each of C0/C1 (descriptive --
        # no delta decision rule, no Holm correction; H2's confirmatory pairing is Stage 1's
        # adaptive-vs-Always-C2 test only).
        pairwise: dict[str, Any] = {}
        for policy in ADAPTIVE_POLICIES:
            for baseline in REFERENCE_ARMS:
                r = h2_paired(
                    dom_episodes,
                    policy_strategy=policy,
                    baseline=baseline,
                    n_boot=n_boot,
                    seed=seed,
                )
                pairwise[f"{policy}_vs_{baseline}"] = {
                    k: v
                    for k, v in r.items()
                    if k
                    in (
                        "n_pairs",
                        "mean_success_diff",
                        "mean_log_token_diff",
                        "success_ci_low",
                        "success_ci_high",
                        "log_token_ci_low",
                        "log_token_ci_high",
                    )
                }

        spectrum = _arm_absolute_stats(
            dom_episodes, domain=domain, arms=SPECTRUM_ORDER, n_boot=n_boot, seed=seed
        )
        by_domain[domain] = {"pairwise_vs_reference": pairwise, "spectrum": spectrum}

    return {"status": "exploratory_only", "by_domain": by_domain}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase1-manifest", default="data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    parser.add_argument(
        "--phase2-checkpoint-dir", default="data/results/phase2/phase2_stage1_20260805_UTC"
    )
    parser.add_argument(
        "--output", default="data/results/phase2_analysis/stage2/c0_c1_reference.json"
    )
    parser.add_argument("--figures-output", default="data/results/phase2_analysis/stage2/figures")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    p1_manifest = (
        REPO_ROOT / args.phase1_manifest
        if not Path(args.phase1_manifest).is_absolute()
        else Path(args.phase1_manifest)
    )
    p2_dir = (
        REPO_ROOT / args.phase2_checkpoint_dir
        if not Path(args.phase2_checkpoint_dir).is_absolute()
        else Path(args.phase2_checkpoint_dir)
    )
    if not p1_manifest.exists():
        print(f"Stage 2 FAILED -- Phase 1 manifest not found at {p1_manifest}", file=sys.stderr)
        return 1
    if not p2_dir.exists():
        print(f"Stage 2 FAILED -- Phase 2 checkpoint dir not found at {p2_dir}", file=sys.stderr)
        return 1

    p1_ds = load_canonical_dataset_from_manifest(p1_manifest)
    p2_ds = load_run_dataset(p2_dir)
    apply_textworld_holdout_correction(p1_ds.episodes, TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES)
    apply_textworld_holdout_correction(p2_ds.episodes, TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES)
    result = run(p1_ds.episodes, p2_ds.episodes, n_boot=args.n_boot, seed=args.seed)

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    from src.analysis.visualization import plot_h2_pareto

    figures_dir = (
        REPO_ROOT / args.figures_output
        if not Path(args.figures_output).is_absolute()
        else Path(args.figures_output)
    )
    written_figures: dict[str, str] = {}
    for domain in DOMAINS:
        spectrum = result["by_domain"][domain]["spectrum"]
        arms = {k: v for k, v in spectrum.items() if k in SPECTRUM_ORDER}
        written_figures.update(
            plot_h2_pareto(
                arms,
                figures_dir,
                name=f"{domain}_full_spectrum",
                title=f"Compute-success spectrum ({domain}): C0-C1-adaptive-C2 [exploratory]",
            )
        )
    if written_figures:
        (figures_dir / "figures_manifest.json").write_text(
            json.dumps(written_figures, indent=2), encoding="utf-8"
        )

    print(f"Stage 2 OK -- C0/C1 reference written to {out_path}")
    for domain in DOMAINS:
        print(f"[{domain}]")
        spectrum = result["by_domain"][domain]["spectrum"]
        for arm in SPECTRUM_ORDER:
            s = spectrum.get(arm, {})
            if s.get("success_mean") is None:
                print(f"  {arm}: no data")
                continue
            print(
                f"  {arm}: n={s['n_instances']} success={s['success_mean']:.3f} "
                f"[{s['success_ci_low']:.3f},{s['success_ci_high']:.3f}] "
                f"tokens={s['tokens_mean']:.0f} [{s['tokens_ci_low']:.0f},{s['tokens_ci_high']:.0f}]"
            )
        for key, r in result["by_domain"][domain]["pairwise_vs_reference"].items():
            if r.get("n_pairs", 0) == 0:
                continue
            print(
                f"  {key}: n={r['n_pairs']} succ_diff={r['mean_success_diff']:.4f} "
                f"[{r['success_ci_low']:.4f},{r['success_ci_high']:.4f}] "
                f"log_tok_diff={r['mean_log_token_diff']:.4f} "
                f"[{r['log_token_ci_low']:.4f},{r['log_token_ci_high']:.4f}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
