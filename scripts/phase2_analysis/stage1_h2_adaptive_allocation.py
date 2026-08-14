#!/usr/bin/env python3
"""
Phase 2 real-data analysis -- Stage 1: H2 (adaptive allocation, non-inferiority plus superiority).

Per (policy, domain): episode success non-inferiority and log-output-token superiority against
Always-C2, decided on the lower bound of the paired cluster-bootstrap CI (thesis Table 5.2: "Bound
of DeltaP > -delta and bound of DeltaK_out < 0, per policy and domain"). Family B = {adaptive_tle,
adaptive_vc} x {textworld, tower_of_hanoi}, Holm-corrected -- restricted to textworld only here
(see below).

Collection status (docs/consistency_log.md, 2026-08-10 entry): always_c2/tower_of_hanoi was
aborted at 12/50 instances complete (1 partial, 37 untouched) after an OOM-driven collection
problem that repeated fix attempts did not fully resolve. textworld is complete for all six
strategies. Consequently:
  - textworld: full N=50 confirmatory comparison, both policies, Holm-corrected as Family B.
  - tower_of_hanoi: run and reported for transparency, but on a reduced, non-preregistered N
    (n_pairs typically 12-13, not 50) -- EXPLORATORY ONLY per the abort decision, not part of
    Family B's Holm correction and not to be read as a confirmatory H2 result.

Usage:
  python scripts/phase2_analysis/stage1_h2_adaptive_allocation.py \
      --checkpoint-dir data/results/phase2/phase2_stage1_20260805_UTC \
      --output data/results/phase2_analysis/stage1/h2_adaptive_allocation.json
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
from src.analysis.inference import cluster_bootstrap, h2_paired, holm  # noqa: E402

POLICIES = ("adaptive_tle", "adaptive_vc")
BASELINE = "always_c2"
CONFIRMATORY_DOMAIN = "textworld"
EXPLORATORY_DOMAIN = "tower_of_hanoi"
DOMAINS = (CONFIRMATORY_DOMAIN, EXPLORATORY_DOMAIN)


def _arm_absolute_stats(
    episodes: list[dict[str, Any]],
    *,
    domain: str,
    policy_strategy: str,
    baseline: str,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """Absolute (not paired-difference) per-arm success/token means with cluster-bootstrap CIs,
    over the same instance set h2_paired pairs on (non-holdout instances with data for both the
    policy and the baseline in this domain) -- what the Pareto plot needs, since h2_paired itself
    only returns the paired difference the confirmatory decision rests on."""
    by_inst: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ep in episodes:
        if str(ep.get("domain")) != domain or bool(ep.get("holdout")):
            continue
        inst = str(ep.get("instance"))
        strat = str(ep.get("strategy", ""))
        by_inst.setdefault(inst, {}).setdefault(strat, []).append(ep)

    paired_insts = [inst for inst, m in by_inst.items() if policy_strategy in m and baseline in m]

    def _rows_for(strategy: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for inst in paired_insts:
            runs = by_inst[inst][strategy]
            succ = sum(1 for r in runs if bool(r.get("task_success"))) / len(runs)
            toks = sum(max(1.0, float(r.get("total_tokens_generated") or 1)) for r in runs) / len(
                runs
            )
            rows.append({"instance_key": inst, "success": succ, "tokens": toks})
        return rows

    out: dict[str, dict[str, float | None]] = {}
    for strategy in (policy_strategy, baseline):
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


def run_h2(
    episodes: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    delta: float = 0.05,
) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        dom_episodes = [e for e in episodes if str(e.get("domain")) == domain]
        by_policy: dict[str, Any] = {}
        for policy in POLICIES:
            result = h2_paired(
                dom_episodes,
                policy_strategy=policy,
                baseline=BASELINE,
                delta=delta,
                n_boot=n_boot,
                seed=seed,
            )
            result["arms"] = _arm_absolute_stats(
                dom_episodes,
                domain=domain,
                policy_strategy=policy,
                baseline=BASELINE,
                n_boot=n_boot,
                seed=seed,
            )
            by_policy[policy] = result
        by_domain[domain] = by_policy

    # Family B Holm correction: textworld only (confirmatory N=50 for both policies).
    # tower_of_hanoi is excluded from the family and reported separately as exploratory --
    # correcting it alongside a preregistered N=50 family would misrepresent a reduced,
    # post-hoc-truncated sample as carrying the same confirmatory weight.
    family_b_pvalues = [by_domain[CONFIRMATORY_DOMAIN][p]["family_pvalue"] for p in POLICIES]
    holm_result = holm(family_b_pvalues, family="B")
    for i, policy in enumerate(POLICIES):
        entry = dict(holm_result[i])
        entry["reject"] = entry["adjusted"] < 0.05  # one-sided alpha=.05, thesis §5.8
        by_domain[CONFIRMATORY_DOMAIN][policy]["holm"] = entry
        by_domain[CONFIRMATORY_DOMAIN][policy]["h2_holds_confirmatory"] = (
            by_domain[CONFIRMATORY_DOMAIN][policy]["non_inferiority_holds"]
            and by_domain[CONFIRMATORY_DOMAIN][policy]["token_superiority_holds"]
            and entry["reject"]
        )
    for policy in POLICIES:
        by_domain[EXPLORATORY_DOMAIN][policy]["holm"] = None
        by_domain[EXPLORATORY_DOMAIN][policy]["status"] = "exploratory_reduced_n"

    return {
        "family": "B",
        "confirmatory_domain": CONFIRMATORY_DOMAIN,
        "exploratory_domain": EXPLORATORY_DOMAIN,
        "exploratory_reason": (
            "always_c2/tower_of_hanoi collection aborted 2026-08-10 at 12/50 instances "
            "(1 partial, 37 untouched); see docs/consistency_log.md"
        ),
        "by_domain": by_domain,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-dir", default="data/results/phase2/phase2_stage1_20260805_UTC"
    )
    parser.add_argument(
        "--output", default="data/results/phase2_analysis/stage1/h2_adaptive_allocation.json"
    )
    parser.add_argument("--figures-output", default="data/results/phase2_analysis/stage1/figures")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint_dir = (
        REPO_ROOT / args.checkpoint_dir
        if not Path(args.checkpoint_dir).is_absolute()
        else Path(args.checkpoint_dir)
    )
    if not checkpoint_dir.exists():
        print(f"Stage 1 FAILED -- checkpoint dir not found at {checkpoint_dir}", file=sys.stderr)
        return 1

    ds = load_run_dataset(checkpoint_dir)
    result = run_h2(ds.episodes, n_boot=args.n_boot, seed=args.seed, delta=args.delta)

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
        for policy in POLICIES:
            r = result["by_domain"][domain][policy]
            arms = {policy: r["arms"][policy], BASELINE: r["arms"][BASELINE]}
            tag = "confirmatory" if domain == CONFIRMATORY_DOMAIN else "exploratory"
            written_figures.update(
                plot_h2_pareto(
                    arms,
                    figures_dir,
                    name=f"{domain}_{policy}",
                    title=f"H2 Pareto ({domain}, {policy} vs {BASELINE}) [{tag}]",
                )
            )
    if written_figures:
        (figures_dir / "figures_manifest.json").write_text(
            json.dumps(written_figures, indent=2), encoding="utf-8"
        )

    print(f"Stage 1 OK -- H2 written to {out_path}")
    for domain in DOMAINS:
        tag = "CONFIRMATORY" if domain == CONFIRMATORY_DOMAIN else "EXPLORATORY (reduced N)"
        print(f"[{tag}] {domain}")
        for policy in POLICIES:
            d = result["by_domain"][domain][policy]
            if d.get("n_pairs", 0) == 0:
                print(f"  {policy}: no paired instances")
                continue
            holm_str = ""
            if d.get("holm") is not None:
                holm_str = (
                    f" holm_adjusted={d['holm']['adjusted']:.4f} reject={d['holm']['reject']}"
                )
            print(
                f"  {policy}: n_pairs={d['n_pairs']} "
                f"succ_diff={d['mean_success_diff']:.4f} [{d['success_ci_low']:.4f}, {d['success_ci_high']:.4f}] "
                f"log_tok_diff={d['mean_log_token_diff']:.4f} [{d['log_token_ci_low']:.4f}, {d['log_token_ci_high']:.4f}] "
                f"non_inf={d['non_inferiority_holds']} superior={d['token_superiority_holds']}{holm_str}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
