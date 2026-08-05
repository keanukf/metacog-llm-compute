#!/usr/bin/env python3
"""
H2 simulation-based power check (analogue of the H3 power simulation, never previously run for H2).

Context / why this exists: `scripts/analysis_rehearsal/h3_power_simulation.py` (`docs/
gate_e_h3_power_simulation.md`) built a clustered-binary generator seeded with a real pilot ICC and
Monte-Carlo-simulated the *actual* preregistered H3 test before Phase 1 data collection. H2 --
Phase 2's confirmatory hypothesis, "adaptive policy is non-inferior on success and superior on log
output tokens vs. Always-C2" (`chapters/05_methodology.md` sec 5.6/5.8 in ../metacog-thesis,
`src/analysis/inference.py::h2_paired`) -- never got this treatment. The only existing sample-size
justification is a generic two-sample Cohen's-d=0.5 formula (`blueprints/thesis_design.md:319`,
"~34 instances needed, 50 available, ausreichend gepowert"), which the methodology chapter itself
calls "a floor rather than an estimate" (`chapters/05_methodology.md:164`) and says a simulation is
"the appropriate confirmation" -- said about H3, but the same logic applies to H2. This script runs
that check for H2, the day before the one-shot Phase 2 GPU collection (3000 episodes, not
re-runnable), so its output is a same-day go/no-go input, not archival material.

What this does, concretely:

1.  Real-data calibration (`src/analysis/phase1_canonical.py::build_canonical_dataset`, the exact
    canonical Phase 1 dataset `docs/phase1_analysis_report.md` was generated from -- 1500 episodes,
    50 instances/domain):
    - Two ICC estimates per domain, both via the shared `src/analysis/icc.py::estimate_icc`
      (GEE dep_params + ANOVA-ICC(1) cross-check), because H2's clustering unit differs from H3's:
        (a) step-level `y_optimal` clustered by `instance_key` -- this is H3's ICC, reported in
            `docs/phase1_analysis_report.md` (textworld 0.0361, tower_of_hanoi 0.0134). Kept here
            for direct comparison only -- it is NOT what H2 needs, because H2's outcome
            (`task_success`) lives at the *episode* level, not the step level.
        (b) episode-level `task_success` clustered by `instance` (real field name on episode
            records is `instance`, not `instance_key` -- `instance_key` is unset on episode rows,
            only present on step rows; confirmed empirically, see the loader call below), computed
            for the C2 compute stage specifically (Always-C2's own real stage) and for all three
            stages pooled. This is the ICC that actually governs how much a Phase-2 instance's
            *episode success* clusters across its runs -- the correct anchor for this simulation's
            shared per-instance random intercept, and it turns out to be far larger than the
            step-level H3 ICC (episode-level "some task instances are just easy/hard" clustering is
            a much stronger effect than step-level within-episode clustering): C2-stage episode-
            level ICC(GEE) = 0.205 (textworld) / 0.504 (tower_of_hanoi), vs. 0.036 / 0.013
            step-level. Using the step-level number here would have been a real, silent
            mis-calibration -- documented explicitly rather than left implicit.
    - Real C2-stage success rates (the best available real anchor for "Always-C2" specifically,
      since C2 is Always-C2's own compute stage under the frozen Phase-1/Phase-2 difficulty
      manifests) and real all-stage-pooled success rates (a more conservative/generic anchor),
      both per domain, both via `task_success` averaged over the real C2 episodes.
    - Real C2-stage token totals (`total_tokens_generated`) per domain: mean, SD, and the
      implied coefficient of variation, used to calibrate the log-normal per-episode token
      generator (CV ~= 0.54-0.57 empirically in both domains -- documented, not guessed).

2.  External anchor for the *adaptive-vs-Always-C2 token ratio*: no adaptive-policy tokens exist
    anywhere in this repo yet (Phase 2 hasn't run) -- the only real signal is yesterday's 36-episode
    GPU smoke test (n=6/strategy, reported to this script's author directly, not persisted anywhere
    queryable in-repo): adaptive_tle avg_tokens=62369 vs. always_c2 avg_tokens=194713, ratio
    ~=3.12x fewer tokens. Given n=6/strategy is too small to trust precisely, this is used as one
    ("good case") scenario alongside two more conservative, judgment-call scenarios (2.0x, 1.5x
    fewer) -- see `TOKEN_RATIO_SCENARIOS` below.

3.  Simulates the *actual* H2 test structure at the *actual* planned Phase-2 sample size (n
    instances/domain x 5 runs/condition, `configs/experiment_core.yaml` `phase2:`), per domain, per
    scenario, at n in {50 (current design), 34 (the thesis's own "floor" figure), 25 (a stress
    test)}. Per Monte Carlo replicate, per instance: one shared instance-level random intercept
    `b_i ~ N(0, sigma_b2)` (sigma_b2 from the real episode-level ICC via the standard
    logistic-normal latent-ICC conversion, reused verbatim from
    `h3_power_simulation.py::_latent_icc_to_sigma_b2` -- not reimplemented) drives *both* arms'
    success probability for that instance (this is what "paired by instance" means generatively:
    the same task instance is not independently difficult twice), giving `p_base_i` for Always-C2
    and `p_policy_i = clip(p_base_i + adaptive_delta_p, 0, 0.999)` for the adaptive policy
    (`adaptive_delta_p=0.0` = exact parity, the load-bearing case for a non-inferiority claim;
    `+0.02` run separately as a sensitivity check). 5 independent Bernoulli(p) draws per arm per
    instance are generated (the real `runs_per_condition: 5`) and averaged into one per-instance
    success-rate row -- see the flagged note in section 4 below on why this per-instance
    aggregation step, rather than raw per-episode rows, is what actually gets bootstrapped. Tokens:
    5 independent log-normal draws per arm per instance (mean = domain's real C2 token mean,
    divided by the scenario's ratio for the policy arm; CV = the real empirical C2 CV), averaged
    the same way.

4.  Runs the *real* decision rule on the resulting paired per-instance rows, using the *real*
    `src/analysis/inference.py::cluster_bootstrap` primitive with the exact statistic definitions
    and exact thresholds `h2_paired` uses (`succ_ci_low > -delta`, `log_tok_ci_low > 0`,
    `delta=0.05`) -- not a reimplementation of the bootstrap itself.

    **Finding from this simulation, since fixed:** `h2_paired` originally built its `by_inst` dict
    keyed only by `(domain, instance)` -> one episode per strategy, so calling it directly with
    multiple runs' raw episode dicts sharing the same `(domain, instance, strategy)` key meant
    later runs silently overwrote earlier ones -- only the *last* of the 5 production runs would
    ever inform the test, not an aggregate. Caught here before any real Phase 2 data existed for
    it to silently corrupt; fixed in `src/analysis/inference.py::h2_paired` the same day (now
    averages success/tokens across all runs per cell before pairing -- see docs/consistency_log.md
    2026-08-05). `verify_h2_paired_run_overwrite_behavior()` below is a same-day regression check
    confirming the fix (mirrored by `tests/analysis/test_inference.py::
    test_h2_paired_averages_all_runs_not_just_the_last`); this script still bootstraps
    pre-aggregated per-instance rows directly via `cluster_bootstrap` rather than calling the
    now-fixed `h2_paired`, since that's what was already built and verified before the fix landed.

5.  Power = fraction of replicates where both the non-inferiority and superiority bounds hold,
    at >=500 replicates/cell (default 1000; see `--n-reps`). Grid: 2 domains x 2 success-rate
    anchor scenarios (C2-specific / all-stage-pooled) x 3 token-ratio scenarios x 3 instance counts
    = 36 main cells, plus small supplementary grids at n=50 only: an `adaptive_delta_p=+0.02`
    (slight superiority) sensitivity check, and an `adaptive_delta_p=-0.10` (adaptive substantially
    *worse*) validity/sanity check that the test correctly fails non-inferiority when it should.

Usage:
    python scripts/analysis_rehearsal/h2_power_simulation.py \
        --out data/results/gate_e_h2_power/h2_power_simulation.json \
        --n-reps 1000 --n-boot 2000 --workers 7

Runtime: ~35-45 CPU-minutes total (pure-Python cluster bootstrap on ~25-50-row paired tables --
cheap per replicate, ~0.02-0.06s, but there are 44 cells x 1000 reps), parallelized across cells
with `ProcessPoolExecutor`, so wall-clock is roughly that divided by `--workers` (a few minutes on
an 8-core machine).
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# h3_power_simulation.py lives alongside this script but scripts/ is not a package (no
# __init__.py) -- reuse its intercept<->ICC conversion helper via a direct sys.path insert
# rather than reimplementing the (small but easy to get subtly wrong) logistic-normal formula.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from h3_power_simulation import _latent_icc_to_sigma_b2  # noqa: E402

from src.analysis.icc import estimate_icc  # noqa: E402
from src.analysis.inference import cluster_bootstrap, h2_paired  # noqa: E402
from src.analysis.phase1_canonical import build_canonical_dataset  # noqa: E402

DOMAINS = ("textworld", "tower_of_hanoi")
DELTA_MARGIN = (
    0.05  # H2's preregistered non-inferiority margin (chapters/05_methodology.md sec 5.6)
)
N_RUNS = 5  # phase2.runs_per_condition, configs/experiment_core.yaml

# External anchor, not derivable from any file in this repo: yesterday's 36-episode real-hardware
# GPU smoke test (6 episodes/strategy -- too small to trust precisely, hence also testing 2.0x and
# 1.5x below), reported directly by the user rather than logged to a queryable results directory.
SMOKE_TEST_ADAPTIVE_TLE_AVG_TOKENS = 62369.0
SMOKE_TEST_ALWAYS_C2_AVG_TOKENS = 194713.0
SMOKE_TEST_RATIO = SMOKE_TEST_ALWAYS_C2_AVG_TOKENS / SMOKE_TEST_ADAPTIVE_TLE_AVG_TOKENS  # ~3.12

TOKEN_RATIO_SCENARIOS: dict[str, float] = {
    "good_smoke_test_ratio": round(SMOKE_TEST_RATIO, 3),  # ~3.12x fewer tokens
    "moderate_2x": 2.0,
    "conservative_1.5x": 1.5,
}

N_INSTANCES_GRID = (50, 34, 25)  # current design / thesis's own "floor" figure / stress test


# --------------------------------------------------------------------------------------------
# 1. Real-data calibration
# --------------------------------------------------------------------------------------------


@dataclass
class DomainRealStats:
    domain: str
    n_clusters: int
    icc_step_level_h3: dict[str, Any]  # for comparison only -- NOT used to calibrate this sim
    icc_episode_level_c2: dict[str, Any]  # calibrates the C2-anchored scenario
    icc_episode_level_pooled: dict[str, Any]  # calibrates the pooled-anchored scenario
    success_rate_c2: float
    success_rate_pooled: float
    tokens_c2_mean: float
    tokens_c2_sd: float
    tokens_c2_cv: float
    n_episodes_c2: int


def compute_real_stats() -> dict[str, DomainRealStats]:
    ds = build_canonical_dataset()
    out: dict[str, DomainRealStats] = {}
    for dom in DOMAINS:
        steps = [s for s in ds.steps if str(s.get("domain")) == dom]
        icc_step = estimate_icc(steps, group_key="instance_key", value_key="y_optimal")

        eps = [e for e in ds.episodes if str(e.get("domain")) == dom]
        eps_c2 = [e for e in eps if str(e.get("compute_stage")) == "C2"]
        # NB: episode records carry the instance id in `instance`, not `instance_key`
        # (`instance_key` is a step-row-only field here, empirically confirmed empty on episodes).
        icc_ep_c2 = estimate_icc(eps_c2, group_key="instance", value_key="task_success")
        icc_ep_pooled = estimate_icc(eps, group_key="instance", value_key="task_success")

        succ_c2 = float(np.mean([float(bool(e.get("task_success"))) for e in eps_c2]))
        succ_pooled = float(np.mean([float(bool(e.get("task_success"))) for e in eps]))

        toks_c2 = np.array(
            [float(e.get("total_tokens_generated") or 0.0) for e in eps_c2], dtype=float
        )
        toks_mean = float(toks_c2.mean())
        toks_sd = float(toks_c2.std(ddof=1))

        out[dom] = DomainRealStats(
            domain=dom,
            n_clusters=int(icc_step["n_clusters"]),
            icc_step_level_h3=icc_step,
            icc_episode_level_c2=icc_ep_c2,
            icc_episode_level_pooled=icc_ep_pooled,
            success_rate_c2=succ_c2,
            success_rate_pooled=succ_pooled,
            tokens_c2_mean=toks_mean,
            tokens_c2_sd=toks_sd,
            tokens_c2_cv=toks_sd / toks_mean,
            n_episodes_c2=len(eps_c2),
        )
    return out


def verify_h2_paired_run_overwrite_behavior() -> dict[str, Any]:
    """Same-day regression check for the run-multiplicity bug this simulation found and that was
    then fixed in `h2_paired` (module docstring, section 4): feeds it 3 runs for the same
    (domain, instance, strategy) pair, with a different `task_success`/`total_tokens_generated`
    on each run, and confirms the output now reflects an average over all 3 runs -- not just the
    last one seen. Pre-fix this returned `mean_success_diff == 1.0 - 0.0 == 1.0` (last adaptive
    run True vs. last c2 run False, the other two runs of each silently dropped); post-fix it's
    the averaged `1/3 - 1/3 == 0.0`.
    """
    episodes = []
    for run in range(3):
        episodes.append(
            {
                "domain": "textworld",
                "instance": 0,
                "strategy": "adaptive_tle",
                "task_success": run == 2,  # only the last run succeeds
                "total_tokens_generated": 1000 * (run + 1),  # 1000, 2000, 3000
                "holdout": False,
            }
        )
        episodes.append(
            {
                "domain": "textworld",
                "instance": 0,
                "strategy": "always_c2",
                "task_success": run == 0,  # only the first run succeeds
                "total_tokens_generated": 5000,
                "holdout": False,
            }
        )
    result = h2_paired(episodes, n_boot=100)
    all_runs_averaged = (
        result["n_pairs"] == 1
        and abs(result["mean_success_diff"] - (1 / 3 - 1 / 3)) < 1e-9  # averaged, not last-only
    )
    return {
        "n_pairs": result["n_pairs"],
        "mean_success_diff": result["mean_success_diff"],
        "all_3_runs_averaged": all_runs_averaged,
        "note": (
            "Regression check for the h2_paired run-overwrite bug found by this simulation and "
            "fixed the same day (docs/consistency_log.md 2026-08-05): feeding 3 runs/strategy for "
            "one instance now yields mean_success_diff averaged over all 3 runs (1/3 - 1/3 = 0.0), "
            "not just the last run seen (which would have given 1.0)."
        ),
    }


# --------------------------------------------------------------------------------------------
# 2. Design cells
# --------------------------------------------------------------------------------------------


@dataclass
class DesignCell:
    domain: str
    n_instances: int
    n_runs: int
    success_scenario: str  # "c2_anchored" | "pooled_anchored"
    baseline_success_p: float
    icc_source: str
    icc: float
    sigma_b2: float
    adaptive_delta_p: float  # additive shift, baseline -> adaptive true success prob (0.0 = parity)
    token_ratio_scenario: str
    token_ratio: float  # baseline_tokens / policy_tokens
    baseline_token_mean: float
    token_cv: float
    delta_margin: float
    rationale: str


def build_design_cells(
    real_stats: dict[str, DomainRealStats],
    *,
    n_instances_grid: tuple[int, ...] = N_INSTANCES_GRID,
    adaptive_delta_p: float = 0.0,
) -> list[DesignCell]:
    cells: list[DesignCell] = []
    for dom in DOMAINS:
        s = real_stats[dom]
        success_scenarios = {
            "c2_anchored": (s.success_rate_c2, s.icc_episode_level_c2, "icc_episode_level_c2"),
            "pooled_anchored": (
                s.success_rate_pooled,
                s.icc_episode_level_pooled,
                "icc_episode_level_pooled",
            ),
        }
        for scen_name, (p, icc_dict, icc_source) in success_scenarios.items():
            icc = icc_dict["icc_gee"] if icc_dict["icc_gee"] is not None else icc_dict["icc_anova"]
            sigma_b2 = _latent_icc_to_sigma_b2(icc if icc is not None else 0.05)
            for ratio_name, ratio in TOKEN_RATIO_SCENARIOS.items():
                for n_inst in n_instances_grid:
                    cells.append(
                        DesignCell(
                            domain=dom,
                            n_instances=n_inst,
                            n_runs=N_RUNS,
                            success_scenario=scen_name,
                            baseline_success_p=p,
                            icc_source=icc_source,
                            icc=icc,
                            sigma_b2=sigma_b2,
                            adaptive_delta_p=adaptive_delta_p,
                            token_ratio_scenario=ratio_name,
                            token_ratio=ratio,
                            baseline_token_mean=s.tokens_c2_mean,
                            token_cv=s.tokens_c2_cv,
                            delta_margin=DELTA_MARGIN,
                            rationale=(
                                f"baseline_success_p={p:.3f} ({scen_name}, real Phase-1 "
                                f"{'C2-stage' if scen_name == 'c2_anchored' else 'all-stage-pooled'} "
                                f"task_success rate); ICC={icc:.4f} ({icc_source}, sigma_b2={sigma_b2:.4f} "
                                "via the logistic-normal latent-ICC conversion, real episode-level "
                                "task_success clustered by instance -- NOT the step-level H3 ICC); "
                                f"token_ratio={ratio} ({ratio_name}); baseline_token_mean="
                                f"{s.tokens_c2_mean:.0f} (real Phase-1 C2-stage total_tokens_generated); "
                                f"token_cv={s.tokens_c2_cv:.3f} (real Phase-1 C2-stage empirical CV); "
                                f"adaptive_delta_p={adaptive_delta_p:+.2f} (additive true success-prob "
                                "shift, adaptive vs. Always-C2; 0.0 = exact parity, the load-bearing "
                                "case for the non-inferiority claim)."
                            ),
                        )
                    )
    return cells


# --------------------------------------------------------------------------------------------
# 3. Monte Carlo generative model + the real H2 decision rule (top-level for multiprocessing)
# --------------------------------------------------------------------------------------------


def _simulate_paired_rows(cell: DesignCell, rng: np.random.Generator) -> list[dict[str, Any]]:
    """One synthetic (domain, scenario) dataset: for each instance, a shared random intercept
    drives both arms' true success probability (the generative meaning of "paired by instance" --
    the same task is not independently difficult twice), 5 independent Bernoulli draws per arm are
    averaged into one per-instance success rate, and 5 independent log-normal token draws per arm
    are averaged into one per-instance mean token count. One row per instance, matching the shape
    `h2_paired`'s internal statistic functions expect (see module docstring section 4 on why this
    aggregation happens here rather than inside `h2_paired` itself)."""
    sigma_ln = math.sqrt(math.log(1.0 + cell.token_cv**2))
    policy_token_mean = cell.baseline_token_mean / cell.token_ratio
    mu_base = math.log(cell.baseline_token_mean) - 0.5 * sigma_ln**2
    mu_policy = math.log(policy_token_mean) - 0.5 * sigma_ln**2
    logit_base_p = math.log(cell.baseline_success_p / (1.0 - cell.baseline_success_p))

    rows: list[dict[str, Any]] = []
    for i in range(cell.n_instances):
        b_i = rng.normal(0.0, math.sqrt(cell.sigma_b2))
        p_base = 1.0 / (1.0 + math.exp(-(logit_base_p + b_i)))
        p_policy = min(0.999, max(0.001, p_base + cell.adaptive_delta_p))

        succ_base = rng.random(cell.n_runs) < p_base
        succ_policy = rng.random(cell.n_runs) < p_policy
        tok_base = np.exp(rng.normal(mu_base, sigma_ln, cell.n_runs))
        tok_policy = np.exp(rng.normal(mu_policy, sigma_ln, cell.n_runs))

        rows.append(
            {
                "instance_key": f"{cell.domain}:{i}",
                "succ_diff": float(succ_policy.mean() - succ_base.mean()),
                "log_tok_diff": float(math.log(tok_base.mean()) - math.log(tok_policy.mean())),
            }
        )
    return rows


def _h2_decision(
    paired_rows: list[dict[str, Any]], *, delta: float, n_boot: int, seed: int
) -> dict[str, Any]:
    """Mirrors `h2_paired`'s decision-rule tail exactly (same statistic definitions, same
    thresholds, same `cluster_bootstrap` primitive) -- reused, not reimplemented; see module
    docstring section 4 for why this operates on pre-aggregated rows rather than calling
    `h2_paired` itself."""
    succ_boot = cluster_bootstrap(
        paired_rows,
        lambda rs: sum(r["succ_diff"] for r in rs) / len(rs),
        n_boot=n_boot,
        seed=seed,
    )
    log_boot = cluster_bootstrap(
        paired_rows,
        lambda rs: sum(r["log_tok_diff"] for r in rs) / len(rs),
        n_boot=n_boot,
        seed=seed + 1,
    )
    succ_ci_low = succ_boot["ci_low"]
    log_ci_low = log_boot["ci_low"]
    non_inferiority_holds = succ_ci_low is not None and succ_ci_low > -delta
    token_superiority_holds = log_ci_low is not None and log_ci_low > 0
    return {
        "non_inferiority_holds": non_inferiority_holds,
        "token_superiority_holds": token_superiority_holds,
        "h2_holds": bool(non_inferiority_holds and token_superiority_holds),
        "succ_point": succ_boot["point"],
        "succ_ci_low": succ_ci_low,
        "log_point": log_boot["point"],
        "log_ci_low": log_ci_low,
    }


def _run_one_replicate(task: dict[str, Any]) -> dict[str, Any]:
    cell = DesignCell(**task["cell"])
    rng = np.random.default_rng(task["seed"])
    rows = _simulate_paired_rows(cell, rng)
    return _h2_decision(
        rows, delta=cell.delta_margin, n_boot=task["n_boot"], seed=task["seed"] * 7 + 1
    )


# --------------------------------------------------------------------------------------------
# 4. Orchestration
# --------------------------------------------------------------------------------------------


def _cell_key(cell: DesignCell) -> str:
    return (
        f"{cell.domain}/{cell.success_scenario}/{cell.token_ratio_scenario}/"
        f"n{cell.n_instances}/dp{cell.adaptive_delta_p:+.2f}"
    )


def _run_cell(cell: DesignCell, n_reps: int, n_boot: int, base_seed: int) -> dict[str, Any]:
    outcomes = [
        _run_one_replicate({"cell": asdict(cell), "n_boot": n_boot, "seed": base_seed + rep})
        for rep in range(n_reps)
    ]
    n_ni = sum(1 for o in outcomes if o["non_inferiority_holds"])
    n_sup = sum(1 for o in outcomes if o["token_superiority_holds"])
    n_both = sum(1 for o in outcomes if o["h2_holds"])
    return {
        "cell": asdict(cell),
        "n_reps": n_reps,
        "n_boot": n_boot,
        "power_non_inferiority": n_ni / n_reps,
        "power_token_superiority": n_sup / n_reps,
        "power_h2_holds": n_both / n_reps,
    }


def _run_cell_task(task: dict[str, Any]) -> dict[str, Any]:
    cell = DesignCell(**task["cell"])
    return _run_cell(cell, task["n_reps"], task["n_boot"], task["base_seed"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="data/results/gate_e_h2_power/h2_power_simulation.json")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading canonical real Phase 1 dataset and computing real calibration stats...")
    real_stats = compute_real_stats()
    for dom, s in real_stats.items():
        print(
            f"  {dom}: n_clusters={s.n_clusters} "
            f"icc_step_h3(gee)={s.icc_step_level_h3['icc_gee']:.4f} "
            f"icc_episode_c2(gee)={s.icc_episode_level_c2['icc_gee']:.4f} "
            f"icc_episode_pooled(gee)={s.icc_episode_level_pooled['icc_gee']:.4f} "
            f"success_c2={s.success_rate_c2:.3f} success_pooled={s.success_rate_pooled:.3f} "
            f"tokens_c2_mean={s.tokens_c2_mean:.0f} tokens_c2_cv={s.tokens_c2_cv:.3f}"
        )

    print("\nVerifying h2_paired's run-multiplicity behavior (empirical, not speculative)...")
    overwrite_check = verify_h2_paired_run_overwrite_behavior()
    print(f"  {overwrite_check}")

    print(
        f"\nSmoke-test token-ratio anchor: adaptive_tle={SMOKE_TEST_ADAPTIVE_TLE_AVG_TOKENS:.0f}, "
        f"always_c2={SMOKE_TEST_ALWAYS_C2_AVG_TOKENS:.0f}, ratio={SMOKE_TEST_RATIO:.3f}x "
        "(n=6/strategy, external anchor -- not persisted in this repo)"
    )

    print("\n=== Main grid: adaptive_delta_p=0.0 (exact parity, the load-bearing NI case) ===")
    main_cells = build_design_cells(real_stats, adaptive_delta_p=0.0)
    print(f"{len(main_cells)} cells x {args.n_reps} reps x n_boot={args.n_boot}")

    print("\n=== Supplementary: adaptive_delta_p=+0.02 (slight superiority) at n=50 only ===")
    sens_cells = build_design_cells(real_stats, n_instances_grid=(50,), adaptive_delta_p=0.02)
    # Restrict the sensitivity grid to the "good" token ratio (the token axis is not the point
    # of this supplementary check).
    sens_cells = [c for c in sens_cells if c.token_ratio_scenario == "good_smoke_test_ratio"]

    print("\n=== Sanity check: adaptive_delta_p=-0.10 (adaptive substantially worse) at n=50 ===")
    sanity_cells = build_design_cells(real_stats, n_instances_grid=(50,), adaptive_delta_p=-0.10)
    sanity_cells = [c for c in sanity_cells if c.token_ratio_scenario == "good_smoke_test_ratio"]

    all_cells = main_cells + sens_cells + sanity_cells
    print(f"\nTotal cells (all grids): {len(all_cells)}")

    t0 = time.time()
    results: dict[str, dict[str, Any]] = {}
    tasks = [
        {
            "cell": asdict(cell),
            "n_reps": args.n_reps,
            "n_boot": args.n_boot,
            "base_seed": args.seed + 1_000_000 * idx,
        }
        for idx, cell in enumerate(all_cells)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_cell_task, t): t for t in tasks}
            for n_done, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                key = _cell_key(DesignCell(**res["cell"]))
                results[key] = res
                if n_done % 5 == 0 or n_done == len(tasks):
                    print(f"  [{n_done}/{len(tasks)}] {key}: power={res['power_h2_holds']:.3f}")
    else:
        for n_done, t in enumerate(tasks, start=1):
            res = _run_cell_task(t)
            key = _cell_key(DesignCell(**res["cell"]))
            results[key] = res
            print(f"  [{n_done}/{len(tasks)}] {key}: power={res['power_h2_holds']:.3f}")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "H2 simulation-based power check, run before the one-shot Phase 2 GPU collection "
            "(3000 episodes, not re-runnable). See module docstring for full method."
        ),
        "delta_margin": DELTA_MARGIN,
        "n_runs_per_condition": N_RUNS,
        "n_instances_grid": list(N_INSTANCES_GRID),
        "token_ratio_scenarios": TOKEN_RATIO_SCENARIOS,
        "smoke_test_anchor": {
            "adaptive_tle_avg_tokens": SMOKE_TEST_ADAPTIVE_TLE_AVG_TOKENS,
            "always_c2_avg_tokens": SMOKE_TEST_ALWAYS_C2_AVG_TOKENS,
            "ratio": SMOKE_TEST_RATIO,
            "n_per_strategy": 6,
            "note": "External anchor, not persisted elsewhere in this repo (see module docstring).",
        },
        "real_stats": {dom: asdict(s) for dom, s in real_stats.items()},
        "h2_paired_run_overwrite_check": overwrite_check,
        "n_reps": args.n_reps,
        "n_boot": args.n_boot,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Headline summary: current design (n=50) vs. the thesis's own floor (n=34), at parity,
    # for the good-case and conservative-case token scenarios, per domain.
    print("\n=== Headline: power at parity (adaptive_delta_p=0.0), by n_instances ===")
    for dom in DOMAINS:
        for scen in ("c2_anchored", "pooled_anchored"):
            for ratio_name in ("good_smoke_test_ratio", "conservative_1.5x"):
                line = f"{dom:15s} {scen:15s} {ratio_name:22s} "
                for n_inst in N_INSTANCES_GRID:
                    key = f"{dom}/{scen}/{ratio_name}/n{n_inst}/dp+0.00"
                    p = results.get(key, {}).get("power_h2_holds")
                    line += f" n={n_inst}:{p:.2f}" if p is not None else f" n={n_inst}:NA"
                print(line)


if __name__ == "__main__":
    main()
