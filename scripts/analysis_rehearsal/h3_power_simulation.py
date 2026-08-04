#!/usr/bin/env python3
"""
Gate E (WEICH) — H3 simulation-based power check.

`blueprints/gate_p1_readiness.md` (Gate E, "H3-Power-Simulation") and thesis §5.8
("Effect sizes, power, and reporting") call for a simulation-based power check for the H3
signal x position_norm interaction, seeded with the pilot ICC and entropy distribution,
because the interaction test (clustered logistic / GEE, instance-level clustering) has no
closed-form power formula and the pilot itself (9-12 clusters/domain) cannot estimate the
interaction with useful precision (§5.9 names this explicitly, citing X. Zhao 2026 on
underpowered step-indexed interaction tests).

What this does, concretely:

1.  Loads the real Gate-C pilot run (`data/results/instrument_validation/phase1_20260714_105004`,
    72 episodes) via `src/analysis/datasets.py::load_run_dataset` (same loader Gate E's rehearsal
    used) and computes, per domain:
    - the instance-level ICC of the binary step outcome `y_optimal`, two ways: (a) a GEE
      intercept-only model with an Exchangeable working correlation (`statsmodels`), whose
      `dep_params` *is* the ICC a GEE-based confirmatory analysis would itself estimate --
      methodologically the most consistent choice, since H3's actual test is a GEE with the
      same working-correlation structure; (b) a classical ANOVA-based ICC(1) on the 0/1 outcome
      as a cross-check (Eldridge et al., cluster-RCT convention for binary outcomes).
    - the empirical TLE (`tle_mean_entropy`) and VC distributions (mean/SD) per domain.
    - the real (noisy, n=12 clusters/domain) H3 interaction point estimate via the *actual*
      production model, `src/analysis/inference.py::fit_h3_model`, reported for context only --
      not used as a "true effect" (see report caveats).

2.  Monte-Carlo simulates clustered step-level binary outcomes under the planned Phase-1 design
    (50 instances/domain, 5 runs/condition x 3 compute stages pooled = 15 episodes/instance,
    per `configs/experiment_core.yaml` `phase1:`), with a logistic random-intercept generative
    model calibrated to the pilot's ICC, base rate, and signal-outcome relationship, across a
    grid of true interaction effect sizes (in standardized-signal-per-position_norm-unit logit
    units). Each simulated dataset is fit with the *actual* `fit_h3_model` (same GEE call the
    real confirmatory analysis will use) -- this is a real Monte Carlo power simulation, not a
    closed-form approximation dressed up as one.

3.  Reports empirical rejection rate (power) at the preregistered one-sided alpha=.05, and at
    the Holm-conservative alpha=.025 (H3 is its own family of 2 tests -- TLE and VC -- per
    §5.8's family grouping; under Holm with m=2, the stricter member of the pair needs
    alpha/2), interpolates the effect size at which power crosses 80%, and states explicit,
    inspectable assumptions for every simulation parameter that is not read directly off the
    pilot data.

Usage:
    python scripts/analysis_rehearsal/h3_power_simulation.py \
        --run-dir data/results/instrument_validation/phase1_20260714_105004 \
        --out data/results/gate_e_h3_power/h3_power_simulation.json \
        --n-reps 200 --workers 7

Runtime: full grid (TLE both domains + VC secondary grid) takes on the order of 15-30 minutes
on an 8-core laptop with `--workers 7`; each replicate is one real `statsmodels` GEE fit on a
simulated dataset of several thousand rows, which is the actual cost driver.
"""

from __future__ import annotations

import os

# Must precede numpy/statsmodels import: avoid BLAS thread oversubscription once we fan out
# replicate fits across multiple worker *processes* (each process would otherwise also try to
# multithread its own BLAS calls, thrashing an 8-core machine).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.datasets import load_run_dataset  # noqa: E402
from src.analysis.icc import anova_icc1, gee_icc  # noqa: E402
from src.analysis.inference import fit_h3_model  # noqa: E402

DOMAINS = ("textworld", "tower_of_hanoi")
Z975 = 1.959963984540054  # for latent-ICC <-> variance conversion cross-checks, unused directly


# --------------------------------------------------------------------------------------------
# 1. Pilot-seeded statistics
# --------------------------------------------------------------------------------------------


@dataclass
class DomainPilotStats:
    domain: str
    n_steps: int
    n_clusters: int
    base_rate: float
    icc_gee: float | None
    icc_anova: float | None
    tle_mean: float
    tle_sd: float
    vc_mean: float
    vc_sd: float
    episode_lengths: list[int]
    h3_pilot_tle: dict[str, Any]
    h3_pilot_vc: dict[str, Any]


def load_toh_frozen_corridor_lengths(run_dir: Path, num_disks: int) -> list[int]:
    """Real episode lengths for the frozen ToH corridor (C1, ``num_disks`` disks), read straight
    off actual per-episode result JSONs rather than the stale pre-calibration Gate-C pilot.

    ``num_disks`` isn't stored as its own field on these episode records, but every step's judge
    prompt states it verbatim ("Goal state: Peg C holds all N disks, ..."), so it's recovered by
    regex instead of re-deriving it from peg contents.
    """
    import re

    lengths: list[int] = []
    for f in sorted(Path(run_dir).glob("*_C1_*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        nd = None
        for step in data.get("steps_detail") or []:
            text = step.get("vc_prompt") or step.get("reason_prompt") or ""
            m = re.search(r"holds all (\d+) disks", text)
            if m:
                nd = int(m.group(1))
                break
        if nd != num_disks:
            continue
        length = data.get("episode_length_steps") or data.get("steps")
        if length:
            lengths.append(int(length))
    return lengths


# _anova_icc1 / _gee_icc used to be defined here; lifted 2026-07-28 into the shared, public,
# independently-tested src/analysis/icc.py (revision_audit P1-stat-8) so the real Stage 1
# preanalysis screen can use the exact same estimators, not a second reimplementation.
_anova_icc1 = anova_icc1
_gee_icc = gee_icc


def compute_pilot_seed_stats(run_dir: Path) -> dict[str, DomainPilotStats]:
    ds = load_run_dataset(run_dir)
    out: dict[str, DomainPilotStats] = {}
    ep_lengths_by_domain: dict[str, list[int]] = {d: [] for d in DOMAINS}
    for ep in ds.episodes:
        dom = str(ep.get("domain"))
        if dom not in ep_lengths_by_domain:
            continue
        length = ep.get("episode_length_steps") or ep.get("steps")
        if length:
            ep_lengths_by_domain[dom].append(int(length))

    for dom in DOMAINS:
        dom_rows = [r for r in ds.steps if str(r.get("domain")) == dom]
        y_rows = [r for r in dom_rows if r.get("y_optimal") is not None]
        n_clusters = len({str(r.get("instance_key")) for r in y_rows})
        base_rate = (
            float(np.mean([float(r["y_optimal"]) for r in y_rows])) if y_rows else float("nan")
        )
        icc_gee = _gee_icc(y_rows, "instance_key", "y_optimal")
        icc_anova = _anova_icc1(y_rows, "instance_key", "y_optimal")
        tle_vals = [
            float(r["tle_mean_entropy"]) for r in dom_rows if r.get("tle_mean_entropy") is not None
        ]
        vc_vals = [float(r["vc"]) for r in dom_rows if r.get("vc") is not None]
        h3_tle = fit_h3_model(ds.steps, signal="tle", domain=dom)
        h3_vc = fit_h3_model(ds.steps, signal="vc", domain=dom)
        out[dom] = DomainPilotStats(
            domain=dom,
            n_steps=len(y_rows),
            n_clusters=n_clusters,
            base_rate=base_rate,
            icc_gee=icc_gee,
            icc_anova=icc_anova,
            tle_mean=float(np.mean(tle_vals)) if tle_vals else float("nan"),
            tle_sd=float(np.std(tle_vals, ddof=1)) if len(tle_vals) > 1 else float("nan"),
            vc_mean=float(np.mean(vc_vals)) if vc_vals else float("nan"),
            vc_sd=float(np.std(vc_vals, ddof=1)) if len(vc_vals) > 1 else float("nan"),
            episode_lengths=ep_lengths_by_domain[dom],
            h3_pilot_tle=h3_tle,
            h3_pilot_vc=h3_vc,
        )
    return out


# --------------------------------------------------------------------------------------------
# 2. Simulation design: generative model + explicit assumptions
# --------------------------------------------------------------------------------------------


@dataclass
class DesignCell:
    """One (domain, signal) simulation configuration. Every field not read straight off the
    pilot data is a documented assumption (see ``rationale``)."""

    domain: str
    signal: str  # "tle" | "vc"
    n_instances: int
    episodes_per_instance: int  # runs_per_condition x compute_stages, pooled across stages
    length_mode: str  # "uniform_15_40" | "bootstrap_pilot"
    length_bootstrap_pool: list[int]
    target_base_rate: float
    beta_z: float  # main effect of standardized signal, per SD
    beta_pos: float  # main effect of position_norm on Y (nuisance, not the H3 target)
    sigma_b2: float  # instance random-intercept variance (from ICC)
    icc_source: str
    rationale: str


def _latent_icc_to_sigma_b2(icc: float) -> float:
    """Logistic-normal (latent-variable) ICC conversion (Larsen & Merlo 2005 / standard
    multilevel-logistic convention): icc = sigma_b^2 / (sigma_b^2 + pi^2/3)."""
    icc = min(max(icc, 1e-4), 0.95)
    return icc / (1.0 - icc) * (math.pi**2 / 3.0)


def build_design_cells(
    pilot: dict[str, DomainPilotStats],
    instances_per_domain: int,
    runs_per_condition: int,
    compute_stages: int,
    toh_frozen_corridor_lengths: list[int] | None = None,
) -> dict[tuple[str, str], DesignCell]:
    cells: dict[tuple[str, str], DesignCell] = {}
    episodes_per_instance = runs_per_condition * compute_stages

    # Main-effect (beta_z) anchors: pilot GEE z_c coefficient (raw, un-standardized) x pilot SD
    # of the raw signal gives an approximate per-SD standardized coefficient; cross-checked
    # against a Cohen's-d -> logistic-coefficient conversion (beta ~= d * pi/sqrt(3)) from the
    # pilot's own AUROC/Cohen's-d screen (`preanalysis_screen.json`). Both pilot point
    # estimates are noisy (n=12 clusters/domain) -- rounded to one decimal as a judgment call,
    # not read off verbatim.
    anchors = {
        ("textworld", "tle"): 0.30,
        ("textworld", "vc"): 0.25,
        ("tower_of_hanoi", "tle"): 1.20,
        ("tower_of_hanoi", "vc"): 1.00,
    }
    pos_anchors = {"textworld": -0.6, "tower_of_hanoi": -0.9}

    for dom in DOMAINS:
        stats = pilot[dom]
        icc = stats.icc_gee if stats.icc_gee is not None else stats.icc_anova
        icc_source = "gee_dep_params" if stats.icc_gee is not None else "anova_icc1"
        sigma_b2 = _latent_icc_to_sigma_b2(icc if icc is not None else 0.05)
        if dom == "textworld":
            length_mode = "uniform_15_40"
            rationale_len = (
                "Gate D's TextWorld difficulty-sweep target corridor as revised 2026-07-18 "
                "(blueprints/gate_p1_readiness.md, docs/consistency_log.md: 15-40 steps/episode, "
                "only the 15-step floor load-bearing for H3 positional resolution, 40 a soft "
                "practical ceiling) -- supersedes the original a-priori 8-15 figure used in the "
                "first run of this simulation (2026-07-17), which real post-fix sweeps showed was "
                "empirically almost unreachable (winning episodes cluster at 15-40+ steps, not 8-15) "
                "rather than the raw pilot distribution, whose episodes cluster near the "
                f"{stats.episode_lengths and max(stats.episode_lengths)}-step cap because Gate D "
                "difficulty calibration was not yet applied when the pilot ran (pre-calibration "
                "episodes are harder/longer than the calibrated design will be)."
            )
        elif toh_frozen_corridor_lengths:
            length_mode = "bootstrap_toh_frozen_corridor"
            n_pool = len(toh_frozen_corridor_lengths)
            mean_len = sum(toh_frozen_corridor_lengths) / n_pool
            rationale_len = (
                f"Bootstrap-resampling {n_pool} real episode lengths from the frozen ToH corridor "
                "itself (4 disks, C1 reference, partial_start_mode=random_scramble -- "
                "data/results/gate_d_calibration/toh_corridor_scramble_n30, the isolated 4-disk/C1 "
                "subset documented in docs/consistency_log.md 2026-07-19), superseding the earlier "
                "run's bootstrap from the raw, not-yet-difficulty-calibrated Gate-C pilot lengths "
                f"(mean ~18, many near the 20-step cap). Mean length in this pool: {mean_len:.1f} "
                f"steps. Caveat: n={n_pool} is a small resampling pool (few distinct values), "
                "smaller than the original cross-domain pilot pool but on-target for the actual "
                "frozen config rather than off-target on a larger, stale one."
            )
        else:
            length_mode = "bootstrap_pilot"
            rationale_len = (
                "No explicit ToH episode-length corridor is specified (only a 30-50% C0-success target); "
                "bootstrap-resampling the pilot's own episode-length distribution is the most defensible "
                "available anchor, with the caveat that ToH's Gate-D disk-count calibration is also not "
                "yet frozen, so these lengths (mean ~18, many near the 20-step cap) may shorten once "
                "calibrated."
            )
        length_pool = (
            toh_frozen_corridor_lengths
            if (dom == "tower_of_hanoi" and toh_frozen_corridor_lengths)
            else stats.episode_lengths
        )
        for sig in ("tle", "vc"):
            beta_z = anchors[(dom, sig)]
            cells[(dom, sig)] = DesignCell(
                domain=dom,
                signal=sig,
                n_instances=instances_per_domain,
                episodes_per_instance=episodes_per_instance,
                length_mode=length_mode,
                length_bootstrap_pool=list(length_pool),
                target_base_rate=stats.base_rate,
                beta_z=beta_z,
                beta_pos=pos_anchors[dom],
                sigma_b2=sigma_b2,
                icc_source=icc_source,
                rationale=(
                    f"beta_z={beta_z} (standardized-signal main effect, per-SD logit units) anchored to "
                    f"the pilot's own noisy GEE point estimate + a Cohen's-d-derived cross-check "
                    f"(preanalysis_screen.json); beta_pos={pos_anchors[dom]} (position main effect, "
                    "nuisance parameter, not the H3 target) anchored to the pilot's p_c coefficients; "
                    f"ICC={icc if icc is not None else float('nan'):.4f} from {icc_source} "
                    f"(sigma_b^2={sigma_b2:.4f} via the standard logistic-normal latent-ICC conversion). "
                    f"Episode length: {rationale_len}"
                ),
            )
    return cells


# --------------------------------------------------------------------------------------------
# 3. Monte Carlo generative model + replicate fitting (top-level for multiprocessing)
# --------------------------------------------------------------------------------------------


def _sample_episode_length(rng: np.random.Generator, cell: DesignCell) -> int:
    if cell.length_mode == "uniform_15_40":
        return int(rng.integers(15, 41))
    pool = cell.length_bootstrap_pool
    return int(rng.choice(pool)) if pool else 15


def _simulate_rows(
    cell: DesignCell, beta_int: float, beta0: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(cell.n_instances):
        b_i = rng.normal(0.0, math.sqrt(cell.sigma_b2))
        instance_key = f"{cell.domain}:{i}"
        for _e in range(cell.episodes_per_instance):
            length = _sample_episode_length(rng, cell)
            denom = max(length - 1, 1)
            z_vals = rng.normal(0.0, 1.0, size=length)
            for t in range(length):
                pos = t / denom
                z = float(z_vals[t])
                logit = beta0 + cell.beta_z * z + cell.beta_pos * pos + beta_int * z * pos + b_i
                p = 1.0 / (1.0 + math.exp(-logit))
                y = int(rng.random() < p)
                row: dict[str, Any] = {
                    "y_optimal": y,
                    "position_norm": pos,
                    "domain": cell.domain,
                    "instance_key": instance_key,
                }
                if cell.signal == "tle":
                    row["tle_mean_entropy"] = -z  # fit_h3_model negates TLE -> recovers z
                    row["vc"] = None
                else:
                    row["vc"] = z  # fit_h3_model uses VC as-is -> recovers z
                    row["tle_mean_entropy"] = None
                rows.append(row)
    return rows


def _calibrate_intercept(
    cell: DesignCell,
    beta_int: float,
    *,
    n_probe_instances: int = 20,
    n_probe_episodes: int = 8,
    iters: int = 40,
    seed: int = 999,
) -> float:
    """Bisection on beta0 so the *analytic* mean of expit(logit) (no binomial sampling noise)
    matches the pilot base rate, using a moderate-size probe population. Ignores beta_int's
    (small) effect on the marginal mean -- calibrated once per (cell, beta_int=0) and reused
    across the whole effect-size grid, which is documented as a simplification below."""
    rng = np.random.default_rng(seed)
    b_i = rng.normal(0.0, math.sqrt(cell.sigma_b2), size=n_probe_instances)
    logits_fixed = []
    for bi in b_i:
        for _e in range(n_probe_episodes):
            length = _sample_episode_length(rng, cell)
            denom = max(length - 1, 1)
            z_vals = rng.normal(0.0, 1.0, size=length)
            for t in range(length):
                pos = t / denom
                z = float(z_vals[t])
                logits_fixed.append(
                    (cell.beta_z * z + cell.beta_pos * pos + beta_int * z * pos + bi)
                )
    logits_fixed_arr = np.array(logits_fixed)

    def mean_p(beta0: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(beta0 + logits_fixed_arr)))))

    lo, hi = -10.0, 5.0
    target = cell.target_base_rate
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if mean_p(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _run_one_replicate(task: dict[str, Any]) -> dict[str, Any]:
    cell = DesignCell(**task["cell"])
    beta_int = task["beta_int"]
    beta0 = task["beta0"]
    seed = task["seed"]
    rng = np.random.default_rng(seed)
    rows = _simulate_rows(cell, beta_int, beta0, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_h3_model(rows, signal=cell.signal, domain=cell.domain)
    if not result.get("converged"):
        return {"converged": False}
    coef = result["params"].get("interaction")
    pval_two_sided = result["pvalues"].get("interaction")
    if (
        coef is None
        or pval_two_sided is None
        or not math.isfinite(coef)
        or not math.isfinite(pval_two_sided)
    ):
        return {"converged": False}
    # H1,3: interaction coefficient significantly < 0 (degradation). One-sided p-value: half
    # the two-sided p if the sign matches the predicted direction, else 1 - p/2 (evidence points
    # the wrong way).
    p_one_sided = pval_two_sided / 2.0 if coef < 0 else 1.0 - pval_two_sided / 2.0
    return {"converged": True, "coef": coef, "p_one_sided": p_one_sided}


# --------------------------------------------------------------------------------------------
# 4. Orchestration
# --------------------------------------------------------------------------------------------

ALPHA_NOMINAL = 0.05
ALPHA_HOLM2 = 0.025  # Holm, family of 2 (TLE + VC in H3's own family, §5.8), stricter member


def run_power_grid(
    cell: DesignCell,
    effect_grid: list[float],
    n_reps: int,
    workers: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    beta0 = _calibrate_intercept(cell, 0.0, seed=base_seed)
    results = []
    for beta_int in effect_grid:
        tasks = [
            {
                "cell": asdict(cell),
                "beta_int": beta_int,
                "beta0": beta0,
                "seed": base_seed + 100_000 * (effect_grid.index(beta_int) + 1) + rep,
            }
            for rep in range(n_reps)
        ]
        outcomes = []
        t0 = time.time()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one_replicate, t) for t in tasks]
                for f in as_completed(futs):
                    outcomes.append(f.result())
        else:
            outcomes = [_run_one_replicate(t) for t in tasks]
        elapsed = time.time() - t0
        converged = [o for o in outcomes if o.get("converged")]
        n_conv = len(converged)
        rej_nominal = sum(1 for o in converged if o["p_one_sided"] < ALPHA_NOMINAL)
        rej_holm = sum(1 for o in converged if o["p_one_sided"] < ALPHA_HOLM2)
        mean_coef = float(np.mean([o["coef"] for o in converged])) if converged else None
        results.append(
            {
                "beta_int_true": beta_int,
                "n_reps": n_reps,
                "n_converged": n_conv,
                "convergence_rate": n_conv / n_reps if n_reps else None,
                "power_alpha_nominal_0.05": rej_nominal / n_conv if n_conv else None,
                "power_alpha_holm_0.025": rej_holm / n_conv if n_conv else None,
                "mean_estimated_interaction_coef": mean_coef,
                "wall_time_s": round(elapsed, 1),
            }
        )
        print(
            f"  [{cell.domain}/{cell.signal}] beta_int={beta_int:+.3f}: "
            f"power(.05)={results[-1]['power_alpha_nominal_0.05']}, "
            f"power(.025)={results[-1]['power_alpha_holm_0.025']}, "
            f"conv={n_conv}/{n_reps}, {elapsed:.0f}s",
            flush=True,
        )
    return results


def _interpolate_crossing(
    grid_results: list[dict[str, Any]], power_key: str, target: float = 0.80
) -> float | None:
    """Linear-interpolate the |beta_int| at which power first reaches ``target``, walking the
    grid from 0 outward (grid is expected sorted by |beta_int| ascending, beta_int <= 0)."""
    pts = sorted(grid_results, key=lambda r: abs(r["beta_int_true"]))
    for a, b in zip(pts, pts[1:]):
        pa, pb = a[power_key], b[power_key]
        if pa is None or pb is None:
            continue
        if pa < target <= pb:
            xa, xb = abs(a["beta_int_true"]), abs(b["beta_int_true"])
            if pb == pa:
                return xa
            frac = (target - pa) / (pb - pa)
            return xa + frac * (xb - xa)
    if pts and pts[-1][power_key] is not None and pts[-1][power_key] < target:
        return None  # not reached within the grid
    if pts and pts[0][power_key] is not None and pts[0][power_key] >= target:
        return abs(pts[0]["beta_int_true"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", default="data/results/instrument_validation/phase1_20260714_105004"
    )
    parser.add_argument("--out", default="data/results/gate_e_h3_power/h3_power_simulation.json")
    parser.add_argument(
        "--n-reps", type=int, default=200, help="Replicates per (domain, signal, effect-size) cell."
    )
    parser.add_argument(
        "--n-reps-secondary", type=int, default=150, help="Replicates for the VC secondary grid."
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--effect-grid",
        default="0,-0.05,-0.1,-0.15,-0.2,-0.3,-0.4,-0.5,-0.75,-1.0",
        help="Comma-separated true interaction coefficients (logit units, standardized signal x position_norm).",
    )
    parser.add_argument(
        "--effect-grid-secondary",
        default="0,-0.15,-0.3,-0.5,-1.0",
        help="Reduced grid used for the VC secondary run.",
    )
    parser.add_argument(
        "--toh-length-run-dir",
        default=None,
        help="Directory of real per-episode ToH result JSONs to bootstrap episode lengths from "
        "(e.g. data/results/gate_d_calibration/toh_corridor_scramble_n30), filtered to "
        "--toh-length-num-disks and C1. Omit to fall back to the raw pre-calibration pilot "
        "episode lengths (original v1/v2 behavior).",
    )
    parser.add_argument("--toh-length-num-disks", type=int, default=4)
    args = parser.parse_args()

    run_dir = (
        REPO_ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    )
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pilot data from {run_dir} ...")
    pilot = compute_pilot_seed_stats(run_dir)
    for dom, s in pilot.items():
        print(
            f"  {dom}: n_steps={s.n_steps} n_clusters={s.n_clusters} base_rate={s.base_rate:.3f} "
            f"icc_gee={s.icc_gee} icc_anova={s.icc_anova} tle_mean/sd={s.tle_mean:.4f}/{s.tle_sd:.4f} "
            f"vc_mean/sd={s.vc_mean:.2f}/{s.vc_sd:.2f}"
        )

    toh_frozen_lengths: list[int] | None = None
    if args.toh_length_run_dir:
        toh_run_dir = (
            REPO_ROOT / args.toh_length_run_dir
            if not Path(args.toh_length_run_dir).is_absolute()
            else Path(args.toh_length_run_dir)
        )
        toh_frozen_lengths = load_toh_frozen_corridor_lengths(
            toh_run_dir, num_disks=args.toh_length_num_disks
        )
        print(
            f"Loaded {len(toh_frozen_lengths)} real ToH C1/{args.toh_length_num_disks}-disk "
            f"episode lengths from {toh_run_dir} for the length bootstrap pool."
        )

    cells = build_design_cells(
        pilot,
        instances_per_domain=50,
        runs_per_condition=5,
        compute_stages=3,
        toh_frozen_corridor_lengths=toh_frozen_lengths,
    )

    effect_grid = sorted({float(x) for x in args.effect_grid.split(",")}, reverse=True)
    effect_grid_secondary = sorted(
        {float(x) for x in args.effect_grid_secondary.split(",")}, reverse=True
    )

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_run_dir": str(run_dir),
        "planned_design": {
            "instances_per_domain": 50,
            "runs_per_condition": 5,
            "compute_stages": 3,
            "episodes_per_instance": 15,
        },
        "alpha_one_sided_nominal": ALPHA_NOMINAL,
        "alpha_one_sided_holm_family_of_2": ALPHA_HOLM2,
        "pilot_stats": {dom: asdict(s) for dom, s in pilot.items()},
        "design_cells": {},
        "power_grids": {},
        "power_80_crossing": {},
    }

    print("\n=== Primary: TLE, full grid, both domains ===")
    for dom in DOMAINS:
        cell = cells[(dom, "tle")]
        report["design_cells"][f"{dom}/tle"] = asdict(cell)
        grid_res = run_power_grid(cell, effect_grid, args.n_reps, args.workers, args.seed)
        report["power_grids"][f"{dom}/tle"] = grid_res
        report["power_80_crossing"][f"{dom}/tle"] = {
            "beta_int_at_80pct_power_alpha_0.05": _interpolate_crossing(
                grid_res, "power_alpha_nominal_0.05"
            ),
            "beta_int_at_80pct_power_alpha_0.025": _interpolate_crossing(
                grid_res, "power_alpha_holm_0.025"
            ),
        }

    print("\n=== Secondary: VC, reduced grid, both domains ===")
    for dom in DOMAINS:
        cell = cells[(dom, "vc")]
        report["design_cells"][f"{dom}/vc"] = asdict(cell)
        grid_res = run_power_grid(
            cell, effect_grid_secondary, args.n_reps_secondary, args.workers, args.seed + 1
        )
        report["power_grids"][f"{dom}/vc"] = grid_res
        report["power_80_crossing"][f"{dom}/vc"] = {
            "beta_int_at_80pct_power_alpha_0.05": _interpolate_crossing(
                grid_res, "power_alpha_nominal_0.05"
            ),
            "beta_int_at_80pct_power_alpha_0.025": _interpolate_crossing(
                grid_res, "power_alpha_holm_0.025"
            ),
        }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
