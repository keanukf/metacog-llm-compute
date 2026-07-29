"""Full per-variable descriptive statistics ("codebook") for the Phase 1 preanalysis screen.

Separated from ``preanalysis_screen.py`` deliberately: that module answers "is something broken"
(variance/missingness gates, empty-cell warnings); this one answers "what do my variables actually
look like" -- the psych::describe()/naniar::miss_var_summary()-style report a reader would expect
before trusting any downstream hypothesis test. Uses numpy (already a hard project dependency, see
requirements.txt and ``src/analysis/icc.py``), not a from-scratch reimplementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Variables z-standardized per compute_stage downstream (ADR-006: fit_h3_model, fit_tle_calibrator)
# -- described per stage in addition to pooled, so a reader can see whether that standardization
# assumption (non-degenerate variance within each stage) actually holds.
STAGE_CONDITIONAL_STEP_VARS = ("tle_mean_entropy", "vc")
POOLED_STEP_VARS = ("tle_mean_entropy", "vc", "position_norm")
EPISODE_VARS = ("episode_length_steps", "normalized_compute_cost")

# Measurement-scale documentation (qualitative, not computed from data) -- role/scale/note per
# variable actually collected. Not just bookkeeping: VC's scale assumption in particular is load-
# bearing for whether Pearson correlation/OLS-style reasoning about it is even appropriate (see
# the empirical check in compute_signal_correlation below, which is why VC's note points there).
MEASUREMENT_SCALES: list[dict[str, str]] = [
    {
        "variable": "domain",
        "role": "Grouping factor",
        "scale": "Nominal (2 levels)",
        "note": "tower_of_hanoi, textworld -- not ordered.",
    },
    {
        "variable": "compute_stage",
        "role": "Grouping factor / manipulated condition",
        "scale": "Ordinal (3 levels)",
        "note": "C0 < C1 < C2 by increasing compute investment, but analyzed as an unordered "
        "categorical factor (separate stage-wise z-standardization, ADR-006), not a numeric "
        "predictor -- the ordering is conceptual, not used as a linear scale in any model.",
    },
    {
        "variable": "instance_key",
        "role": "Clustering unit",
        "scale": "Nominal (50 levels per domain)",
        "note": "The resampling/clustering unit for cluster_bootstrap and the GEE grouping "
        "variable -- not a variable of substantive interest itself.",
    },
    {
        "variable": "holdout",
        "role": "Design flag",
        "scale": "Nominal (binary)",
        "note": "5 of 50 instances per domain, frozen after Gate D; used for calibrator/threshold "
        "fitting, never for confirmatory hypothesis tests on the same steps.",
    },
    {
        "variable": "tle_mean_entropy (TLE)",
        "role": "Predictor signal",
        "scale": "Ratio",
        "note": "Shannon entropy over top-K renormalized logprobs; true zero = complete "
        "certainty. Right-skewed, often near-floor (see Table 4) -- analyses use it "
        "stage-wise z-standardized, never the raw scale directly.",
    },
    {
        "variable": "vc (VC)",
        "role": "Predictor signal",
        "scale": "Interval, by convention -- with a caveat",
        "note": "Self-reported 0-100 confidence, but only 16 distinct values occur in the real "
        "data (multiples of 5/10), not a continuous 0-100 scale. Conventionally analyzed as "
        "interval/ratio (as the thesis and this pipeline both do), but strictly it is a coarse, "
        "self-reported ordinal judgment -- Pearson correlations/OLS-style linear assumptions on "
        "it should be read with that caveat in mind (see the Spearman comparison in Table 7).",
    },
    {
        "variable": "position_norm",
        "role": "Covariate (H3 temporal-degradation axis)",
        "scale": "Ratio / interval, deterministic",
        "note": "t / max(episode_length - 1, 1) -- computed exactly from step index and episode "
        "length, not measured with error. Its near-uniform distribution (Table 3) is guaranteed "
        "by construction, not an empirical finding.",
    },
    {
        "variable": "y_optimal / task_success",
        "role": "Outcome (dependent variable)",
        "scale": "Nominal (binary, Bernoulli)",
        "note": "Step-level (y_optimal) and episode-level (task_success) correctness. Mean = "
        "proportion correct; skewness/quantiles are not meaningful for a binary variable and are "
        "not reported (see Table 5 instead of Tables 3-4's continuous-variable format).",
    },
    {
        "variable": "episode_length_steps",
        "role": "Episode-level descriptive / covariate",
        "scale": "Ratio (discrete count)",
        "note": "Right-censored at the frozen difficulty-manifest step ceiling (max 45) -- values "
        "at the maximum reflect the cap, not necessarily the episode's true difficulty.",
    },
    {
        "variable": "normalized_compute_cost",
        "role": "Episode-level descriptive",
        "scale": "Ratio, bounded [0, 1] by construction",
        "note": "Normalized against the maximum possible compute for that condition.",
    },
    {
        "variable": "difficulty_tier",
        "role": "Instance-level covariate",
        "scale": "Ordinal (observed levels: easy, medium)",
        "note": "Constant ('medium') for every tower_of_hanoi episode -- ToH's frozen manifest "
        "(4 disks, one scramble seed) gives no per-instance difficulty variation. Genuinely "
        "heterogeneous within textworld (Table 8) -- an asymmetry between domains worth keeping "
        "in mind whenever comparing them (H4), since only textworld's instance pool has this "
        "extra source of variance.",
    },
]


def describe_values(values: list[float]) -> dict[str, Any]:
    """Numeric summary of already-filtered (non-missing) values. Missingness is the caller's
    responsibility (``describe_variable``) since "missing" is only meaningful relative to a known
    total row count, which this function doesn't see."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "skewness": None,
            "n_outliers_iqr": None,
            "outlier_rate_iqr": None,
        }
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else None
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    n_outliers = int(np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": sd,
        "min": float(arr.min()),
        "q1": q1,
        "median": float(np.percentile(arr, 50)),
        "q3": q3,
        "max": float(arr.max()),
        "skewness": _skewness(arr) if arr.size >= 3 else None,
        # Standard Tukey 1.5xIQR fence -- a plain count/rate to go with the skewness figure,
        # relevant here specifically because TLE's real skewness (2-38 across domain/stage,
        # Table 4) is extreme enough that "how many points, exactly" is worth stating rather
        # than leaving a reader to infer it from the boxplot figure alone.
        "n_outliers_iqr": n_outliers,
        "outlier_rate_iqr": n_outliers / arr.size,
    }


def _skewness(arr: np.ndarray) -> float | None:
    sd = float(arr.std(ddof=0))
    if sd == 0:
        return None
    return float(np.mean(((arr - arr.mean()) / sd) ** 3))


def describe_variable(rows: list[dict[str, Any]], key: str, *, total: int | None = None) -> dict[str, Any]:
    """``describe_values`` plus missingness, relative to ``total`` rows (defaults to ``len(rows)``
    -- pass an explicit ``total`` when describing a variable within an already-filtered subset,
    e.g. per compute_stage, so the missing rate is relative to that subset, not the whole table)."""
    total = total if total is not None else len(rows)
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    n_missing = total - len(values)
    desc = describe_values(values)
    desc["n_missing"] = n_missing
    desc["missing_rate"] = (n_missing / total) if total else None
    return desc


def compute_variable_codebook(
    steps: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Full descriptive codebook, by domain: pooled step-level vars, step-level vars broken out
    per compute_stage (C0/C1/C2 -- the granularity the stage-conditional z-standardization
    actually operates at), and episode-level vars."""
    by_dom_steps: dict[str, list[dict[str, Any]]] = {}
    for r in steps:
        by_dom_steps.setdefault(str(r.get("domain", "unknown")), []).append(r)
    by_dom_eps: dict[str, list[dict[str, Any]]] = {}
    for e in episodes:
        by_dom_eps.setdefault(str(e.get("domain", "unknown")), []).append(e)

    codebook: dict[str, Any] = {"step_level": {}, "step_level_by_stage": {}, "episode_level": {}}

    for dom in sorted(by_dom_steps):
        rows = by_dom_steps[dom]
        codebook["step_level"][dom] = {
            var: describe_variable(rows, var) for var in POOLED_STEP_VARS
        }

        by_stage: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            stage = r.get("compute_stage")
            if stage:
                by_stage.setdefault(str(stage), []).append(r)
        codebook["step_level_by_stage"][dom] = {
            stage: {var: describe_variable(stage_rows, var) for var in STAGE_CONDITIONAL_STEP_VARS}
            for stage, stage_rows in sorted(by_stage.items())
        }

        codebook["step_level"][dom]["y_optimal"] = _describe_binary(rows, "y_optimal")
        codebook["step_level_by_stage"][dom] = {
            stage: {
                **codebook["step_level_by_stage"][dom][stage],
                "y_optimal": _describe_binary(stage_rows, "y_optimal"),
            }
            for stage, stage_rows in sorted(by_stage.items())
        }
        codebook["signal_correlation"] = codebook.get("signal_correlation", {})
        codebook["signal_correlation"][dom] = compute_signal_correlation(rows)

    for dom in sorted(by_dom_eps):
        eps = by_dom_eps[dom]
        success = [1.0 if bool(e.get("task_success")) else 0.0 for e in eps]
        codebook["episode_level"][dom] = {
            var: describe_variable(eps, var) for var in EPISODE_VARS
        }
        codebook["episode_level"][dom]["task_success_rate"] = (
            (sum(success) / len(success)) if success else None
        )
        codebook["episode_level"][dom]["n_episodes"] = len(eps)

    codebook["sample_composition"] = compute_sample_composition(episodes)
    codebook["difficulty_tier_breakdown"] = compute_difficulty_tier_breakdown(episodes)
    return codebook


def _describe_binary(rows: list[dict[str, Any]], key: str, *, total: int | None = None) -> dict[str, Any]:
    """Bernoulli summary (rate, not mean/sd/quantiles/skew -- those aren't meaningful for a
    0/1 outcome). ``sd`` is still reported since it's fully determined by the rate
    (sqrt(p(1-p))) and downstream readers may want it for a quick power/precision sense-check."""
    total = total if total is not None else len(rows)
    values = [int(r[key]) for r in rows if r.get(key) is not None]
    n_missing = total - len(values)
    rate = (sum(values) / len(values)) if values else None
    sd = (rate * (1 - rate)) ** 0.5 if rate is not None else None
    return {
        "n": len(values),
        "n_missing": n_missing,
        "missing_rate": (n_missing / total) if total else None,
        "rate": rate,
        "sd": sd,
    }


def compute_signal_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TLE-VC association (convergent validity of the two signals): Pearson r AND Spearman rho,
    since VC's coarse 16-value self-report scale (see MEASUREMENT_SCALES) makes the linear-
    correlation assumption behind Pearson r questionable -- reporting both lets a reader see
    whether they actually diverge (they do: e.g. r=-0.12 vs rho=-0.34 on the real textworld data,
    meaning the relationship is real but not well captured by a straight line)."""
    paired = [
        (float(r["tle_mean_entropy"]), float(r["vc"]))
        for r in rows
        if r.get("tle_mean_entropy") is not None and r.get("vc") is not None
    ]
    if len(paired) < 3:
        return {"n": len(paired), "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
    tle = np.array([p[0] for p in paired], dtype=float)
    vc = np.array([p[1] for p in paired], dtype=float)
    from scipy.stats import pearsonr, spearmanr

    r, p_r = pearsonr(tle, vc)
    rho, p_rho = spearmanr(tle, vc)
    return {
        "n": len(paired),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


def compute_sample_composition(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episode-level design cell counts (domain x compute_stage x holdout) -- makes the
    preregistered 2x3x(5/45 holdout split) design tangible as numbers, not just prose."""
    counts: dict[tuple[str, str, bool], int] = {}
    for e in episodes:
        key = (str(e.get("domain", "unknown")), str(e.get("compute_stage", "unknown")), bool(e.get("holdout")))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"domain": dom, "compute_stage": stage, "holdout": holdout, "n_episodes": n}
        for (dom, stage, holdout), n in sorted(counts.items())
    ]


def compute_difficulty_tier_breakdown(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Instance-level difficulty covariate (``difficulty_tier``), by domain: n episodes and task
    success rate per tier. Surfaces a real domain asymmetry -- tower_of_hanoi's frozen manifest
    gives every episode the same tier ('medium'), while textworld's 50 procedurally generated
    instances land in two tiers ('easy'/'medium') with real prevalence -- so this covariate is a
    source of variance in one domain but not the other, worth checking against outcome rather
    than assuming it's inert."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in episodes:
        key = (str(e.get("domain", "unknown")), str(e.get("difficulty_tier", "unknown")))
        by_key.setdefault(key, []).append(e)
    rows = []
    for (dom, tier), eps in sorted(by_key.items()):
        success = [1.0 if bool(e.get("task_success")) else 0.0 for e in eps]
        rows.append(
            {
                "domain": dom,
                "difficulty_tier": tier,
                "n_episodes": len(eps),
                "task_success_rate": (sum(success) / len(success)) if success else None,
            }
        )
    return rows


_VAR_LABELS = {
    "tle_mean_entropy": "TLE",
    "vc": "VC",
    "position_norm": "position_norm",
    "episode_length_steps": "Episode length (steps)",
    "normalized_compute_cost": "Normalized compute cost",
}


def _fmt(x: float | None, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "--"


# TLE lives on a compressed ~0-1.3 entropy scale where most mass sits near 0 (see the C1/C2
# rows of Table 2 in practice) -- 2 decimals rounds almost every cell to "0.00" and destroys the
# exact information the table exists to convey. VC (0-100) and position_norm (0-1, but spread
# across the whole range) don't have this problem at 2 decimals.
_VAR_DECIMALS = {"tle_mean_entropy": 4}


def _nd_for(var: str) -> int:
    return _VAR_DECIMALS.get(var, 2)


def render_apa_codebook_markdown(codebook: dict[str, Any], *, table_number_start: int = 1) -> str:
    """Render the codebook as APA-7-styled Markdown tables (title above, note below, no vertical
    rules -- Markdown tables are horizontal-rule-only by construction, which already matches
    APA's own convention)."""
    n = table_number_start
    lines: list[str] = []
    domains = sorted(codebook.get("step_level", {}).keys())

    # Table: measurement scales and roles (qualitative, orients the reader before any numbers).
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Variable roles and measurement scales*")
    lines.append("")
    lines.append("| Variable | Role | Measurement scale | Note |")
    lines.append("|---|---|---|---|")
    for row in MEASUREMENT_SCALES:
        lines.append(f"| {row['variable']} | {row['role']} | {row['scale']} | {row['note']} |")
    lines.append("")
    n += 1

    # Table: sample composition (design cell counts).
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Sample composition: episodes by domain, compute stage, and holdout status*")
    lines.append("")
    lines.append("| Domain | Stage | Holdout | N episodes |")
    lines.append("|---|---|---|---:|")
    for row in codebook.get("sample_composition", []):
        lines.append(
            f"| {row['domain']} | {row['compute_stage']} | {row['holdout']} | {row['n_episodes']} |"
        )
    lines.append("")
    lines.append(
        "*Note.* Holdout = True are the 5-of-50 instances per domain frozen after Gate D for "
        "calibrator/threshold fitting; Holdout = False (45 instances x 5 runs = 225) are used "
        "for the confirmatory hypothesis tests."
    )
    lines.append("")
    n += 1

    # Table: pooled continuous step-level signals (TLE, VC, position_norm).
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Descriptive statistics for step-level signals, by domain*")
    lines.append("")
    lines.append(
        "| Domain | Variable | N | N missing | M | SD | Min | Mdn | Max | Skewness | "
        "Outliers (1.5xIQR) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dom in domains:
        for var in POOLED_STEP_VARS:
            d = codebook["step_level"][dom][var]
            nd = _nd_for(var)
            outlier_cell = (
                f"{d['n_outliers_iqr']} ({_fmt(d['outlier_rate_iqr'], 3)})"
                if d.get("n_outliers_iqr") is not None
                else "--"
            )
            lines.append(
                f"| {dom} | {_VAR_LABELS.get(var, var)} | {d['n']} | {d['n_missing']} | "
                f"{_fmt(d['mean'], nd)} | {_fmt(d['sd'], nd)} | {_fmt(d['min'], nd)} | "
                f"{_fmt(d['median'], nd)} | {_fmt(d['max'], nd)} | {_fmt(d['skewness'])} | "
                f"{outlier_cell} |"
            )
    lines.append("")
    lines.append(
        "*Note.* N missing = steps where the variable could not be extracted (e.g. no closed "
        "`</think>` block for TLE). TLE = token-level entropy; VC = verbalized confidence "
        "(0-100 scale, raw, not yet mapped to a stage-conditional z-score). Outliers = points "
        "beyond Q1-1.5xIQR or Q3+1.5xIQR (Tukey fence), count (rate). See Table 1 for "
        "measurement-scale caveats."
    )
    lines.append("")
    n += 1

    # Table: TLE/VC by domain and compute stage (the ADR-006 z-standardization granularity).
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Descriptive statistics for TLE and VC, by domain and compute stage*")
    lines.append("")
    lines.append("| Domain | Stage | Variable | N | M | SD | Skewness |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for dom in domains:
        stages = sorted(codebook.get("step_level_by_stage", {}).get(dom, {}).keys())
        for stage in stages:
            for var in STAGE_CONDITIONAL_STEP_VARS:
                d = codebook["step_level_by_stage"][dom][stage][var]
                nd = _nd_for(var)
                lines.append(
                    f"| {dom} | {stage} | {_VAR_LABELS.get(var, var)} | {d['n']} | "
                    f"{_fmt(d['mean'], nd)} | {_fmt(d['sd'], nd)} | {_fmt(d['skewness'])} |"
                )
    lines.append("")
    lines.append(
        "*Note.* Reported per compute stage because the confirmatory H3/H1b models "
        "z-standardize TLE/VC within each compute stage separately (ADR-006), not pooled -- a "
        "near-zero SD within any single stage row here would invalidate that standardization."
    )
    lines.append("")
    n += 1

    # Table: binary outcome (y_optimal), pooled and by stage -- separate from Tables 3-4 since
    # mean/quantiles/skewness aren't meaningful summaries for a 0/1 variable.
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Step-level correctness rate (y_optimal), by domain and compute stage*")
    lines.append("")
    lines.append("| Domain | Stage | N | Rate correct | SD |")
    lines.append("|---|---|---:|---:|---:|")
    for dom in domains:
        pooled = codebook["step_level"][dom]["y_optimal"]
        lines.append(f"| {dom} | pooled | {pooled['n']} | {_fmt(pooled['rate'], 3)} | {_fmt(pooled['sd'], 3)} |")
        stages = sorted(codebook.get("step_level_by_stage", {}).get(dom, {}).keys())
        for stage in stages:
            d = codebook["step_level_by_stage"][dom][stage]["y_optimal"]
            lines.append(f"| {dom} | {stage} | {d['n']} | {_fmt(d['rate'], 3)} | {_fmt(d['sd'], 3)} |")
    lines.append("")
    lines.append(
        "*Note.* Rate = proportion of steps with the optimal action; SD = sqrt(rate x (1-rate)), "
        "the Bernoulli standard deviation implied by the rate, not an independently estimated "
        "quantity."
    )
    lines.append("")
    n += 1

    # Table: episode-level variables.
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Descriptive statistics for episode-level variables, by domain*")
    lines.append("")
    lines.append("| Domain | N episodes | Task success rate | Episode length M (SD) | "
                  "Compute cost M (SD) |")
    lines.append("|---|---:|---:|---:|---:|")
    for dom in sorted(codebook.get("episode_level", {}).keys()):
        d = codebook["episode_level"][dom]
        el = d["episode_length_steps"]
        cc = d["normalized_compute_cost"]
        lines.append(
            f"| {dom} | {d['n_episodes']} | {_fmt(d['task_success_rate'], 3)} | "
            f"{_fmt(el['mean'])} ({_fmt(el['sd'])}) | {_fmt(cc['mean'], 3)} ({_fmt(cc['sd'], 3)}) |"
        )
    lines.append("")
    lines.append(
        "*Note.* M = mean, SD = standard deviation. Task success rate is the raw episode-level "
        "success proportion, not the calibrated proxy objective used in policy threshold search."
    )
    lines.append("")
    n += 1

    # Table: TLE-VC association (convergent validity of the two signals).
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*TLE-VC association, by domain*")
    lines.append("")
    lines.append("| Domain | N | Pearson r | p | Spearman rho | p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dom in domains:
        c = codebook.get("signal_correlation", {}).get(dom, {})
        lines.append(
            f"| {dom} | {c.get('n')} | {_fmt(c.get('pearson_r'), 3)} | {_fmt(c.get('pearson_p'), 4)} | "
            f"{_fmt(c.get('spearman_rho'), 3)} | {_fmt(c.get('spearman_p'), 4)} |"
        )
    lines.append("")
    lines.append(
        "*Note.* Both reported because VC's coarse self-report scale (Table 1) makes Pearson's "
        "linearity assumption questionable; a Pearson/Spearman gap indicates the association is "
        "real but non-linear, not that one estimate is simply wrong. Negative sign is expected: "
        "higher TLE (entropy) should co-occur with lower VC (confidence)."
    )
    lines.append("")
    n += 1

    # Table: instance-level difficulty covariate, by domain -- surfaces the domain asymmetry
    # (ToH constant, textworld heterogeneous) rather than leaving difficulty_tier undocumented.
    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Instance difficulty tier, by domain*")
    lines.append("")
    lines.append("| Domain | Difficulty tier | N episodes | Task success rate |")
    lines.append("|---|---|---:|---:|")
    for row in codebook.get("difficulty_tier_breakdown", []):
        lines.append(
            f"| {row['domain']} | {row['difficulty_tier']} | {row['n_episodes']} | "
            f"{_fmt(row['task_success_rate'], 3)} |"
        )
    lines.append("")
    lines.append(
        "*Note.* tower_of_hanoi's frozen manifest (4 disks, single scramble seed) gives every "
        "episode the same tier -- no per-instance difficulty variance. textworld's 50 "
        "procedurally generated instances land in two tiers with real prevalence; this is a "
        "source of variance present in one domain but not the other, relevant whenever comparing "
        "domains directly (H4)."
    )

    return "\n".join(lines) + "\n"
