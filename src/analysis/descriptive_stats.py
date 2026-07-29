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
        }
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": sd,
        "min": float(arr.min()),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "skewness": _skewness(arr) if arr.size >= 3 else None,
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

    return codebook


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
    APA's own convention). One table for pooled step-level variables, one for episode-level
    variables, one per domain for the stage-conditional breakdown."""
    n = table_number_start
    lines: list[str] = []

    domains = sorted(codebook.get("step_level", {}).keys())

    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Descriptive statistics for step-level signals, by domain*")
    lines.append("")
    lines.append("| Domain | Variable | N | N missing | M | SD | Min | Mdn | Max | Skewness |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dom in domains:
        for var, d in codebook["step_level"][dom].items():
            nd = _nd_for(var)
            lines.append(
                f"| {dom} | {_VAR_LABELS.get(var, var)} | {d['n']} | {d['n_missing']} | "
                f"{_fmt(d['mean'], nd)} | {_fmt(d['sd'], nd)} | {_fmt(d['min'], nd)} | "
                f"{_fmt(d['median'], nd)} | {_fmt(d['max'], nd)} | {_fmt(d['skewness'])} |"
            )
    lines.append("")
    lines.append(
        "*Note.* N missing = steps where the variable could not be extracted (e.g. no closed "
        "`</think>` block for TLE). TLE = token-level entropy; VC = verbalized confidence "
        "(0-100 scale, raw, not yet mapped to a stage-conditional z-score)."
    )
    lines.append("")
    n += 1

    lines.append(f"*Table {n}*")
    lines.append("")
    lines.append("*Descriptive statistics for TLE and VC, by domain and compute stage*")
    lines.append("")
    lines.append("| Domain | Stage | Variable | N | M | SD | Skewness |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for dom in domains:
        stages = sorted(codebook.get("step_level_by_stage", {}).get(dom, {}).keys())
        for stage in stages:
            for var, d in codebook["step_level_by_stage"][dom][stage].items():
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

    return "\n".join(lines) + "\n"
