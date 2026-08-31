"""
Plots and tables for calibration and Phase 2 results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def reliability_diagram(
    predictions: list[float],
    correctness: list[float],
    save_path: str | None = None,
    *,
    n_bins: int = 10,
    title: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot a reliability diagram (mean confidence vs mean accuracy per bin).

    Requires matplotlib.
    """
    from src.analysis.calibration import reliability_diagram_data

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # Visualization is optional; allow analysis pipelines to run without matplotlib.
        return

    xs, mean_conf, mean_acc = reliability_diagram_data(predictions, correctness, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="perfect")
    ax.plot(mean_conf, mean_acc, marker="o", linewidth=2.0, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_phase2_results(
    results_path: str,
    output_dir: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Backward-compatible wrapper: generate Phase 2 plots from a checkpoint directory.

    Prefer using `plot_run_overview(...)` from analysis scripts; this function stays minimal.
    """
    run_dir = Path(results_path)
    out_dir = Path(output_dir) if output_dir is not None else (run_dir / "analysis" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.analysis.datasets import load_run_dataset

    ds = load_run_dataset(run_dir)
    plot_run_overview(ds.episodes, ds.steps, out_dir)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def plot_run_overview(
    episodes_rows: Iterable[dict[str, Any]],
    steps_rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Generate a small set of high-signal static figures.

    Returns a mapping of figure name -> path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # 1) Success vs normalized compute cost by strategy/stage
    written.update(_plot_success_vs_cost(episodes_rows, output_dir / "success_vs_cost.png"))

    # 2) Stage mix
    written.update(_plot_stage_mix(steps_rows, output_dir / "stage_mix_overall.png"))

    # 3) Reliability diagram from VC where step labels exist
    preds: list[float] = []
    labs: list[float] = []
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        vc = _safe_float(r.get("vc"))
        y = r.get("step_correct_optimal")
        if vc is None or y is None:
            continue
        preds.append(max(0.0, min(1.0, vc / 100.0)))
        labs.append(float(y))
    if preds:
        p = output_dir / "reliability_vc_optimal_only.png"
        reliability_diagram(preds, labs, save_path=str(p), title="Reliability (VC, optimal-only)")
        written["reliability_vc_optimal_only"] = str(p)

    return written


def _plot_success_vs_cost(
    episodes_rows: Iterable[dict[str, Any]],
    save_path: Path,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    # Group key: strategy if any, else compute_stage.
    rows = [ep for ep in episodes_rows if isinstance(ep, dict)]
    group_key = "strategy" if any("strategy" in ep for ep in rows) else "compute_stage"
    groups: dict[str, list[dict[str, Any]]] = {}
    for ep in rows:
        g = str(ep.get(group_key, "unknown"))
        groups.setdefault(g, []).append(ep)

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for g, eps in sorted(groups.items()):
        n = len(eps)
        if n == 0:
            continue
        succ = sum(1 for e in eps if bool(e.get("task_success")))
        success_rate = succ / n
        costs = [_safe_float(e.get("normalized_compute_cost")) for e in eps]
        costs_clean = [c for c in costs if c is not None]
        mean_cost = (sum(costs_clean) / len(costs_clean)) if costs_clean else 0.0
        xs.append(mean_cost)
        ys.append(success_rate)
        labels.append(g)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=9)
    ax.set_xlabel("Mean normalized compute cost")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return {"success_vs_cost": str(save_path)}


def _plot_stage_mix(
    steps_rows: Iterable[dict[str, Any]],
    save_path: Path,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    counts: dict[str, int] = {}
    total = 0
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        st = r.get("compute_stage")
        if st is None:
            continue
        st = str(st)
        counts[st] = counts.get(st, 0) + 1
        total += 1
    keys = sorted(counts.keys())
    vals = [counts[k] / total if total else 0.0 for k in keys]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(keys, vals)
    ax.set_ylabel("Fraction of steps")
    ax.set_ylim(0, 1)
    ax.set_title("Stage mix (overall)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return {"stage_mix_overall": str(save_path)}


def plot_auroc_comparison_bars(
    h1a_results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Phase 1 analysis Stage 6: grouped TLE-vs-VC AUROC bars per domain, from a Stage 2
    (``stage2_h1a_discrimination.py``) result dict. Bar heights come from the independent
    descriptive path (``descriptive_cross_check[domain]["optimal_only"]``, point-estimate AUROC,
    no bootstrap CI of its own); the confirmatory ΔAUROC(TLE,VC) 90% CI and Holm-adjusted decision
    are annotated per domain since that -- not either raw AUROC alone -- is what H1a decides on.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "h1a_auroc_comparison.png"

    domains = sorted(h1a_results.get("by_domain", {}).keys())
    if not domains:
        return {}
    tle_vals = []
    vc_vals = []
    annotations = []
    for dom in domains:
        desc = h1a_results["descriptive_cross_check"].get(dom, {}).get("optimal_only", {})
        tle_vals.append(_safe_float(desc.get("tle", {}).get("auroc")) or 0.0)
        vc_vals.append(_safe_float(desc.get("vc", {}).get("auroc")) or 0.0)
        conf = h1a_results["by_domain"][dom]
        holds = conf.get("decision_holds")
        annotations.append(
            f"ΔAUROC={conf['point']:.3f}\nCI=[{conf['ci_low']:.3f},{conf['ci_high']:.3f}]\nholds={holds}"
            if conf.get("point") is not None
            else "n/a"
        )

    x = range(len(domains))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.bar([i - width / 2 for i in x], tle_vals, width, label="TLE")
    ax.bar([i + width / 2 for i in x], vc_vals, width, label="VC")
    for i, ann in enumerate(annotations):
        ax.annotate(
            ann,
            (i, max(tle_vals[i], vc_vals[i]) + 0.02),
            ha="center",
            fontsize=7,
            va="bottom",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(domains)
    ax.set_ylabel("AUROC (optimal-only, descriptive)")
    ax.set_ylim(0, 1.15)
    ax.set_title("H1a: TLE vs VC discrimination per domain")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return {"h1a_auroc_comparison": str(save_path)}


def _empirical_position_binned_means(
    steps: list[dict[str, Any]], *, signal: str, domain: str, n_bins: int = 8
) -> dict[str, list[Any]] | None:
    """Empirical (binned-average) counterpart to the fitted H3 curve: real early-half/late-half
    steps, z_c-binned via the exact same stage-wise standardization ``fit_h3_model`` uses
    (``build_h3_frame``, so this can never silently drift out of sync with the fit), with mean
    observed y per bin -- what a reader needs to visually judge model fit (linearity-in-logit),
    not just see the fitted curve on its own. Returns None if there isn't enough data for either
    half to bin meaningfully."""
    import pandas as pd

    from src.analysis.inference import build_h3_frame

    frame, _note = build_h3_frame(steps, signal=signal, domain=domain)
    if frame is None or len(frame) < 2 * n_bins:
        return None

    out: dict[str, list[Any]] = {}
    for label, mask in (
        ("early", frame["position_norm"] < 0.5),
        ("late", frame["position_norm"] >= 0.5),
    ):
        sub = frame[mask]
        if len(sub) < n_bins:
            continue
        sub = sub.assign(_bin=pd.qcut(sub["z_c"], q=n_bins, duplicates="drop"))
        grouped = sub.groupby("_bin", observed=True).agg(
            z_mean=("z_c", "mean"), y_mean=("y", "mean"), n=("y", "size")
        )
        out[label] = [
            {"z": float(row.z_mean), "y": float(row.y_mean), "n": int(row.n)}
            for row in grouped.itertuples()
        ]
    return out or None


def plot_h3_marginal_effect(
    h3_results: dict[str, Any],
    output_dir: str | Path,
    *,
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """
    Phase 1 analysis Stage 6: signal x position marginal-effect curves from a Stage 4
    (``stage4_h3_temporal.py``) result dict -- the temporal-degradation plot the thesis
    methodology (Ch.5 §5.8) describes: predicted P(correct) vs. the stage-wise z-standardized
    signal at an early (position_norm=0.1) and late (position_norm=0.9) episode position. Model
    coefficients (const, z_c, p_c, interaction) already fully specify the fitted curve -- no need
    to re-touch the raw per-step table for that part.

    Pass ``steps`` (the canonical dataset's step rows) to additionally overlay empirical, binned
    real data (z_c deciles within the early/late position halves, mean observed y per bin) as
    scatter points alongside the fitted lines -- without this, the plot only shows what the model
    *implies*, with no way for a reader to visually judge whether the linear-in-logit interaction
    model actually fits the real data shape. Omit ``steps`` to get the fitted-curve-only plot (e.g.
    from a context where the raw data isn't loaded).

    NOTE on the TLE sign flip: ``fit_h3_model`` builds its ``z`` column as ``-tle_mean_entropy``
    for the tle signal (higher z = lower entropy = more certain), not raw entropy, so the x-axis
    for tle plots is labeled "-TLE (certainty)" -- labeling it as plain "TLE" would make the curve
    read backwards to anyone comparing it against the raw entropy definition in the thesis text.
    """
    try:
        import math as _math

        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _sigmoid(v: float) -> float:
        return 1.0 / (1.0 + _math.exp(-v))

    z_grid = [i / 20.0 for i in range(-40, 41)]  # -2.0 .. 2.0
    for dom, sig_results in h3_results.get("results", {}).items():
        for sig, fit in sig_results.items():
            if not fit.get("converged"):
                continue
            params = fit["params"]
            const, z_c, p_c, interaction = (
                params["const"],
                params["z_c"],
                params["p_c"],
                params["interaction"],
            )
            early = [_sigmoid(const + z_c * z + p_c * 0.1 + interaction * z * 0.1) for z in z_grid]
            late = [_sigmoid(const + z_c * z + p_c * 0.9 + interaction * z * 0.9) for z in z_grid]

            axis_label = "-TLE (certainty, stage-wise z)" if sig == "tle" else "VC (stage-wise z)"
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            (early_line,) = ax.plot(z_grid, early, label="early (position_norm=0.1), fitted")
            (late_line,) = ax.plot(z_grid, late, label="late (position_norm=0.9), fitted")

            if steps is not None:
                empirical = _empirical_position_binned_means(steps, signal=sig, domain=dom)
                if empirical:
                    for label_key, line in (("early", early_line), ("late", late_line)):
                        pts = empirical.get(label_key)
                        if not pts:
                            continue
                        ax.scatter(
                            [pt["z"] for pt in pts],
                            [pt["y"] for pt in pts],
                            color=line.get_color(),
                            edgecolor="black",
                            linewidth=0.5,
                            s=[max(15, min(120, pt["n"] / 10)) for pt in pts],
                            zorder=3,
                            label=f"{label_key} (position < / >= 0.5), empirical (binned)",
                        )

            ax.set_xlabel(f"z-standardized {axis_label}")
            ax.set_ylabel("Predicted P(correct)")
            ax.set_ylim(0, 1)
            ax.set_title(f"H3 marginal effect: {dom}/{sig} (interaction={interaction:.3f})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7, frameon=False)
            fig.tight_layout()
            p = output_dir / f"h3_marginal_effect_{dom}_{sig}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            written[f"h3_marginal_effect_{dom}_{sig}"] = str(p)

    return written


def plot_signal_histograms(
    steps: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Phase 1 analysis Stage 1: one figure per domain, TLE and VC histograms side by side --
    the raw-variable distribution picture the descriptive codebook table
    (``src/analysis/descriptive_stats.py``) summarizes numerically.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    by_dom: dict[str, list[dict[str, Any]]] = {}
    for r in steps:
        if isinstance(r, dict):
            by_dom.setdefault(str(r.get("domain", "unknown")), []).append(r)

    for dom, rows in sorted(by_dom.items()):
        tle = [_safe_float(r.get("tle_mean_entropy")) for r in rows]
        tle = [v for v in tle if v is not None]
        vc = [_safe_float(r.get("vc")) for r in rows]
        vc = [v for v in vc if v is not None]
        if not tle and not vc:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        if tle:
            axes[0].hist(tle, bins=30, color="tab:blue")
        axes[0].set_title("TLE")
        axes[0].set_xlabel("tle_mean_entropy")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3)
        if vc:
            axes[1].hist(vc, bins=30, color="tab:orange")
        axes[1].set_title("VC")
        axes[1].set_xlabel("vc")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"Signal distributions -- {dom}")
        fig.tight_layout()
        p = output_dir / f"hist_signals_{dom}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written[f"hist_signals_{dom}"] = str(p)

    return written


def plot_signal_boxplots(
    steps: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Phase 1 analysis Stage 1: whisker (box) plots of TLE and VC across domains, side by side.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dom: dict[str, list[dict[str, Any]]] = {}
    for r in steps:
        if isinstance(r, dict):
            by_dom.setdefault(str(r.get("domain", "unknown")), []).append(r)
    domains = sorted(by_dom.keys())
    if not domains:
        return {}

    tle_by_dom = [
        [v for v in (_safe_float(r.get("tle_mean_entropy")) for r in by_dom[d]) if v is not None]
        for d in domains
    ]
    vc_by_dom = [
        [v for v in (_safe_float(r.get("vc")) for r in by_dom[d]) if v is not None] for d in domains
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].boxplot([v if v else [0.0] for v in tle_by_dom], tick_labels=domains)
    axes[0].set_title("TLE by domain")
    axes[0].set_ylabel("tle_mean_entropy")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].boxplot([v if v else [0.0] for v in vc_by_dom], tick_labels=domains)
    axes[1].set_title("VC by domain")
    axes[1].set_ylabel("vc")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = output_dir / "boxplot_signals_by_domain.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return {"boxplot_signals_by_domain": str(p)}


def plot_episode_length_boxplot(
    episodes: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Phase 1 analysis Stage 1: whisker (box) plot of episode length across domains.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dom: dict[str, list[float]] = {}
    for e in episodes:
        if not isinstance(e, dict):
            continue
        dom = str(e.get("domain", "unknown"))
        length = _safe_float(e.get("episode_length_steps"))
        if length is not None:
            by_dom.setdefault(dom, []).append(length)
    domains = sorted(by_dom.keys())
    if not domains:
        return {}

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.boxplot([by_dom[d] for d in domains], tick_labels=domains)
    ax.set_ylabel("Episode length (steps)")
    ax.set_title("Episode length by domain")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = output_dir / "boxplot_episode_length.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return {"boxplot_episode_length": str(p)}


def plot_h2_pareto(
    arms: dict[str, dict[str, float | None]],
    output_dir: str | Path,
    *,
    name: str,
    title: str | None = None,
) -> dict[str, str]:
    """
    H2 Pareto plot: success rate (y) vs. output tokens (x, log scale) per arm, with a bootstrap
    CI cross on both axes (thesis §5.8: "displayed as a Pareto plot of success against output
    tokens with confidence intervals on both axes").

    ``arms`` maps an arm label (e.g. "adaptive_tle", "always_c2") to a dict with keys
    ``tokens_mean``, ``tokens_ci_low``, ``tokens_ci_high``, ``success_mean``, ``success_ci_low``,
    ``success_ci_high`` -- absolute per-arm statistics, not the paired difference that
    ``h2_paired`` returns (the difference is what the confirmatory decision rests on; the
    absolute values are what the plot needs to be readable as a trade-off).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}
    if not arms:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    # Fixed colors for the known spectrum arms so repeated plots stay visually consistent;
    # anything else (e.g. a custom label) falls back to matplotlib's default cycle.
    known_colors = {
        "always_c0": "tab:gray",
        "always_c1": "tab:orange",
        "adaptive_tle": "tab:green",
        "adaptive_vc": "tab:blue",
        "always_c2": "tab:red",
    }
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:purple"])
    for label, s in arms.items():
        x = s.get("tokens_mean")
        y = s.get("success_mean")
        if x is None or y is None:
            continue
        xerr = None
        xlo, xhi = s.get("tokens_ci_low"), s.get("tokens_ci_high")
        if xlo is not None and xhi is not None:
            xerr = [[max(0.0, x - xlo)], [max(0.0, xhi - x)]]
        yerr = None
        ylo, yhi = s.get("success_ci_low"), s.get("success_ci_high")
        if ylo is not None and yhi is not None:
            yerr = [[max(0.0, y - ylo)], [max(0.0, yhi - y)]]
        color = known_colors.get(label, cycle[hash(label) % len(cycle)])
        ax.errorbar(
            [x],
            [y],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=8,
            color=color,
            capsize=4,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean output tokens per episode (log scale)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, which="both")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    p = output_dir / f"h2_pareto_{name}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return {f"h2_pareto_{name}": str(p)}


def plot_bootstrap_distribution(
    reps: list[float],
    output_dir: str | Path,
    *,
    name: str,
    point: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    null_value: float | None = 0.0,
    title: str | None = None,
) -> dict[str, str]:
    """
    Histogram of cluster-bootstrap replicates with point-estimate/CI-bound/null-value markers --
    the standard diagnostic for a percentile-bootstrap CI (visualizes the shape/skew the
    ``skewness`` field already reports numerically, and lets a reader see at a glance whether the
    point estimate falls inside the reported interval). Call this from the stage script that
    actually runs the bootstrap (Stages 2/3/5), not from Stage 6 -- the raw replicate list is
    deliberately dropped from the stage JSON output before it's written to disk (would bloat it
    with thousands of floats per domain), so it only exists in memory at the point of computation.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}
    if not reps:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(reps, bins=50, color="tab:blue", alpha=0.75)
    if point is not None:
        ax.axvline(point, color="black", linewidth=1.5, label=f"point={point:.4f}")
    if ci_low is not None:
        ax.axvline(
            ci_low, color="tab:red", linestyle="--", linewidth=1.2, label=f"CI low={ci_low:.4f}"
        )
    if ci_high is not None:
        ax.axvline(
            ci_high, color="tab:red", linestyle="--", linewidth=1.2, label=f"CI high={ci_high:.4f}"
        )
    if null_value is not None:
        ax.axvline(
            null_value, color="gray", linestyle=":", linewidth=1.2, label=f"null={null_value:.2f}"
        )
    ax.set_xlabel("Bootstrap replicate value")
    ax.set_ylabel("Count")
    ax.set_title(title or f"Bootstrap distribution: {name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    p = output_dir / f"bootstrap_dist_{name}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return {f"bootstrap_dist_{name}": str(p)}
