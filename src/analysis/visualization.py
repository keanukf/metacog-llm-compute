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
