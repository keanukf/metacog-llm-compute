#!/usr/bin/env python3
"""
Post-hoc analysis for a single run folder.

Outputs (under --out-dir, default: <run_dir>/analysis):
- episodes.csv, steps.csv (and optionally parquet)
- analysis_metrics.json
- figures/*.png
- report.md

Usage:
  python scripts/analyze_run.py --run-dir data/results/phase2/phase2_..._UTC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _try_write_table(path: Path, rows: list[dict], *, parquet: bool = False) -> dict[str, str]:
    written: dict[str, str] = {}
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written["csv"] = str(path)
        if parquet:
            pq = path.with_suffix(".parquet")
            df.to_parquet(pq, index=False)
            written["parquet"] = str(pq)
        return written
    except Exception:
        # Fallback: write CSV minimally without pandas
        import csv

        if not rows:
            return {}
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        written["csv"] = str(path)
        return written


def _render_report_md(metrics: dict, figures: dict[str, str]) -> str:
    run_health = metrics.get("run_health") or {}
    lines: list[str] = []
    lines.append("# Run analysis report")
    lines.append("")
    if metrics.get("run_dir"):
        lines.append(f"- **run_dir**: `{metrics['run_dir']}`")
    if metrics.get("episodes") is not None:
        lines.append(f"- **episodes**: {metrics['episodes']}")
    if metrics.get("steps") is not None:
        lines.append(f"- **steps**: {metrics['steps']}")
    lines.append("")

    lines.append("## Run health")
    for k in (
        "episodes_steps_detail_synthesized_rate",
        "episodes_with_logprobs_sidecar_rate",
        "episodes_with_vc_sidecar_rate",
        "episodes_with_trace_rate",
        "missing_vc_rate",
        "missing_tle_rate",
        "missing_step_label_rate",
    ):
        if k in run_health:
            lines.append(f"- **{k}**: {run_health[k]:.3f}")
    lines.append("")

    if metrics.get("efficiency_by_group"):
        lines.append("## Efficiency summary")
        for row in metrics["efficiency_by_group"]:
            keys = [k for k in ("strategy", "compute_stage") if k in row]
            label = row.get(keys[0], "group") if keys else "group"
            lines.append(
                f"- **{label}**: success={row.get('success_rate', 0):.3f}, "
                f"cost={row.get('mean_normalized_compute_cost', 0):.3f}, "
                f"eff={row.get('efficiency')}"
            )
        lines.append("")

    if figures:
        lines.append("## Figures")
        for name, p in figures.items():
            rel = Path(p).as_posix()
            lines.append(f"- **{name}**: `{rel}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir", required=True, help="Run/checkpoint directory containing ep_*.json"
    )
    ap.add_argument("--out-dir", default="", help="Output directory (default: <run-dir>/analysis)")
    ap.add_argument(
        "--parquet", action="store_true", help="Also write Parquet (requires pandas + pyarrow)"
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis")
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.analysis.allocation import (
        efficiency_summary,
        regret_vs_baselines,
        run_health,
        stage_mix,
    )
    from src.analysis.datasets import load_run_dataset
    from src.analysis.visualization import plot_run_overview

    ds = load_run_dataset(run_dir)

    figures_dir = out_dir / "figures"
    try:
        figures = plot_run_overview(ds.episodes, ds.steps, figures_dir)
    except Exception:
        # Plots are best-effort; analysis artifacts must still be written.
        figures = {}

    metrics: dict = {
        "run_dir": str(run_dir),
        "episodes": len(ds.episodes),
        "steps": len(ds.steps),
        "run_health": run_health(ds.episodes, ds.steps),
        "stage_mix": stage_mix(ds.steps),
        "efficiency_by_group": efficiency_summary(ds.episodes),
        "regret_vs_baselines": regret_vs_baselines(ds.episodes),
        "figures": figures,
        "run_metadata": ds.run_metadata,
        "run_info": ds.run_info,
        "run_summary": ds.run_summary,
        "errors": ds.errors,
    }

    _write_json(out_dir / "analysis_metrics.json", metrics)
    _try_write_table(out_dir / "episodes.csv", ds.episodes, parquet=args.parquet)
    _try_write_table(out_dir / "steps.csv", ds.steps, parquet=args.parquet)

    report_md = _render_report_md(metrics, figures)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
