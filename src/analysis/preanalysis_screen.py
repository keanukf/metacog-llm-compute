"""Pre-analysis data quality screen (thesis §5.8)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.analysis.calibration import signal_discrimination_report
from src.analysis.inference import cluster_bootstrap

# Step count below which AUROC is pipeline-smoke only (not a scientific estimate).
AUROC_INTERPRET_MIN_STEPS = 50


def _variance(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


def _discrimination_for_domain(
    episodes: list[dict[str, Any]],
    domain: str,
    signal: str,
) -> dict[str, Any]:
    dom_eps = [e for e in episodes if str(e.get("domain")) == domain]
    if not dom_eps:
        return {"auroc": None, "n_steps": 0, "cohens_d": None}
    rep = signal_discrimination_report(dom_eps, signal, collapse_policy="optimal_only")
    return {
        "auroc": rep.get("auroc"),
        "n_steps": rep.get("n_steps"),
        "cohens_d": rep.get("cohens_d"),
        "mean_signal_correct": rep.get("mean_signal_correct"),
        "mean_signal_incorrect": rep.get("mean_signal_incorrect"),
    }


def run_preanalysis_screen(
    steps: list[dict[str, Any]],
    episodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable screen metrics."""
    episodes = episodes or []
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in steps:
        by_dom[str(r.get("domain", "unknown"))].append(r)

    screen: dict[str, Any] = {"by_domain": {}, "episodes": {}}
    for dom, rows in sorted(by_dom.items()):
        tle_vals = [
            float(r["tle_mean_entropy"]) for r in rows if r.get("tle_mean_entropy") is not None
        ]
        vc_vals = [float(r["vc"]) for r in rows if r.get("vc") is not None]
        vc_missing = sum(1 for r in rows if r.get("vc") is None)
        y_opt = [int(r["y_optimal"]) for r in rows if r.get("y_optimal") is not None]
        clusters = {str(r.get("instance_key")) for r in rows}
        vc_mode = Counter(int(round(v)) for v in vc_vals).most_common(1)
        tle_disc = _discrimination_for_domain(episodes, dom, "tle")
        vc_disc = _discrimination_for_domain(episodes, dom, "vc")
        tle_n = int(tle_disc.get("n_steps") or 0)
        vc_n = int(vc_disc.get("n_steps") or 0)
        dom_report: dict[str, Any] = {
            "n_steps": len(rows),
            "n_clusters": len(clusters),
            "tle_variance": _variance(tle_vals),
            "vc_variance": _variance(vc_vals),
            "vc_missing_rate": vc_missing / len(rows) if rows else None,
            "vc_modal_share": (vc_mode[0][1] / len(vc_vals)) if vc_vals and vc_mode else None,
            "y_optimal_positive_rate": (sum(y_opt) / len(y_opt)) if y_opt else None,
            "tle_auroc": tle_disc.get("auroc"),
            "tle_auroc_n_steps": tle_n,
            "tle_auroc_interpretable": tle_n >= AUROC_INTERPRET_MIN_STEPS,
            "tle_cohens_d": tle_disc.get("cohens_d"),
            "vc_auroc": vc_disc.get("auroc"),
            "vc_auroc_n_steps": vc_n,
            "vc_auroc_interpretable": vc_n >= AUROC_INTERPRET_MIN_STEPS,
            "vc_cohens_d": vc_disc.get("cohens_d"),
        }
        if tle_vals:
            boot = cluster_bootstrap(
                rows,
                lambda rs: (
                    _variance(
                        [
                            float(x["tle_mean_entropy"])
                            for x in rs
                            if x.get("tle_mean_entropy") is not None
                        ]
                    )
                    or 0.0
                ),
                n_boot=500,
                seed=1,
            )
            dom_report["bootstrap_skewness_tle_var"] = boot.get("skewness")
        screen["by_domain"][dom] = dom_report

    ep_lens = [int(e.get("episode_length_steps") or e.get("steps") or 0) for e in episodes]
    screen["episodes"] = {
        "n_episodes": len(episodes),
        "mean_length": (sum(ep_lens) / len(ep_lens)) if ep_lens else None,
    }
    return screen


def write_preanalysis_report(
    steps: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    episodes: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_preanalysis_screen(steps, episodes=episodes)
    json_path = out / "preanalysis_screen.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_lines = [
        "# Pre-analysis screen\n",
        "| Domain | Steps | Clusters | VC missing | y+ rate | TLE AUROC | VC AUROC |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for dom, d in report.get("by_domain", {}).items():
        rate = d.get("y_optimal_positive_rate")
        miss = d.get("vc_missing_rate")
        tle_a = d.get("tle_auroc")
        vc_a = d.get("vc_auroc")
        tle_ok = d.get("tle_auroc_interpretable")
        vc_ok = d.get("vc_auroc_interpretable")
        tle_cell = (f"{tle_a:.3f}" if isinstance(tle_a, (int, float)) else "n/a") + (
            "" if tle_ok else " (smoke)"
        )
        vc_cell = (f"{vc_a:.3f}" if isinstance(vc_a, (int, float)) else "n/a") + (
            "" if vc_ok else " (smoke)"
        )
        miss_fmt = f"{miss:.3f}" if miss is not None else "n/a"
        rate_fmt = f"{rate:.3f}" if rate is not None else "n/a"
        md_lines.append(
            f"| {dom} | {d.get('n_steps')} | {d.get('n_clusters')} | "
            f"{miss_fmt} | {rate_fmt} | {tle_cell} | {vc_cell} |\n"
        )
    md_path = out / "preanalysis_screen.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-analysis data quality screen on a run folder")
    parser.add_argument("run_dir", type=Path, help="Phase1/pilot run directory with ep_*.json")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.analysis.datasets import load_run_dataset

    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 2

    ds = load_run_dataset(run_dir)
    if not ds.episodes:
        print(f"error: no ep_*.json episodes in {run_dir}", file=sys.stderr)
        return 2

    json_path, md_path = write_preanalysis_report(ds.steps, run_dir, episodes=ds.episodes)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for dom, d in run_preanalysis_screen(ds.steps, ds.episodes).get("by_domain", {}).items():
        tle_a = d.get("tle_auroc")
        vc_a = d.get("vc_auroc")
        tle_tag = "" if d.get("tle_auroc_interpretable") else " (smoke-only)"
        vc_tag = "" if d.get("vc_auroc_interpretable") else " (smoke-only)"
        tle_str = f"{tle_a:.3f}" if isinstance(tle_a, (int, float)) else "n/a"
        vc_str = f"{vc_a:.3f}" if isinstance(vc_a, (int, float)) else "n/a"
        print(f"  {dom}: tle_auroc={tle_str}{tle_tag} vc_auroc={vc_str}{vc_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
