"""Pre-analysis data quality screen (thesis §5.8)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.analysis.inference import cluster_bootstrap


def _variance(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


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
        dom_report: dict[str, Any] = {
            "n_steps": len(rows),
            "n_clusters": len(clusters),
            "tle_variance": _variance(tle_vals),
            "vc_variance": _variance(vc_vals),
            "vc_missing_rate": vc_missing / len(rows) if rows else None,
            "vc_modal_share": (vc_mode[0][1] / len(vc_vals)) if vc_vals and vc_mode else None,
            "y_optimal_positive_rate": (sum(y_opt) / len(y_opt)) if y_opt else None,
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
        "| Domain | Steps | Clusters | VC missing | y+ rate |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for dom, d in report.get("by_domain", {}).items():
        rate = d.get("y_optimal_positive_rate")
        miss = d.get("vc_missing_rate")
        md_lines.append(
            f"| {dom} | {d.get('n_steps')} | {d.get('n_clusters')} | "
            f"{miss:.3f if miss is not None else 'n/a'} | "
            f"{rate:.3f if rate is not None else 'n/a'} |\n"
        )
    md_path = out / "preanalysis_screen.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    return json_path, md_path
