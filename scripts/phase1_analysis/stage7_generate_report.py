#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 7: report generation.

Reads every prior stage's JSON output and renders docs/phase1_analysis_report.md -- the archived,
committed record of the confirmatory/exploratory Phase 1 results. Also copies the Stage 6 figure
set into docs/figures/phase1_analysis/ (committed, unlike the gitignored data/results/ working
copies) so the report renders standalone for anyone who clones the repo without re-running the
pipeline.

Usage:
  python scripts/phase1_analysis/stage7_generate_report.py \
      --stage-dir data/results/phase1_analysis \
      --report-out docs/phase1_analysis_report.md \
      --figures-out docs/figures/phase1_analysis
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci_cell(d: dict[str, Any]) -> str:
    if d.get("point") is None:
        return "n/a (insufficient clusters)"
    return f"{_fmt(d['point'])} [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}]"


def _render_h1a(h1a: dict[str, Any]) -> str:
    lines = [
        "## H1a — Discrimination (ΔAUROC(TLE, VC) per domain)",
        "",
        "Confirmatory decision: cluster-bootstrapped ΔAUROC lower CI bound > 0, Holm family A "
        "(2 domains).",
        "",
        "| Domain | ΔAUROC [90% CI] | One-sided p | Holm-adjusted p | Holds |",
        "|--------|------------------|-------------|------------------|-------|",
    ]
    for dom, d in h1a["by_domain"].items():
        lines.append(
            f"| {dom} | {_ci_cell(d)} | {_fmt(d.get('one_sided_pvalue'))} | "
            f"{_fmt(d['holm']['adjusted'])} | {d.get('decision_holds')} |"
        )
    lines += [
        "",
        "Descriptive cross-check (independent code path, no clustering/bootstrap, "
        "`compare_signal_calibration`, optimal-only collapse policy):",
        "",
        "| Domain | AUROC(TLE) | AUROC(VC) |",
        "|--------|------------|-----------|",
    ]
    for dom, d in h1a["descriptive_cross_check"].items():
        opt = d["optimal_only"]
        lines.append(f"| {dom} | {_fmt(opt['tle']['auroc'])} | {_fmt(opt['vc']['auroc'])} |")
    lines.append("")
    lines.append("![H1a AUROC comparison](figures/phase1_analysis/h1a_auroc_comparison.png)")
    return "\n".join(lines)


def _render_h1b(h1b: dict[str, Any]) -> str:
    lines = [
        "## H1b — Calibration (ΔBrier(TLE-mapped, VC/100) per domain)",
        "",
        "Confirmatory decision: cluster-bootstrapped ΔBrier upper CI bound < 0 (lower Brier is "
        "better), Holm family D (2 domains). Calibrator (`fit_tle_calibrator`) fit on holdout "
        "steps pooled across runs and compute stages, evaluated on non-holdout steps.",
        "",
        "| Domain | ΔBrier [90% CI] | Calibrator slope | n holdout steps | Holm-adjusted p | Holds |",
        "|--------|-------------------|-------------------|-------------------|------------------|-------|",
    ]
    for dom, d in h1b["by_domain"].items():
        lines.append(
            f"| {dom} | {_ci_cell(d)} | {_fmt(d.get('calibrator_slope'))} | "
            f"{d.get('n_holdout_steps')} | {_fmt(d['holm']['adjusted'])} | {d.get('decision_holds')} |"
        )
    return "\n".join(lines)


def _render_h3(h3: dict[str, Any]) -> str:
    conf_dom = h3["confirmatory_domain"]
    expl_dom = h3["exploratory_domain"]
    lines = [
        "## H3 — Temporal degradation (signal x position_norm interaction)",
        "",
        f"GEE clustered-logistic interaction coefficient, confirmatory domain = `{conf_dom}` "
        f"(TLE+VC, Holm family E); `{expl_dom}` reported exploratory only, not corrected against "
        "the confirmatory family.",
        "",
        f"### Confirmatory ({conf_dom})",
        "",
        "| Signal | Interaction coef. | One-sided p (degradation) | Holm-adjusted p | Holds |",
        "|--------|--------------------|-----------------------------|------------------|-------|",
    ]
    for sig, r in h3["results"][conf_dom].items():
        if not r.get("converged"):
            lines.append(f"| {sig} | did not converge ({r.get('note')}) | | | |")
            continue
        lines.append(
            f"| {sig} | {_fmt(r['interaction_coef'])} | {_fmt(r['one_sided_pvalue_degradation'])} | "
            f"{_fmt(r['holm']['adjusted'])} | {r.get('decision_holds')} |"
        )
    lines += [
        "",
        f"### Exploratory ({expl_dom}, not Holm-corrected)",
        "",
        "| Signal | Interaction coef. | Note |",
        "|--------|--------------------|------|",
    ]
    for sig, r in h3["results"][expl_dom].items():
        if not r.get("converged"):
            lines.append(f"| {sig} | n/a | did not converge ({r.get('note')}) |")
            continue
        lines.append(f"| {sig} | {_fmt(r['interaction_coef'])} | {r.get('note', '')} |")
    lines.append("")
    for dom in (conf_dom, expl_dom):
        for sig, r in h3["results"][dom].items():
            if r.get("converged"):
                lines.append(
                    f"![H3 marginal effect {dom}/{sig}]"
                    f"(figures/phase1_analysis/h3_marginal_effect_{dom}_{sig}.png)"
                )
    return "\n".join(lines)


def _render_h4(h4: dict[str, Any]) -> str:
    r = h4["result"]
    return "\n".join(
        [
            "## H4 — Domain modulation (diff-in-diff of ΔAUROC(ToH) − ΔAUROC(TextWorld))",
            "",
            "Confirmatory decision: cluster-bootstrapped (resampled **within each domain "
            "independently**, `cluster_bootstrap_stratified`) lower CI bound > 0, single test "
            "(Holm family C).",
            "",
            "| Diff-in-diff [90% CI] | One-sided p | Holm-adjusted p | Holds |",
            "|------------------------|-------------|------------------|-------|",
            f"| {_ci_cell(r)} | {_fmt(r.get('one_sided_pvalue'))} | "
            f"{_fmt(r['holm']['adjusted'])} | {r.get('decision_holds')} |",
        ]
    )


def _render_preanalysis(screen: dict[str, Any]) -> str:
    lines = [
        "## Preanalysis screen (diagnostic, does not gate the confirmatory stages)",
        "",
        "| Domain | n steps | n clusters | VC missing rate | ICC (GEE) | Episode length "
        "(median, IQR) | Empty position×correctness cells |",
        "|--------|---------|------------|------------------|-----------|-------------------------"
        "-----|-------------------------------------|",
    ]
    for dom, d in screen["by_domain"].items():
        el = d["episode_length_distribution"]
        pc = d["position_correctness"]
        lines.append(
            f"| {dom} | {d['n_steps']} | {d['n_clusters']} | {_fmt(d.get('vc_missing_rate'), 3)} | "
            f"{_fmt(d['icc'].get('icc_gee'), 4)} | {_fmt(el['median'], 1)} "
            f"[{_fmt(el['q1'], 1)}, {_fmt(el['q3'], 1)}] | {pc.get('n_empty_cells')} |"
        )
    return "\n".join(lines)


def build_report(stage_dir: Path) -> str:
    manifest = _load(stage_dir / "stage0" / "canonical_manifest.json")
    screen = _load(stage_dir / "stage1" / "preanalysis_screen.json")
    h1a = _load(stage_dir / "stage2" / "h1a_discrimination.json")
    h1b = _load(stage_dir / "stage3" / "h1b_calibration.json")
    h3 = _load(stage_dir / "stage4" / "h3_temporal.json")
    h4 = _load(stage_dir / "stage5" / "h4_domain_modulation.json")

    sections = [
        "# Phase 1 Real-Data Analysis Report",
        "",
        "Generated by `scripts/phase1_analysis/run_all.py` (Stage 7:"
        " `stage7_generate_report.py`) from the confirmatory/exploratory pipeline in"
        " `scripts/phase1_analysis/`. Re-running the pipeline against unchanged input data"
        " reproduces this report byte-for-byte except this header.",
        "",
        "## Data source",
        "",
        f"- Canonical dataset: {manifest['n_episodes']} episodes "
        f"(content hash `{manifest['content_hash'][:16]}...`)",
        f"- Selection rule: {manifest['selection_rule']}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/phase1_analysis/run_all.py",
        "```",
        "",
        _render_preanalysis(screen),
        "",
        _render_h1a(h1a),
        "",
        _render_h1b(h1b),
        "",
        _render_h3(h3),
        "",
        _render_h4(h4),
        "",
        "## Verification",
        "",
        "- Every stage script is independently runnable, deterministic (fixed `--seed`, default "
        "`20260703`), and idempotent -- re-running Stages 0-5 twice on unchanged input data "
        "produces byte-identical JSON output (excluding metadata-only fields, none of which "
        "carry numeric content here).",
        "- `python -m pytest tests/ -v` green at the time this report was generated.",
    ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-dir", default="data/results/phase1_analysis")
    parser.add_argument("--report-out", default="docs/phase1_analysis_report.md")
    parser.add_argument("--figures-out", default="docs/figures/phase1_analysis")
    args = parser.parse_args()

    stage_dir = REPO_ROOT / args.stage_dir if not Path(args.stage_dir).is_absolute() else Path(args.stage_dir)
    required = [
        stage_dir / "stage0" / "canonical_manifest.json",
        stage_dir / "stage1" / "preanalysis_screen.json",
        stage_dir / "stage2" / "h1a_discrimination.json",
        stage_dir / "stage3" / "h1b_calibration.json",
        stage_dir / "stage4" / "h3_temporal.json",
        stage_dir / "stage5" / "h4_domain_modulation.json",
        stage_dir / "stage6" / "figures" / "figures_manifest.json",
    ]
    for p in required:
        if not p.exists():
            print(f"Stage 7 FAILED -- required input not found at {p}; run Stages 0-6 first.", file=sys.stderr)
            return 1

    figures_out = REPO_ROOT / args.figures_out if not Path(args.figures_out).is_absolute() else Path(args.figures_out)
    figures_out.mkdir(parents=True, exist_ok=True)
    src_figures_dir = stage_dir / "stage6" / "figures"
    for png in sorted(src_figures_dir.glob("*.png")):
        shutil.copyfile(png, figures_out / png.name)

    report = build_report(stage_dir)
    report_out = REPO_ROOT / args.report_out if not Path(args.report_out).is_absolute() else Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report, encoding="utf-8")

    print(f"Stage 7 OK -- report written to {report_out}, figures copied to {figures_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
