#!/usr/bin/env python3
"""
Summarize a `pilot_batch_*` folder produced by `scripts/run_pilot_models.py`.

Prints a compact table (TSV) to stdout with one row per model run directory.

Usage::

  python scripts/summarize_pilot_batch.py data/results/pilot_batch_20260423_120000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def summarize_run_dir(run_dir: Path) -> dict[str, Any]:
    run_info = _read_json(run_dir / "run_info.json") or {}
    model = run_info.get("model_name") or run_dir.name

    t1 = _read_json(run_dir / "pilot_test1_inference.json") or {}
    t3 = _read_json(run_dir / "pilot_test3_vc.json") or {}
    t5 = _read_json(run_dir / "pilot_test5_toh.json") or {}
    san = _read_json(run_dir / "pilot_sanity.json") or {}
    fea = _read_json(run_dir / "pilot_feasibility.json") or {}
    t2 = _read_json(run_dir / "pilot_test2_tle.json") or {}

    return {
        "model": str(model),
        "dir": str(run_dir),
        "tok_s": t1.get("tokens_per_sec"),
        "vc_parse_rate": fea.get("summary", {}).get("vc_parse_rate") if isinstance(fea.get("summary"), dict) else None,
        "test3_parse_rate": t3.get("parse_rate"),
        "tle_mean_entropy_avg": _safe_get(t2, "summary", "mean_entropy_avg"),
        "toh_parse_rate": t5.get("parse_rate"),
        "toh_success_rate": t5.get("success_rate"),
        "toh_oscillation_rate": t5.get("oscillation_rate"),
        "toh_avg_optimal_rate": t5.get("avg_optimal_rate"),
        "feasibility_go": fea.get("go"),
        "feasibility_passed": f"{fea.get('passed')}/{fea.get('total')}" if fea.get("passed") is not None else None,
        "sanity_has_logprobs": san.get("has_logprobs"),
        "sanity_completion_tokens_observed": san.get("completion_tokens_observed"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize pilot_batch_* outputs from run_pilot_models.py")
    p.add_argument("batch_dir", type=Path, help="Path to data/results/pilot_batch_<UTC>/")
    args = p.parse_args()

    batch_dir = args.batch_dir
    if not batch_dir.is_dir():
        print(f"error: not a directory: {batch_dir}", file=sys.stderr)
        sys.exit(2)

    manifest = _read_json(batch_dir / "pilot_batch_manifest.json")
    run_dirs: list[Path] = []
    if isinstance(manifest, dict) and isinstance(manifest.get("runs"), list):
        for r in manifest["runs"]:
            if isinstance(r, dict) and r.get("output_dir"):
                run_dirs.append(Path(str(r["output_dir"])))
    else:
        # Fallback: any child directory containing run_info.json
        run_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and (d / "run_info.json").is_file()])

    rows = [summarize_run_dir(d) for d in run_dirs if d.is_dir()]

    cols = [
        "model",
        "tok_s",
        "sanity_has_logprobs",
        "sanity_completion_tokens_observed",
        "tle_mean_entropy_avg",
        "vc_parse_rate",
        "test3_parse_rate",
        "toh_parse_rate",
        "toh_success_rate",
        "toh_oscillation_rate",
        "toh_avg_optimal_rate",
        "feasibility_go",
        "feasibility_passed",
        "dir",
    ]
    print("\t".join(cols))
    for row in rows:
        print("\t".join("" if row.get(c) is None else str(row.get(c)) for c in cols))


if __name__ == "__main__":
    main()
