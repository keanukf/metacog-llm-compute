#!/usr/bin/env python3
"""Set execution.max_concurrent_episodes from a throughput sweep JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "experiment_core.yaml",
    REPO_ROOT / "configs" / "dev" / "smoke.yaml",
    REPO_ROOT / "configs" / "dev" / "format_vc_probe.yaml",
    REPO_ROOT / "configs" / "dev" / "signal_smoke.yaml",
    REPO_ROOT / "configs" / "dev" / "toh_parse_probe.yaml",
    REPO_ROOT / "configs" / "dev" / "throughput_probe.yaml",
]


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _recommended_n(report: dict) -> int:
    rec = report.get("recommended") or {}
    n = rec.get("max_concurrent_episodes")
    if n is None:
        viable = [r for r in report.get("results", []) if r.get("smoke_go")]
        if not viable:
            raise SystemExit("No viable candidate in sweep report")
        best = max(viable, key=lambda r: float(r.get("episodes_per_hour") or 0))
        n = best["max_concurrent_episodes"]
    return int(n)


def _patch_yaml(path: Path, n: int) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    data.setdefault("execution", {})["max_concurrent_episodes"] = int(n)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "sweep_json",
        type=Path,
        help="Path to throughput_sweep.json (or throughput_sweep_extended.json)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chosen N without editing YAML files",
    )
    args = p.parse_args()
    path = args.sweep_json if args.sweep_json.is_absolute() else REPO_ROOT / args.sweep_json
    report = _load_report(path)
    n = _recommended_n(report)
    print(f"Recommended production N={n}")
    if args.dry_run:
        return 0
    for cfg in CONFIG_PATHS:
        if not cfg.is_file():
            print(f"SKIP missing {cfg}", file=sys.stderr)
            continue
        _patch_yaml(cfg, n)
        print(f"Updated {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
