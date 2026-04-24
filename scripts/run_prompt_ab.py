#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    name: str
    config_path: Path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _rate_nonnull(values: list[Any]) -> float | None:
    if not values:
        return None
    nn = sum(1 for v in values if v is not None)
    return nn / len(values)


def _summarize_variant_dir(run_dir: Path) -> dict[str, Any]:
    # run_pilot.py normally creates a timestamped pilot_* subfolder under --output-dir.
    # Prefer a direct layout (via --no-timestamp-run), but if we see nested pilot_* folders,
    # summarize the newest one.
    root = run_dir
    nested = sorted([p for p in run_dir.glob("pilot_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if nested:
        root = nested[-1]

    san = _read_json(root / "pilot_sanity.json") or {}
    t2 = _read_json(root / "pilot_test2_tle.json") or {}
    t5 = _read_json(root / "pilot_test5_toh.json") or {}

    episodes: list[dict[str, Any]] = []
    for p in sorted(root.glob("ep_textworld_*.json")) + sorted(root.glob("ep_tower_of_hanoi_*.json")):
        ep = _read_json(p)
        if isinstance(ep, dict):
            episodes.append(ep)

    vc_vals: list[Any] = []
    tle_vals: list[Any] = []
    for ep in episodes:
        vc_vals.extend(ep.get("vc_per_step") or [])
        tle_vals.extend(ep.get("tle_per_step") or [])

    return {
        "sanity_has_logprobs": san.get("has_logprobs"),
        "sanity_completion_tokens_observed": san.get("completion_tokens_observed"),
        "test2_mean_entropy_avg": ((t2.get("summary") or {}) if isinstance(t2.get("summary"), dict) else {}).get(
            "mean_entropy_avg"
        ),
        "toh_parse_rate": t5.get("parse_rate"),
        "toh_success_rate": t5.get("success_rate"),
        "toh_avg_optimal_rate": t5.get("avg_optimal_rate"),
        "toh_avg_legal_rate": t5.get("avg_legal_rate"),
        "toh_oscillation_rate": t5.get("oscillation_rate"),
        "vc_nonnull_rate": _rate_nonnull(vc_vals),
        "tle_nonnull_rate": _rate_nonnull(tle_vals),
        "dir": str(root),
    }


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a small A/B sweep over prompt variants.")
    p.add_argument(
        "--variants-dir",
        type=Path,
        default=Path("configs/prompt_variants"),
        help="Directory containing v_*.yaml configs",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/runpod_pilot"),
        help="Base output directory",
    )
    p.add_argument(
        "--pilot-mode",
        type=str,
        default="cuda",
        help="Pilot mode (cuda|hf|lmstudio|mock). For RunPod use cuda.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    variants_dir: Path = (repo_root / args.variants_dir).resolve()
    out_base: Path = (repo_root / args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    variants = [
        Variant("v_base", variants_dir / "v_base.yaml"),
        Variant("v_nofewshot", variants_dir / "v_nofewshot.yaml"),
        Variant("v_think", variants_dir / "v_think.yaml"),
    ]
    for v in variants:
        if not v.config_path.is_file():
            print(f"error: missing variant config: {v.config_path}", file=sys.stderr)
            sys.exit(2)

    stamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ab_root = out_base / f"ab_{stamp}"
    ab_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"ab_root": str(ab_root), "variants": {}}
    for v in variants:
        variant_out = ab_root / v.name
        cmd = [
            sys.executable,
            "scripts/run_pilot.py",
            "--pilot-mode",
            str(args.pilot_mode),
            "--real",
            "--config",
            str(v.config_path),
            "--no-timestamp-run",
            "--only",
            "sanity",
            "test2",
            "test3",
            "test4",
            "test5",
            "--output-dir",
            str(variant_out),
        ]
        print(f"== Running {v.name} ==")
        _run(cmd, cwd=repo_root)
        summary["variants"][v.name] = _summarize_variant_dir(variant_out)

    out_path = ab_root / "ab_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

