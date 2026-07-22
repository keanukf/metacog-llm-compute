#!/usr/bin/env python3
"""Compare batched episode throughput for candidate max_concurrent_episodes values.

Run on Pod after vLLM server is up. Use the result to set production N in
``experiment_core.yaml`` *before* ``verify_backend_parity.py --backend server``.

Example:
  python scripts/instrument_validation/measure_concurrent_throughput.py --real \
    --candidates 1,3,6,8 \
    --output data/results/instrument_validation/throughput_sweep.json
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _latest_smoke_dir(base: Path) -> Path | None:
    dirs = sorted(base.glob("smoke_parallel_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "smoke_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/dev/throughput_probe.yaml"))
    p.add_argument("--candidates", default="1,3,6,8", help="Comma-separated N values to try")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/instrument_validation/throughput_sweep.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/instrument_validation/throughput_sweep"),
    )
    p.add_argument("--real", action="store_true", help="Require vLLM server (fail if unavailable)")
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    base_config = _load_yaml(config_path)
    candidates = [int(x.strip()) for x in str(args.candidates).split(",") if x.strip()]
    out_base = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for n in candidates:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("execution", {})["max_concurrent_episodes"] = n
        tmp_cfg = out_base / f"throughput_n{n}.yaml"
        _write_yaml(tmp_cfg, cfg)
        run_out = out_base / f"n{n}"
        run_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "instrument_validation" / "smoke_parallel.py"),
            "--config",
            str(tmp_cfg),
            "--output-dir",
            str(run_out),
        ]
        if args.real:
            cmd.append("--real")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        smoke_dir = _latest_smoke_dir(run_out)
        summary = _read_summary(smoke_dir) if smoke_dir else {}
        completed = int(summary.get("completed_episodes") or 0)
        ep_per_h = (completed / wall_s * 3600.0) if wall_s > 0 and completed else 0.0
        exec_m = summary.get("execution_metrics") or {}
        row = {
            "max_concurrent_episodes": n,
            "exit_code": proc.returncode,
            "wall_time_s": round(wall_s, 2),
            "completed_episodes": completed,
            "episodes_per_hour": round(ep_per_h, 2),
            "max_in_flight_observed": summary.get("max_in_flight_observed"),
            "avg_tokens_per_episode": exec_m.get("avg_tokens_per_episode"),
            "smoke_go": proc.returncode == 0,
            "run_dir": str(smoke_dir) if smoke_dir else None,
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    viable = [r for r in results if r["smoke_go"] and (r.get("max_in_flight_observed") or 0) >= 1]
    recommended = max(viable, key=lambda r: r["episodes_per_hour"]) if viable else None
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_template": str(config_path),
        "candidates": candidates,
        "results": results,
        "recommended": recommended,
        "note": (
            "Set execution.max_concurrent_episodes in experiment_core.yaml to recommended N "
            "before verify_backend_parity. Match dev configs to the same N."
        ),
    }
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    if recommended:
        print(
            f"Recommended N={recommended['max_concurrent_episodes']} "
            f"({recommended['episodes_per_hour']} ep/h)"
        )
    else:
        print("No candidate passed smoke — inspect results and pick conservative N manually.")
    return 0 if recommended else 1


if __name__ == "__main__":
    raise SystemExit(main())
