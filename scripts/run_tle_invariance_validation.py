#!/usr/bin/env python3
"""Validate TLE invariance (temperature + batch) against vLLM server under production N."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.backend.server import create_server_backend_from_config
from src.execution.config import (
    ExecutionConfig,
    frozen_execution_params_dict,
    write_frozen_execution_params,
)
from src.execution.parity import load_parity_probes, run_tle_invariance_probes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiment_core.yaml"))
    p.add_argument("--probes", type=Path, default=Path("data/probes/parity_prompts.json"))
    p.add_argument("--output", type=Path, default=Path("data/results/tle_invariance_report.json"))
    p.add_argument("--freeze-metadata-dir", type=Path, default=None)
    args = p.parse_args()

    import yaml

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    exec_cfg = ExecutionConfig.from_config(config, real=True)
    probes_path = args.probes if args.probes.is_absolute() else REPO_ROOT / args.probes
    probes = load_parity_probes(probes_path)
    backend = create_server_backend_from_config(config)
    try:
        report = run_tle_invariance_probes(
            backend,
            probes,
            max_concurrent_episodes=exec_cfg.max_concurrent_episodes,
        )
    finally:
        backend.close()

    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(out_path)}, indent=2))

    if args.freeze_metadata_dir is not None:
        meta_dir = (
            args.freeze_metadata_dir
            if args.freeze_metadata_dir.is_absolute()
            else REPO_ROOT / args.freeze_metadata_dir
        )
        write_frozen_execution_params(
            meta_dir,
            frozen_execution_params_dict(
                max_concurrent_episodes=exec_cfg.max_concurrent_episodes,
                tle_invariance_eps=float(report["batch_invariance"]["eps"]),
                eps_derived_under_load=True,
            ),
        )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
