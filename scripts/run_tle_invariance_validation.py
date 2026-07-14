#!/usr/bin/env python3
"""Validate TLE invariance (temperature + batch) against vLLM server under production N.

Prefer ``scripts/verify_backend_parity.py --backend server`` for the canonical §5.7.5 report.
This entry point remains for backward compatibility and ``--freeze-metadata-dir`` workflows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiment_core.yaml"))
    p.add_argument("--probes", type=Path, default=Path("data/probes/parity_prompts.json"))
    p.add_argument("--output", type=Path, default=Path("data/results/tle_invariance_report.json"))
    p.add_argument("--freeze-metadata-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output.parent if args.output.is_absolute() else REPO_ROOT / args.output.parent
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_backend_parity.py"),
        "--backend",
        "server",
        "--config",
        str(args.config),
        "--probes",
        str(args.probes),
        "--output-dir",
        str(out_dir),
    ]
    if args.freeze_metadata_dir is not None:
        cmd.extend(["--freeze-metadata-dir", str(args.freeze_metadata_dir)])
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
