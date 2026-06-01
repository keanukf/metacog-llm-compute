#!/usr/bin/env python3
"""
Run the full pilot suite once per model id, delegating to ``run_pilot.py`` (subprocess).

Output layout::

    data/results/pilot_batch_{UTC_stamp}/
      pilot_batch_manifest.json
      pilot_{UTC_stamp}_{slug}/   # per model — same artifacts as a single run_pilot invocation

**Thesis context:** L0.3 (local LM Studio / GGUF spot-checks of several candidates) fits
``--pilot-mode lmstudio`` — load each model in LM Studio (or point ``LM_STUDIO_BASE_URL`` at
the right server) before the corresponding subprocess. L0.4 (RunPod vLLM model-selection
benchmark) is the formal multi-episode comparison; use ``--pilot-mode cuda`` here when each
vLLM-served ``model.name`` is swapped on the Pod between runs. This script only orchestrates
CLI and output dirs; it does not change inference servers.

Usage::

  python scripts/run_pilot_models.py --config configs/pilot.yaml --pilot-mode lmstudio --real \\
    --models "id1,id2,id3"
  python scripts/run_pilot_models.py --config configs/pilot.yaml --models-file path/to/models.yaml \\
    --pilot-mode cuda --real
  # If --models and --models-file are omitted, ``configs/models.yaml`` is used when present.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_pilot_mode_arg(value: str) -> str:
    """Same rules as ``scripts/run_pilot.parse_pilot_mode_arg`` (keep in sync)."""
    v = (value or "mock").lower().strip()
    if v in ("hf", "m1"):
        raise argparse.ArgumentTypeError(
            f'pilot mode {value!r} was removed; use "lmstudio" or "cuda".'
        )
    allowed = frozenset({"mock", "cuda", "lmstudio"})
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"invalid pilot mode {value!r}; expected one of: mock, cuda, lmstudio"
        )
    return v


def _slugify_model_id(model_id: str, max_len: int = 80) -> str:
    s = model_id.strip()
    for a, b in (("/", "_"), (":", "_"), (" ", "_"), ("\\", "_")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = s.strip("._-") or "model"
    return s[:max_len]


def _load_models_yaml(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        m = raw.get("models")
        if isinstance(m, list):
            return [str(x).strip() for x in m if str(x).strip()]
    raise ValueError(f"expected a list of model ids or {{models: [...]}} in {path}")


def _parse_models_list(s: str | None) -> list[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _unique_slug(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    out = f"{base}-{n}"
    used.add(out)
    return out


def _stderr_tail(s: str | None, max_chars: int = 2000) -> str | None:
    if not s:
        return None
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return "…" + s[-max_chars:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_pilot.py once per model with distinct output directories and an optional manifest."
    )
    parser.add_argument(
        "--config", default="configs/pilot.yaml", help="Pilot config YAML (forwarded)"
    )
    parser.add_argument(
        "--lmstudio-config",
        default=None,
        metavar="PATH",
        help="Forwarded to run_pilot.py when using --pilot-mode lmstudio",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Base directory; batch folder is created under it",
    )
    parser.add_argument(
        "--pilot-mode",
        type=parse_pilot_mode_arg,
        default="mock",
        help="Forwarded: mock | hf | cuda | lmstudio",
    )
    parser.add_argument("--real", action="store_true", help="Forwarded to run_pilot.py")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        default=None,
        help="Forwarded to run_pilot.py (subset of pilot steps)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids (API model.name / HF id / vLLM id); can combine with --models-file",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML: list of ids or {models: [ ... ]}; can combine with --models. "
        "If omitted and --models is unset, uses configs/models.yaml when that file exists.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Run remaining models after a failure; exit non-zero if any failed",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write pilot_batch_manifest.json",
    )
    args = parser.parse_args()

    from_models = _parse_models_list(args.models)
    from_file: list[str] = []
    models_path: Path | None = args.models_file
    if models_path is None and not from_models:
        default_yaml = REPO_ROOT / "configs" / "models.yaml"
        if default_yaml.is_file():
            models_path = default_yaml
    if models_path is not None:
        p = models_path if models_path.is_absolute() else REPO_ROOT / models_path
        if not p.is_file():
            print(f"error: --models-file not found: {p}", file=sys.stderr)
            sys.exit(2)
        from_file = _load_models_yaml(p)

    models: list[str] = []
    seen: set[str] = set()
    for m in from_file + from_models:
        if m not in seen:
            seen.add(m)
            models.append(m)

    if not models:
        print(
            "error: no models — use --models, --models-file, or create configs/models.yaml",
            file=sys.stderr,
        )
        sys.exit(2)

    batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_out = (
        REPO_ROOT / args.output_dir
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    batch_root = base_out / f"pilot_batch_{batch_stamp}"
    batch_root.mkdir(parents=True, exist_ok=False)

    run_pilot_py = REPO_ROOT / "scripts" / "run_pilot.py"
    config_arg = args.config
    if not Path(config_arg).is_absolute():
        config_arg = str(REPO_ROOT / config_arg)

    manifest_runs: list[dict[str, Any]] = []
    any_failed = False
    used_slugs: set[str] = set()

    for model_id in models:
        slug = _unique_slug(_slugify_model_id(model_id), used_slugs)
        run_dir = batch_root / f"pilot_{batch_stamp}_{slug}"
        run_dir.mkdir(parents=False)

        cmd: list[str] = [
            sys.executable,
            str(run_pilot_py),
            "--config",
            config_arg,
            "--output-dir",
            str(run_dir),
            "--no-timestamp-run",
            "--pilot-mode",
            args.pilot_mode,
            "--model-name",
            model_id,
        ]
        if args.real:
            cmd.append("--real")
        if args.lmstudio_config:
            lm = args.lmstudio_config
            if not Path(lm).is_absolute():
                lm = str(REPO_ROOT / lm)
            cmd.extend(["--lmstudio-config", lm])
        if args.only:
            cmd.append("--only")
            cmd.extend(args.only)

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        wall = time.perf_counter() - t0
        err_tail = _stderr_tail(proc.stderr) if proc.returncode != 0 else None
        # Stream child output for visibility (also captured)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
            sys.stdout.flush()
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        entry: dict[str, Any] = {
            "model_id": model_id,
            "output_dir": str(run_dir.resolve()),
            "exit_code": proc.returncode,
            "wall_time_s": round(wall, 3),
        }
        if err_tail:
            entry["stderr_tail"] = err_tail
        manifest_runs.append(entry)

        if proc.returncode != 0:
            any_failed = True
            if not args.continue_on_fail:
                break

    if not args.no_manifest:
        manifest_path = batch_root / "pilot_batch_manifest.json"
        body: dict[str, Any] = {
            "batch_stamp_utc": batch_stamp,
            "config": config_arg,
            "pilot_mode": args.pilot_mode,
            "batch_root": str(batch_root.resolve()),
            "runs": manifest_runs,
            "overall_ok": not any_failed,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
        print(f"Wrote {manifest_path}", file=sys.stderr)

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
