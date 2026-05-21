"""
Timestamped run directories and short run-info sidecars for pilot / phase scripts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe_value(x: Any) -> Any:
    """Recursively convert Path and other non-JSON types to serializable values."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe_value(i) for i in x]
    return x


def make_run_subdirectory(base_dir: str | Path, *, prefix: str = "run") -> Path:
    """
    Create ``base_dir / {prefix}_{YYYYMMDD_HHMMSS}`` (UTC) and return it.

    The parent ``base_dir`` is created if missing; the run subdirectory must not exist yet.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=False)
    return run_dir


def write_short_run_info(
    run_dir: str | Path,
    *,
    script: str,
    config_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Write ``run_info.json`` with a short, human-readable summary of the run.

    Typical ``extra`` keys: pilot_mode, only_steps, model_name, note.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(script),
        "config_path": str(config_path),
        "python": sys.version.split()[0],
    }
    if extra:
        body.update(_json_safe_value(extra))
    path = run_dir / "run_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return path
