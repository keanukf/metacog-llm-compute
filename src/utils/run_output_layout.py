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


def finalize_run_debug_views(
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
    *,
    head_chars: int | None = None,
    tail_chars: int | None = None,
) -> Path | None:
    """
    Build ``debug_views/`` under ``run_dir`` when ``logging.write_debug_views`` is enabled.

    Updates ``run_info.json`` with ``debug_views_dir`` and ``debug_views_built`` when present.
    Returns the ``debug_views`` directory path, or ``None`` if skipped or no traces found.
    """
    from src.utils.trace_debug_view import (
        DEFAULT_HEAD_CHARS,
        DEFAULT_TAIL_CHARS,
        build_run_debug_views,
        resolve_step_trace_flags,
    )

    _, write_debug, cfg_head, cfg_tail = resolve_step_trace_flags(config or {})
    if not write_debug:
        return None

    h = int(head_chars) if head_chars is not None else cfg_head
    t = int(tail_chars) if tail_chars is not None else cfg_tail
    if h < 0:
        h = DEFAULT_HEAD_CHARS
    if t < 0:
        t = DEFAULT_TAIL_CHARS

    summary = build_run_debug_views(run_dir, head_chars=h, tail_chars=t)
    debug_dir = Path(run_dir) / "debug_views"
    if summary is None:
        return None

    info_path = Path(run_dir) / "run_info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(info, dict):
                info["debug_views_dir"] = "debug_views"
                info["debug_views_built"] = True
                info["debug_views_episodes"] = summary.get("episodes_built", 0)
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(_json_safe_value(info), f, indent=2)
        except (json.JSONDecodeError, OSError):
            pass

    return debug_dir
