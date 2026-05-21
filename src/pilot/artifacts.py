"""
Pilot artifact helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logging_utils import (
    write_logprob_distribution_artifacts,
    write_vc_distribution_artifacts,
)
from src.utils.run_progress import log


def logprob_export_settings(config: dict[str, Any]) -> tuple[bool, str, str]:
    lg = config.get("logging") or {}
    return (
        bool(lg.get("save_logprob_distributions", False)),
        str(lg.get("logprob_export_format", "json")).lower(),
        str(lg.get("logprob_subdir", "logprobs")),
    )


def vc_export_settings(config: dict[str, Any]) -> tuple[bool, str, str]:
    lg = config.get("logging") or {}
    return (
        bool(lg.get("save_vc_distributions", False)),
        str(lg.get("vc_export_format", "json")).lower(),
        str(lg.get("vc_subdir", "vc")),
    )


def maybe_write_logprob_artifacts(
    config: dict[str, Any],
    episode_id: str,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    save, fmt, sub = logprob_export_settings(config)
    if not save:
        return
    raw = result.get("logprob_raw_per_step")
    if not raw:
        return
    for path in write_logprob_distribution_artifacts(
        episode_id, raw, output_dir, export_format=fmt, logprob_subdir=sub
    ):
        log(f"Wrote {path}")


def maybe_write_vc_artifacts(
    config: dict[str, Any],
    episode_id: str,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    save, fmt, sub = vc_export_settings(config)
    if not save:
        return
    raw = result.get("vc_detail_per_step")
    if not raw:
        return
    for path in write_vc_distribution_artifacts(
        episode_id, raw, output_dir, export_format=fmt, vc_subdir=sub
    ):
        log(f"Wrote {path}")


def save_json(output_dir: Path, filename_stem: str, data: dict[str, Any]) -> Path:
    """Save a single JSON artifact to output_dir with a predictable filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_stem}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {path}")
    return path


def load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def load_episode_jsons(output_dir: Path, pattern: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob(pattern)):
        try:
            with open(path) as f:
                ep = json.load(f)
            if isinstance(ep, dict):
                out.append(ep)
        except Exception:
            continue
    return out
