"""Task manifest loading (holdout flags, difficulty tiers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_manifest_path(domain: str, config: dict, repo_root: Path) -> Path | None:
    paths = config.get("paths") or {}
    manifests = paths.get("task_manifests") or {}
    rel = manifests.get(domain)
    if not rel:
        return None
    p = Path(str(rel))
    if not p.is_absolute():
        p = repo_root / p
    return p if p.is_file() else None


def load_manifest(
    domain: str,
    config: dict,
    repo_root: Path,
) -> dict[int, dict[str, Any]]:
    """Return ``instance_id -> manifest entry``; empty dict if manifest missing."""
    path = _resolve_manifest_path(domain, config, repo_root)
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            iid = int(e["instance_id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[iid] = dict(e)
    return out


def manifest_entry_for_instance(
    domain: str,
    instance: int,
    config: dict,
    repo_root: Path,
) -> dict[str, Any]:
    """Manifest row for one instance, or empty dict."""
    return load_manifest(domain, config, repo_root).get(int(instance), {})
