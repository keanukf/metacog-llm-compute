"""Minimal episode quarantine helpers (prereg exclusion reasons only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.errors import EnvStateError, LabelError

EXCLUSION_REASONS = frozenset({"env_assertion", "label_error"})


def classify_exclusion_reason(exc: Exception) -> str | None:
    """Map typed episode failures to prereg exclusion reason codes, if any."""
    if isinstance(exc, EnvStateError):
        return "env_assertion"
    if isinstance(exc, LabelError):
        return "label_error"
    return None


def load_quarantined_episode_ids(d: Path | str) -> set[str]:
    p = Path(d) / "quarantine.jsonl"
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep_id = rec.get("episode_id")
        if isinstance(ep_id, str) and ep_id:
            out.add(ep_id)
    return out


def write_quarantine(
    d: Path | str,
    ep_id: str,
    reason: str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    if reason not in EXCLUSION_REASONS:
        raise ValueError(f"unsupported quarantine reason: {reason!r}")
    rec: dict[str, Any] = {"episode_id": ep_id, "reason": reason}
    if meta:
        rec.update(meta)
    with open(Path(d) / "quarantine.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
