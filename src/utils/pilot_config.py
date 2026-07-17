"""
Pilot YAML loading: base config plus optional LM Studio override merge.

When ``--pilot-mode lmstudio``, if ``configs/lmstudio_config.yaml`` exists (or
``--lmstudio-config`` / ``LMSTUDIO_CONFIG_PATH``) and is active, it is deep-merged
into the base pilot config so you can keep hardware-specific settings in one place.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge ``override`` into a copy of ``base``.
    Dict values merge; scalars and lists replace. ``None`` in override skips that key.
    """
    out = deepcopy(base)
    for key, val in override.items():
        if key == "enabled":
            continue
        if val is None:
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def _lmstudio_override_active(override: dict[str, Any]) -> bool:
    """If ``enabled: false``, do not merge; otherwise merge."""
    if override.get("enabled") is False:
        return False
    return True


def _strip_meta_keys(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "enabled"}


def load_yaml_path(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"expected mapping at root of {path}, got {type(raw).__name__}")
    return raw


def load_yaml_with_extends(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Load a YAML config, recursively merging an ``extends: <relative-path>`` key.

    Overlay configs (e.g. ``configs/dev/gate_d_calibration.yaml``) commonly set
    ``extends`` to a base config (usually ``experiment_core.yaml``) and only restate
    the keys they override. A plain single-file load (``load_yaml_path``) silently
    drops every key not restated in the overlay (``model``, ``episode``,
    ``domain_prompts``, ``paths``, ...) — this only surfaces later as a confusing
    ``KeyError``/behavior change, not a load error. This is the same recursive merge
    ``scripts/sweep_textworld_difficulty.py::_load_merged_config`` implements (that
    function now delegates here); kept in one place so the two don't drift apart.

    ``extends`` is resolved relative to the file that declares it, falling back to
    ``repo_root`` (when given) if that path doesn't exist.
    """
    raw = load_yaml_path(path)
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    base_path = path.parent / str(extends)
    if not base_path.is_file() and repo_root is not None:
        base_path = repo_root / str(extends)
    base = load_yaml_with_extends(base_path, repo_root=repo_root)
    merged = dict(base)
    for key, val in raw.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def resolve_lmstudio_config_path(
    repo_root: Path,
    explicit: str | Path | None,
) -> Path:
    """
    Effective LM Studio override path: explicit arg, else ``LMSTUDIO_CONFIG_PATH`` /
    ``LMSTUDIO_CONFIG`` env, else ``configs/lmstudio_config.yaml`` under repo root.
    """
    if explicit is not None and str(explicit).strip() != "":
        p = Path(explicit)
        return p if p.is_absolute() else repo_root / p
    env = os.environ.get("LMSTUDIO_CONFIG_PATH") or os.environ.get("LMSTUDIO_CONFIG")
    if env and str(env).strip():
        p = Path(env.strip())
        return p if p.is_absolute() else repo_root / p
    return repo_root / "configs" / "lmstudio_config.yaml"


def load_pilot_config_with_lmstudio_override(
    base_path: Path,
    pilot_mode: str,
    repo_root: Path,
    lmstudio_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], str | None, Path | None]:
    """
    Load ``base_path`` YAML. For ``pilot_mode == 'lmstudio'``, merge LM Studio
    override file when it exists and is active (``enabled`` not false).

    Returns ``(config, override_note, applied_path)`` where ``applied_path`` is the
    override file when a merge happened, else None.
    """
    config = load_yaml_with_extends(base_path, repo_root=repo_root)
    if pilot_mode != "lmstudio":
        return config, None, None

    path = resolve_lmstudio_config_path(repo_root, lmstudio_config_path)
    if not path.is_file():
        return config, None, None

    try:
        override = load_yaml_path(path)
    except OSError:
        return config, None, None

    if not override:
        return config, None, None
    if not _lmstudio_override_active(override):
        return config, None, None

    stripped = _strip_meta_keys(override)
    merged = deep_merge(config, stripped)
    return merged, f"LM Studio overrides merged from {path}", path
