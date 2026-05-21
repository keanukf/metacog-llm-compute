"""Tests for pilot YAML merge (LM Studio override)."""

from __future__ import annotations

from pathlib import Path

from src.utils.pilot_config import (
    deep_merge,
    load_pilot_config_with_lmstudio_override,
    load_yaml_path,
)


def test_deep_merge_nested_and_none_skip() -> None:
    base = {"model": {"name": "a", "dtype": "f16"}, "pilot": {"instances": 5}}
    ov = {"model": {"name": "b"}, "other": 1}
    m = deep_merge(base, ov)
    assert m["model"]["name"] == "b"
    assert m["model"]["dtype"] == "f16"
    assert m["pilot"]["instances"] == 5
    assert m["other"] == 1


def test_deep_merge_strips_enabled() -> None:
    base = {"model": {"name": "x"}}
    m = deep_merge(base, {"enabled": True, "model": {"name": "y"}})
    assert "enabled" not in m
    assert m["model"]["name"] == "y"


def test_lmstudio_override_disabled_no_merge(tmp_path: Path) -> None:
    base = tmp_path / "pilot.yaml"
    base.write_text("model:\n  name: base-model\n", encoding="utf-8")
    lm = tmp_path / "lm.yaml"
    lm.write_text("enabled: false\nmodel:\n  name: other\n", encoding="utf-8")
    cfg, note, applied = load_pilot_config_with_lmstudio_override(
        base, "lmstudio", tmp_path, lmstudio_config_path=lm
    )
    assert cfg["model"]["name"] == "base-model"
    assert note is None
    assert applied is None


def test_lmstudio_override_merge(tmp_path: Path) -> None:
    base = tmp_path / "pilot.yaml"
    base.write_text(
        "model:\n  name: base-model\n  dtype: int4\ninference:\n  max_tokens: 256\n",
        encoding="utf-8",
    )
    lm = tmp_path / "lm.yaml"
    lm.write_text(
        "enabled: true\nmodel:\n  name: lm-model\ninference:\n  max_tokens: 128\n",
        encoding="utf-8",
    )
    cfg, note, applied = load_pilot_config_with_lmstudio_override(
        base, "lmstudio", tmp_path, lmstudio_config_path=lm
    )
    assert cfg["model"]["name"] == "lm-model"
    assert cfg["model"]["dtype"] == "int4"
    assert cfg["inference"]["max_tokens"] == 128
    assert note is not None
    assert applied == lm


def test_non_lmstudio_mode_ignores_override(tmp_path: Path) -> None:
    base = tmp_path / "pilot.yaml"
    base.write_text("model:\n  name: base\n", encoding="utf-8")
    lm = tmp_path / "lm.yaml"
    lm.write_text("enabled: true\nmodel:\n  name: shadow\n", encoding="utf-8")
    cfg, note, applied = load_pilot_config_with_lmstudio_override(
        base, "cuda", tmp_path, lmstudio_config_path=lm
    )
    assert cfg["model"]["name"] == "base"
    assert applied is None


def test_load_yaml_path_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml_path(p) == {}
