"""Tests for pilot YAML merge (LM Studio override)."""

from __future__ import annotations

from pathlib import Path

from src.utils.pilot_config import (
    deep_merge,
    load_pilot_config_with_lmstudio_override,
    load_yaml_path,
    load_yaml_with_extends,
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


def test_load_yaml_with_extends_merges_base(tmp_path: Path) -> None:
    """An overlay's ``extends`` must pull in base keys it doesn't restate.

    Regression for the same bug class fixed in run_phase1.py/run_phase2.py
    (docs/consistency_log.md 2026-07-17): a plain single-file YAML load of an
    overlay config silently drops every base key (model, episode, ...).
    """
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n  name: base-model\n  dtype: bf16\nepisode:\n  max_steps_per_episode: 40\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "extends: base.yaml\nepisode:\n  max_steps_per_episode: 10\nlogging:\n  save_vc: false\n",
        encoding="utf-8",
    )
    merged = load_yaml_with_extends(overlay)
    assert merged["model"] == {"name": "base-model", "dtype": "bf16"}
    assert merged["episode"]["max_steps_per_episode"] == 10
    assert merged["logging"]["save_vc"] is False
    assert "extends" not in merged


def test_load_yaml_with_extends_no_extends_key_passthrough(tmp_path: Path) -> None:
    p = tmp_path / "plain.yaml"
    p.write_text("model:\n  name: solo\n", encoding="utf-8")
    assert load_yaml_with_extends(p) == {"model": {"name": "solo"}}


def test_load_yaml_with_extends_falls_back_to_repo_root(tmp_path: Path) -> None:
    """When the overlay's relative ``extends`` path isn't next to the file, fall back
    to repo_root-relative resolution (mirrors configs/dev/*.yaml pointing at
    ``../experiment_core.yaml`` from a nested dir)."""
    (tmp_path / "configs").mkdir()
    base = tmp_path / "configs" / "core.yaml"
    base.write_text("model:\n  name: core-model\n", encoding="utf-8")
    overlay_dir = tmp_path / "elsewhere"
    overlay_dir.mkdir()
    overlay = overlay_dir / "overlay.yaml"
    overlay.write_text("extends: configs/core.yaml\nepisode:\n  max_steps_per_episode: 5\n")
    merged = load_yaml_with_extends(overlay, repo_root=tmp_path)
    assert merged["model"]["name"] == "core-model"
    assert merged["episode"]["max_steps_per_episode"] == 5


def test_lmstudio_override_resolves_base_extends(tmp_path: Path) -> None:
    """The real bug: run_pilot.py / run_c1_handoff_gate.py / benchmark_inference.py all
    load their --config through load_pilot_config_with_lmstudio_override, which used a
    plain single-file load with no ``extends`` support. Passing one of the Gate D/E
    overlay configs (configs/dev/gate_d_calibration.yaml etc., which all set
    ``extends: ../experiment_core.yaml``) silently dropped the base config's ``model``
    key entirely — verified live against configs/dev/gate_d_diagnostic.yaml before this
    fix (base config had no ``model`` key at all)."""
    base = tmp_path / "experiment_core.yaml"
    base.write_text(
        "model:\n  name: Qwen/Qwen3-8B\ninference:\n  cot_max_tokens: 8192\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "gate_d_calibration.yaml"
    overlay.write_text(
        "extends: experiment_core.yaml\nepisode:\n  max_steps_per_episode: 45\n",
        encoding="utf-8",
    )
    cfg, _note, _applied = load_pilot_config_with_lmstudio_override(overlay, "mock", tmp_path, None)
    assert cfg["model"]["name"] == "Qwen/Qwen3-8B"
    assert cfg["inference"]["cot_max_tokens"] == 8192
    assert cfg["episode"]["max_steps_per_episode"] == 45
