"""Tests for shared experiment environment helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.experiment_env import resolve_textworld_game_path


def test_resolve_textworld_game_path_flat_z8(tmp_path: Path) -> None:
    repo = tmp_path
    tasks = repo / "data" / "tasks"
    tasks.mkdir(parents=True)
    f = tasks / "textworld_2.z8"
    f.write_bytes(b"fake")
    cfg = {"paths": {"tasks_dir": "data/tasks"}}
    p = resolve_textworld_game_path(2, cfg, repo)
    assert p == f


def test_resolve_textworld_game_path_nested_ulx(tmp_path: Path) -> None:
    repo = tmp_path
    nested = repo / "tasks" / "textworld"
    nested.mkdir(parents=True)
    f = nested / "textworld_0.ulx"
    f.write_bytes(b"x")
    cfg = {"paths": {"tasks_dir": "tasks"}}
    p = resolve_textworld_game_path(0, cfg, repo)
    assert p == f


def test_resolve_textworld_game_path_missing(tmp_path: Path) -> None:
    repo = tmp_path
    cfg = {"paths": {"tasks_dir": "data/tasks"}}
    assert resolve_textworld_game_path(99, cfg, repo) is None
