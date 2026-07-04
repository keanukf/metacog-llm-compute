"""Tests for shared experiment environment helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.utils.experiment_env import create_experiment_model, resolve_textworld_game_path


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


def test_create_experiment_model_vllm_passes_memory_kwargs() -> None:
    cfg = {
        "model": {"name": "Qwen/Qwen3-8B", "dtype": "fp16"},
        "inference": {
            "backend": "vllm",
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.92,
            "top_logprobs": 20,
        },
    }
    with patch("src.utils.model_wrapper.create_wrapper") as mock_create:
        create_experiment_model(cfg, use_real=True)
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs["dtype"] == "float16"
    assert kwargs["max_model_len"] == 16384
    assert kwargs["gpu_memory_utilization"] == 0.92
