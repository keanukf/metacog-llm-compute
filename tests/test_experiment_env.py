"""Tests for shared experiment environment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.environments.tower_of_hanoi import generate_instances
from src.utils.experiment_env import (
    create_experiment_model,
    make_experiment_env,
    resolve_textworld_game_path,
)


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


def _write_toh_manifest(repo: Path, *, seed: int, num_disks: int) -> None:
    manifest_dir = repo / "data" / "tasks" / "tower_of_hanoi"
    manifest_dir.mkdir(parents=True)
    entry = {
        "instance_id": 0,
        "holdout": False,
        "difficulty_tier": "medium",
        "num_disks_range": [num_disks, num_disks],
        "partial_start_range": [0, 3],
        "partial_start_mode": "random_scramble",
        "task_generation_seed": seed,
    }
    (manifest_dir / "difficulty_manifest.json").write_text(json.dumps({"entries": [entry]}))


def test_make_experiment_env_toh_uses_per_instance_cap_not_flat_config_value(
    tmp_path: Path,
) -> None:
    """Regression: make_experiment_env() used to pass the flat config max_steps straight to
    TowerOfHanoiEnv, ignoring the per-instance 3x-optimal_steps cap generate_instances() already
    computes -- exactly the cap the Gate D corridor calibration was tested and frozen against
    (docs/consistency_log.md, 2026-07-20 Gate F budget re-estimate finding)."""
    seed, num_disks = 271828, 4
    _write_toh_manifest(tmp_path, seed=seed, num_disks=num_disks)
    config = {
        "paths": {
            "task_manifests": {
                "tower_of_hanoi": "data/tasks/tower_of_hanoi/difficulty_manifest.json"
            }
        },
        "tower_of_hanoi": {},
    }

    flat_config_max_steps = 999999  # deliberately absurd, must NOT be what the env ends up with
    env = make_experiment_env("tower_of_hanoi", 0, config, flat_config_max_steps, tmp_path)

    expected_instance = generate_instances(
        1,
        seed=seed,
        num_disks_range=(num_disks, num_disks),
        partial_start_range=(0, 3),
        partial_start_mode="random_scramble",
    )[0]
    assert env.max_steps == expected_instance["max_steps"]
    assert env.max_steps != flat_config_max_steps
    assert env.max_steps == 3 * len(expected_instance["optimal_solution"])


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
