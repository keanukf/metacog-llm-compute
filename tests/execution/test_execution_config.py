"""Execution config defaults and the frozen-parameter guard.

Verifies defaults, the frozen-execution-params metadata write, and that a later mismatch against
those frozen values either warns or hard-fails as configured. Freezing execution parameters is a
reproducibility guarantee: Phase 2 must run under the same sampling/concurrency settings the
pilot and Gate checks were calibrated on, so a silent drift here would break comparability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.execution.config import ExecutionConfig, write_frozen_execution_params


def test_execution_config_defaults():
    cfg = ExecutionConfig.from_config({})
    assert cfg.max_concurrent_episodes == 1
    assert cfg.backend_mode == "server"
    assert "8000" in cfg.server_url


def test_execution_config_frozen_mismatch_warns():
    cfg = ExecutionConfig(
        max_concurrent_episodes=4,
        frozen_max_concurrent_episodes=2,
        n_mismatch_mode="warn",
    )
    msgs = cfg.validate_frozen()
    assert msgs


def test_execution_config_frozen_mismatch_hard_fail():
    cfg = ExecutionConfig(
        max_concurrent_episodes=4,
        frozen_max_concurrent_episodes=2,
        n_mismatch_mode="hard_fail",
    )
    with pytest.raises(SystemExit):
        cfg.enforce_frozen_or_exit()


def test_write_frozen_execution_params_creates_metadata(tmp_path: Path) -> None:
    params = {
        "max_concurrent_episodes": 32,
        "tle_invariance_eps": 0.05,
        "eps_derived_under_load": True,
    }
    write_frozen_execution_params(tmp_path, params)
    meta = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["frozen_execution_params"]["max_concurrent_episodes"] == 32
