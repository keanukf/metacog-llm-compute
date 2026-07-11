"""Tests for execution config."""

from __future__ import annotations

import pytest

from src.execution.config import ExecutionConfig


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
