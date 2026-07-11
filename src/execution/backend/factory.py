"""Factory for execution-layer inference backends."""

from __future__ import annotations

from typing import Any

from src.execution.backend.server import create_server_backend_from_config
from src.execution.config import ExecutionConfig
from src.utils.experiment_env import MockExperimentModel


def create_execution_backend(
    config: dict[str, Any],
    *,
    use_real: bool,
) -> Any:
    """
    Return an inference backend for episode workers.

    - ``use_real=False``: ``MockExperimentModel``
    - ``use_real=True``: ``ServerBackend`` (``vllm serve``); ``inprocess`` is reserved
      for a future option and is not implemented in this iteration.
    """
    if not use_real:
        return MockExperimentModel()
    exec_cfg = ExecutionConfig.from_config(config, real=True)
    mode = exec_cfg.backend_mode
    if mode == "server":
        return create_server_backend_from_config(config)
    if mode == "inprocess":
        raise NotImplementedError(
            "execution.backend_mode=inprocess is reserved for a future option; "
            "use backend_mode=server with vllm serve for --real GPU runs"
        )
    raise ValueError(f"Unknown execution.backend_mode: {mode!r}")
