"""MLflow API helpers for the dashboard."""
from __future__ import annotations

import os
from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    MlflowClient = None


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_client() -> Any | None:
    if not _AVAILABLE or not MlflowClient:
        return None
    return MlflowClient(get_tracking_uri())


def list_experiments() -> list[dict]:
    """List experiments with name and experiment_id."""
    client = get_client()
    if not client:
        return []
    try:
        exps = client.search_experiments()
        return [{"name": e.name, "experiment_id": e.experiment_id} for e in exps]
    except Exception:
        return []


def search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str = "",
    max_results: int = 100,
    order_by: list[str] | None = None,
) -> list[Any]:
    """Search runs. Returns list of Run objects."""
    client = get_client()
    if not client:
        return []
    try:
        return client.search_runs(
            experiment_ids=experiment_ids or [],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["attributes.start_time DESC"],
        )
    except Exception:
        return []


def get_run(run_id: str) -> Any | None:
    client = get_client()
    if not client:
        return None
    try:
        return client.get_run(run_id)
    except Exception:
        return None


def get_experiment_by_name(name: str) -> str | None:
    """Return experiment_id for given name."""
    client = get_client()
    if not client:
        return None
    try:
        exp = client.get_experiment_by_name(name)
        return exp.experiment_id if exp else None
    except Exception:
        return None
