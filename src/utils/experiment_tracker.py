"""
Structured experiment tracking via MLflow.
Integrates with run_pilot, run_phase1, run_phase2 for params, metrics, and artifacts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _flatten_params(obj: Any, prefix: str = "") -> dict[str, str]:
    """Convert nested config to flat string params for MLflow (params must be str)."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out[key] = json.dumps(v) if v is not None else ""
            elif v is None:
                out[key] = ""
            else:
                out[key] = str(v)
    elif isinstance(obj, list):
        out[prefix.rstrip(".")] = json.dumps(obj)
    return out


class ExperimentTracker:
    """
    Thin wrapper around MLflow for experiment runs.
    Use start_run() ... log_* ... end_run() around each experiment.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "metacog-llm-compute",
        s3_endpoint_url: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
    ) -> None:
        if not _MLFLOW_AVAILABLE:
            raise RuntimeError("mlflow is not installed; pip install mlflow boto3")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.s3_endpoint_url = s3_endpoint_url
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key
        self._run_id: str | None = None
        self._active = False

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        if s3_endpoint_url:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
        if aws_access_key:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        if aws_secret_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

        mlflow.set_tracking_uri(tracking_uri)
        self._experiment = mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Start an MLflow run; log config as params and set tags."""
        if self._active:
            self.end_run()
        config = config or {}
        tags = dict(tags or {})
        if "git_sha" not in tags:
            sha = _git_sha()
            if sha:
                tags["git_sha"] = sha
        mlflow.start_run(run_name=run_name, tags=tags)
        self._active = True
        self._run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        flat = _flatten_params(config)
        for k, v in flat.items():
            if len(v) <= 500:  # MLflow param value length limit
                mlflow.log_param(k, v)
            else:
                mlflow.log_param(k, v[:497] + "...")

    def log_episode(self, episode_data: dict[str, Any], step_index: int) -> None:
        """
        Log one episode: metrics (success, steps, tokens, time, TLE/VC summary)
        and upload episode JSON as artifact.
        """
        if not self._active or not _MLFLOW_AVAILABLE:
            return
        ep_id = episode_data.get("episode_id", f"ep_{step_index}")
        mlflow.log_metric("episode_success", 1.0 if episode_data.get("task_success") else 0.0, step=step_index)
        mlflow.log_metric("episode_steps", episode_data.get("steps", 0), step=step_index)
        mlflow.log_metric("episode_tokens", episode_data.get("tokens", 0), step=step_index)
        mlflow.log_metric("episode_wall_clock_time", episode_data.get("wall_clock_time", 0.0), step=step_index)
        tle_per_step = episode_data.get("tle_per_step") or []
        if tle_per_step:
            mean_entropies = [s.get("mean_entropy") for s in tle_per_step if isinstance(s, dict) and "mean_entropy" in s]
            if mean_entropies:
                mlflow.log_metric("episode_mean_tle_entropy", sum(mean_entropies) / len(mean_entropies), step=step_index)
        vc_per_step = episode_data.get("vc_per_step") or []
        vc_values = [v for v in vc_per_step if v is not None]
        if vc_values:
            mlflow.log_metric("episode_last_vc", float(vc_values[-1]) if vc_values else 0.0, step=step_index)
        # Artifact: write to temp file then log (MLflow expects local path or dir)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(episode_data, f, indent=2)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="episodes")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def log_aggregate_metrics(
        self,
        episodes: list[dict[str, Any]],
        ece: float | None = None,
        brier: float | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log aggregate metrics over all episodes (success_rate, mean_tokens, etc.)."""
        if not self._active or not _MLFLOW_AVAILABLE:
            return
        n = len(episodes)
        if n == 0:
            return
        success_count = sum(1 for e in episodes if e.get("task_success"))
        total_tokens = sum(e.get("tokens", 0) for e in episodes)
        total_time = sum(e.get("wall_clock_time", 0.0) for e in episodes)
        mlflow.log_metric("success_rate", success_count / n)
        mlflow.log_metric("mean_tokens_per_episode", total_tokens / n)
        mlflow.log_metric("mean_wall_clock_time_per_episode", total_time / n)
        mlflow.log_metric("total_episodes", n)
        if ece is not None:
            mlflow.log_metric("ece", ece)
        if brier is not None:
            mlflow.log_metric("brier_score", brier)
        for k, v in (extra_metrics or {}).items():
            mlflow.log_metric(k, v)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Upload a file or directory to the run's artifacts (MinIO via MLflow)."""
        if not self._active or not _MLFLOW_AVAILABLE:
            return
        p = Path(local_path)
        if not p.exists():
            return
        mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a batch of metrics (e.g. test1 tok/s, test6 ECE)."""
        if not self._active or not _MLFLOW_AVAILABLE:
            return
        mlflow.log_metrics(metrics, step=step)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run. status: FINISHED | FAILED | KILLED."""
        if not _MLFLOW_AVAILABLE:
            return
        if self._active and mlflow.active_run():
            mlflow.end_run(status=status)
        self._active = False
        self._run_id = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        """Current run's experiment ID (only while run is active)."""
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return None
        return mlflow.active_run().info.experiment_id

    @property
    def is_active(self) -> bool:
        return self._active and _MLFLOW_AVAILABLE and mlflow.active_run() is not None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._active:
            self.end_run(status="FAILED" if exc_type else "FINISHED")
