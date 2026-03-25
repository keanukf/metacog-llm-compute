"""
Structured JSON logging for experiment episodes.
One JSON file per episode; used for pilot_calibration and phase1/phase2 results.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_episode(
    episode_id: str,
    data: dict[str, Any],
    path: str | Path,
    tracker: Any = None,
) -> Path:
    """
    Write a single episode's data as one JSON file.

    Args:
        episode_id: Unique id (e.g. ep_{domain}_{instance}_{stage}_{run}).
        data: Dict with keys such as TLE, VC, task_success, steps, lm_calls, tokens, wall_clock_time.
        path: Directory or full file path; if directory, file is path / f"{episode_id}.json".
        tracker: Reserved for optional hooks; ignored by this function (file write only).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    if path.suffix != ".json":
        path = path / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


class EpisodeLogger:
    """Logger that writes one JSON file per episode to a results directory."""

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log(self, episode_id: str, data: dict[str, Any]) -> Path:
        """Log one episode; returns path to written file."""
        return log_episode(episode_id, data, self.results_dir)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _try_git_commit(repo_root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _try_gpu_info() -> tuple[str | None, float | None]:
    """
    Returns (gpu_name, vram_total_gb) if torch+CUDA are available, else (None, None).
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None, None
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = float(props.total_memory) / (1024**3)
        return name, vram_total_gb
    except Exception:
        return None, None


def write_run_metadata(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    *,
    script: str,
    config_path: str | Path,
    pilot_mode: str,
    model_name: str,
    model_dtype: str,
    domains: list[str],
    total_episodes_planned: int,
    resumed_from: int,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write `run_metadata.json` into checkpoint_dir.

    This file is meant to make overnight runs reproducible and debuggable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    meta = {
        "run_id": str(uuid.uuid4()),
        "script": str(script),
        "config_path": str(config_path),
        "config_hash": _sha256_file(config_path),
        "model_name": str(model_name),
        "model_dtype": str(model_dtype),
        "pilot_mode": str(pilot_mode),
        "git_commit": _try_git_commit(repo_root),
        "timestamp_start_utc": _iso_utc_now(),
        "python_version": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_name": None,
        "vram_total_gb": None,
        "domains": list(domains),
        "total_episodes_planned": int(total_episodes_planned),
        "resumed_from": int(resumed_from),
    }
    gpu_name, vram_total = _try_gpu_info()
    meta["gpu_name"] = gpu_name
    meta["vram_total_gb"] = vram_total

    path = checkpoint_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def _synthesize_steps_detail(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a minimal `steps_detail` from legacy episode keys when missing.
    """
    if isinstance(episode.get("steps_detail"), list):
        return list(episode["steps_detail"])
    steps = int(episode.get("steps") or episode.get("episode_length_steps") or 0)
    tle_list = episode.get("tle_per_step") or []
    vc_list = episode.get("vc_per_step") or []
    stage_list = episode.get("stage_per_step") or []
    fixed_stage = episode.get("compute_stage") or None
    step_correctness = episode.get("step_correctness") or []
    corr_by_idx: dict[int, Any] = {}
    if isinstance(step_correctness, list):
        for d in step_correctness:
            if not isinstance(d, dict):
                continue
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            corr_by_idx[idx] = d.get("correctness")

    out: list[dict[str, Any]] = []
    for i in range(steps):
        compute_stage = (
            str(stage_list[i]) if i < len(stage_list) and stage_list[i] is not None else str(fixed_stage) if fixed_stage is not None else "C0"
        )
        tle = tle_list[i] if i < len(tle_list) else None
        vc = vc_list[i] if i < len(vc_list) else None
        out.append(
            {
                "step_index": i,
                "compute_stage": compute_stage,
                "action": "",
                "tokens_generated": 0,
                "lm_calls_this_step": 1,
                "step_wall_time_s": 0.0,
                "tle": tle,
                "vc": vc,
                "correctness": corr_by_idx.get(i),
                "observation_length_chars": 0,
            }
        )
    return out


def load_episodes(
    checkpoint_dir: str | Path,
    as_dataframe: bool = False,
) -> list[dict[str, Any]] | Any:
    """
    Load all episode JSONs from a directory.

    Backward compatibility:
    - Episodes without `steps_detail` will get a synthesized minimal `steps_detail`.

    Args:
        checkpoint_dir: Directory containing `ep_*.json`.
        as_dataframe: If True, return a pandas DataFrame (requires pandas).
    """
    checkpoint_dir = Path(checkpoint_dir)
    episodes: list[dict[str, Any]] = []
    for p in sorted(checkpoint_dir.glob("ep_*.json")):
        try:
            with open(p) as f:
                ep = json.load(f)
            if isinstance(ep, dict):
                ep = dict(ep)
                ep["steps_detail"] = _synthesize_steps_detail(ep)
                episodes.append(ep)
        except Exception:
            continue

    if not as_dataframe:
        return episodes

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("pandas is required for as_dataframe=True") from e
    return pd.DataFrame(episodes)


def load_steps(checkpoint_dir: str | Path):
    """
    Load all episodes and flatten `steps_detail` into one row per step.

    By default this returns a pandas DataFrame when pandas is importable. If pandas is not
    available in the runtime environment, it returns a plain `list[dict]` instead so that
    analysis code can still run in minimal setups.

    Returns:
        pandas.DataFrame (preferred) or list[dict] fallback with episode-level columns joined.
    """
    episodes = load_episodes(checkpoint_dir, as_dataframe=False)
    rows: list[dict[str, Any]] = []
    episode_cols = [
        "episode_id",
        "domain",
        "instance",
        "compute_stage",
        "strategy",
        "run",
        "task_success",
        "episode_length_steps",
        "total_lm_calls",
        "total_tokens_generated",
        "normalized_compute_cost",
        "efficiency_score",
        "timestamp_utc",
    ]
    for ep in episodes:
        base = {k: ep.get(k) for k in episode_cols if k in ep}
        for sd in ep.get("steps_detail") or []:
            if not isinstance(sd, dict):
                continue
            row = dict(base)
            row.update(sd)
            # Unnest tle fields for convenience
            tle = sd.get("tle")
            if isinstance(tle, dict):
                row["tle_mean_entropy"] = tle.get("mean_entropy")
                row["tle_max_entropy"] = tle.get("max_entropy")
            rows.append(row)
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(rows)
    except Exception:
        return rows
