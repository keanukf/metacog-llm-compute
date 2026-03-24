"""
Structured JSON logging for experiment episodes.
One JSON file per episode; used for pilot_calibration and phase1/phase2 results.
"""
from __future__ import annotations

import json
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
