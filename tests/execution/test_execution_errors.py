"""Tests for scheduler error isolation."""

from __future__ import annotations

import json
from pathlib import Path

from src.execution.scheduler import EpisodeScheduler
from src.execution.worklist import EpisodeJob


def test_http_error_appends_errors_jsonl(tmp_path: Path):
    jobs = [
        EpisodeJob("ep_ok", "textworld", 0, 0, "phase1", compute_stage="C0"),
        EpisodeJob("ep_fail", "textworld", 1, 0, "phase1", compute_stage="C0"),
    ]
    errors_path = tmp_path / "errors.jsonl"

    def run_fn(job: EpisodeJob) -> dict:
        if job.episode_id == "ep_fail":
            return {
                "status": "failed",
                "episode_id": job.episode_id,
                "domain": job.domain,
                "instance": job.instance,
                "stage_or_strategy": job.compute_stage,
                "run": job.run,
                "exc": RuntimeError("HTTP 500"),
                "traceback": "trace",
            }
        return {
            "status": "completed",
            "episode_id": job.episode_id,
            "data": {"tokens": 1},
        }

    stats = EpisodeScheduler(2).run(
        jobs, run_fn=run_fn, errors_path=errors_path, checkpoint_dir=tmp_path
    )
    assert stats.episodes_completed == 1
    assert stats.episodes_failed == 1
    lines = errors_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["episode_id"] == "ep_fail"


def test_pool_continues_after_failure(tmp_path: Path):
    jobs = [
        EpisodeJob(f"ep_{i}", "textworld", i, 0, "phase1", compute_stage="C0") for i in range(4)
    ]

    def run_fn(job: EpisodeJob) -> dict:
        if job.instance == 1:
            raise ValueError("boom")
        return {"status": "completed", "episode_id": job.episode_id, "data": {}}

    stats = EpisodeScheduler(2).run(
        jobs, run_fn=run_fn, errors_path=tmp_path / "e.jsonl", checkpoint_dir=tmp_path
    )
    assert stats.episodes_completed == 3
    assert stats.episodes_failed == 1
