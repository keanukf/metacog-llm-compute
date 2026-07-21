"""Tests for EpisodeScheduler."""

from __future__ import annotations

import time
from pathlib import Path

from src.execution.scheduler import EpisodeScheduler
from src.execution.worklist import EpisodeJob


class _SlowBackend:
    def generate(self, prompt, logprobs=False, **kwargs):
        time.sleep(0.05)
        return "ok", [
            {"logprob": -0.1, "token": "ok", "top_logprobs": [{"token": "ok", "logprob": -0.1}]}
        ]


def test_scheduler_sequential_when_n1():
    jobs = [
        EpisodeJob("ep_a", "textworld", 0, 0, "phase1", compute_stage="C0"),
        EpisodeJob("ep_b", "textworld", 0, 0, "phase1", compute_stage="C1"),
    ]
    seen: list[str] = []

    def run_fn(job: EpisodeJob) -> dict:
        seen.append(job.episode_id)
        return {"status": "completed", "episode_id": job.episode_id, "data": {"tokens": 1}}

    stats = EpisodeScheduler(1).run(jobs, run_fn=run_fn)
    assert stats.episodes_completed == 2
    assert stats.max_in_flight_observed == 1
    assert seen == ["ep_a", "ep_b"]


def test_scheduler_parallel_max_in_flight(tmp_path: Path):
    jobs = [
        EpisodeJob(f"ep_{i}", "textworld", i, 0, "phase1", compute_stage="C0") for i in range(6)
    ]

    def run_fn(job: EpisodeJob) -> dict:
        time.sleep(0.08)
        return {
            "status": "completed",
            "episode_id": job.episode_id,
            "data": {"total_tokens_generated": 5, "task_success": True, "steps": 1},
        }

    stats = EpisodeScheduler(3).run(jobs, run_fn=run_fn, errors_path=tmp_path / "errors.jsonl")
    assert stats.episodes_completed == 6
    assert stats.max_in_flight_observed > 1
