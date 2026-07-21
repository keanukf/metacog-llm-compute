"""Bounded parallel episode scheduler (ThreadPoolExecutor).

Runs Phase 1/2 episodes concurrently up to ``max_concurrent_episodes`` (N=32 in production) so vLLM's
continuous batching stays fed. Concurrency here is what the Gate C batch-invariance probe and the
Gate F resume-correctness-under-concurrency check validated, so the in-flight bound and the
completed/quarantined resume sets are correctness-relevant, not just performance knobs.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.execution.episode_runner import append_episode_error
from src.execution.worklist import EpisodeJob
from src.utils.run_resilience import classify_exclusion_reason, write_quarantine


@dataclass
class RunStats:
    episodes_attempted: int = 0
    episodes_completed: int = 0
    episodes_failed: int = 0
    max_in_flight_observed: int = 0
    done_count: int = 0
    rolling: list[dict[str, Any]] = field(default_factory=list)
    total_tokens_generated: int = 0


class EpisodeScheduler:
    """Submits episode jobs to a thread pool and tallies outcomes.

    Serial fast-path when ``max_concurrent_episodes <= 1``; otherwise a ThreadPoolExecutor. Tracks
    peak in-flight count as a check that the effective concurrency matched the frozen N.
    """

    def __init__(self, max_concurrent_episodes: int) -> None:
        self._max_concurrent = max(1, int(max_concurrent_episodes))
        self._in_flight = 0
        self._max_in_flight = 0
        self._lock = threading.Lock()

    @property
    def max_in_flight_observed(self) -> int:
        return self._max_in_flight

    def _inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight += 1
            if self._in_flight > self._max_in_flight:
                self._max_in_flight = self._in_flight

    def _dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def run(
        self,
        jobs: list[EpisodeJob],
        *,
        run_fn: Callable[[EpisodeJob], dict[str, Any]],
        on_complete: Callable[[dict[str, Any], RunStats], None] | None = None,
        errors_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        quarantined: set[str] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> RunStats:
        """Execute all ``jobs`` and return aggregate ``RunStats``.

        Each ``run_fn`` outcome is either completed (checkpoint already written by the worker) or
        failed. Failures split two ways: an infrastructure fault that ``classify_exclusion_reason``
        recognizes is *quarantined* (written to ``checkpoint_dir`` and added to ``quarantined`` so a
        resumed run skips it rather than retrying forever); anything else is appended to
        ``errors_path`` and the run continues. Either way a completed episode's id is durable on
        disk, which is what makes interrupt/resume safe under concurrency.
        """
        stats = RunStats()
        quarantined = quarantined if quarantined is not None else set()
        _log = log_fn or (lambda _m: None)

        def _handle_outcome(outcome: dict[str, Any]) -> None:
            stats.episodes_attempted += 1
            if outcome.get("status") == "completed":
                stats.episodes_completed += 1
                stats.done_count += 1
                data = outcome.get("data") or {}
                stats.total_tokens_generated += int(
                    data.get("total_tokens_generated") or data.get("tokens") or 0
                )
                rolling_entry = {
                    "task_success": bool(data.get("task_success")),
                    "steps": int(data.get("steps") or 0),
                    "ep_wall_time_s": float(outcome.get("ep_wall_time_s") or 0.0),
                    "domain": outcome.get("domain"),
                    "instance": outcome.get("instance"),
                }
                if outcome.get("compute_stage"):
                    rolling_entry["compute_stage"] = outcome.get("compute_stage")
                if outcome.get("strategy"):
                    rolling_entry["strategy"] = outcome.get("strategy")
                stats.rolling.append(rolling_entry)
                if len(stats.rolling) > 10:
                    stats.rolling = stats.rolling[-10:]
            else:
                stats.episodes_failed += 1
                ep_id = str(outcome.get("episode_id", ""))
                exc = outcome.get("exc")
                reason = classify_exclusion_reason(exc) if exc is not None else None
                if reason is not None and checkpoint_dir is not None:
                    write_quarantine(
                        checkpoint_dir,
                        ep_id,
                        reason,
                        meta={
                            "domain": outcome.get("domain"),
                            "instance": outcome.get("instance"),
                            "stage_or_strategy": outcome.get("stage_or_strategy"),
                        },
                    )
                    quarantined.add(ep_id)
                    _log(f"QUARANTINE {ep_id} ({reason})")
                elif errors_path is not None:
                    append_episode_error(
                        errors_path,
                        episode_id=ep_id,
                        domain=str(outcome.get("domain", "")),
                        instance=int(outcome.get("instance") or 0),
                        stage_or_strategy=str(outcome.get("stage_or_strategy", "")),
                        run=int(outcome.get("run") or 0),
                        traceback_text=str(outcome.get("traceback") or ""),
                    )
                    _log(f"Warning: episode failed {ep_id} (continuing)")
            if on_complete is not None:
                on_complete(outcome, stats)

        if self._max_concurrent <= 1:
            for job in jobs:
                self._inc_in_flight()
                try:
                    outcome = run_fn(job)
                finally:
                    self._dec_in_flight()
                _handle_outcome(outcome)
            stats.max_in_flight_observed = self._max_in_flight
            return stats

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = {}

            def _wrapped(j: EpisodeJob) -> dict[str, Any]:
                self._inc_in_flight()
                try:
                    return run_fn(j)
                finally:
                    self._dec_in_flight()

            for job in jobs:
                futures[pool.submit(_wrapped, job)] = job
            for fut in as_completed(futures):
                try:
                    outcome = fut.result()
                except Exception as exc:
                    job = futures[fut]
                    outcome = {
                        "status": "failed",
                        "episode_id": job.episode_id,
                        "domain": job.domain,
                        "instance": job.instance,
                        "stage_or_strategy": job.compute_stage or job.strategy or "",
                        "run": job.run,
                        "exc": exc,
                        "traceback": __import__("traceback").format_exc(),
                    }
                _handle_outcome(outcome)
        stats.max_in_flight_observed = self._max_in_flight
        return stats
