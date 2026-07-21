"""TypedDicts for the persisted episode artifact (``episode.v1``; ADR-003 schema_version).

Structural contract for the ``ep_*.json`` files written per episode and re-read by
``src.analysis.datasets``. ``StepRecord`` is one environment step (with its TLE/VC/correctness);
``EpisodeRecord`` is one episode. Required vs optional fields are split via TypedDict inheritance so
best-effort validation (``datasets._validate_episode_record``) can accept both phases' records.
"""

from __future__ import annotations

from typing import Any, TypedDict


class _StepRecordRequired(TypedDict):
    step_index: int
    compute_stage: str
    action: str
    tokens_generated: int
    lm_calls_this_step: int
    step_wall_time_s: float


class StepRecord(_StepRecordRequired, total=False):
    tle: dict[str, float] | None
    vc: float | None
    correctness: str | None
    observation_length_chars: int


class _EpisodeRecordRequired(TypedDict):
    episode_id: str
    domain: str
    compute_stage: str
    task_success: bool
    episode_length_steps: int
    total_lm_calls: int
    total_tokens_generated: int
    wall_clock_time: float
    tle_per_step: list[dict[str, float] | None]
    vc_per_step: list[float | None]
    steps_detail: list[StepRecord]


class EpisodeRecord(_EpisodeRecordRequired, total=False):
    schema_version: str
    extra: dict[str, Any]
