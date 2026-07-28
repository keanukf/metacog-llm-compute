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
    # Added 2026-07-28 (revision_audit P1-stat-7): backend-reported input-token count for this
    # step's LM call(s), booked per candidate like tokens_generated. Absent on episodes collected
    # before this field existed (Phase 1) -- optional by design, not a missing-data bug.
    prompt_tokens: int


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
    # Added 2026-07-28 (revision_audit P1-stat-7): "Total Tokens Processed" secondary DV
    # (input side; total_tokens_generated already covers output). Deliberately absent on Phase 1
    # episodes (economy decision) -- Phase 1 analyses never depend on it. Present from Phase 2 on.
    total_prompt_tokens: int
