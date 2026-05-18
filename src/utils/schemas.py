from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class StepRecord(TypedDict):
    step_index: int
    compute_stage: str
    action: str
    tokens_generated: int
    lm_calls_this_step: int
    step_wall_time_s: float
    tle: NotRequired[dict[str, float] | None]
    vc: NotRequired[float | None]
    correctness: NotRequired[str | None]
    observation_length_chars: NotRequired[int]


class EpisodeRecord(TypedDict):
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
    schema_version: NotRequired[str]
    extra: NotRequired[dict[str, Any]]
