"""Compute stage implementations (C0, C1, C2) and shared helpers."""

from src.agent.stages.c0 import c0_step, c0_step_core
from src.agent.stages.c1 import c1_step, c1_step_core
from src.agent.stages.c2 import c2_step, c2_step_core, majority_vote
from src.agent.stages.shared import (
    DEFAULT_VC_FOLLOWUP_INSTRUCTION,
    VC_FOLLOWUP_PROMPT_MARKER,
    StepReturn,
)

__all__ = [
    "DEFAULT_VC_FOLLOWUP_INSTRUCTION",
    "VC_FOLLOWUP_PROMPT_MARKER",
    "StepReturn",
    "c0_step",
    "c0_step_core",
    "c1_step",
    "c1_step_core",
    "c2_step",
    "c2_step_core",
    "majority_vote",
]
