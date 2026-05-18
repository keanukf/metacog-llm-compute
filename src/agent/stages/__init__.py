"""Stage-specific wrappers for compute stage entrypoints."""

from src.agent.stages.c0 import c0_step
from src.agent.stages.c1 import c1_step
from src.agent.stages.c2 import c2_step

__all__ = ["c0_step", "c1_step", "c2_step"]
