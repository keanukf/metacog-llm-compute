"""
Pilot orchestration package.

Phase-2 refactor target: move run logic out of scripts into importable modules.
"""

from src.pilot.orchestrator import run_pilot_main

__all__ = ["run_pilot_main"]
