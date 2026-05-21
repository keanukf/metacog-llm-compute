"""
Pilot orchestration entrypoint.

`run_pilot_main` executes parsed pilot CLI arguments via an injected runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_pilot_main(args: Any, run_from_args: Callable[[Any], None]) -> None:
    """
    Run the pilot orchestration for already-parsed CLI args.

    The callable injection keeps this module importable without coupling it to
    script-level path/bootstrap concerns.
    """
    run_from_args(args)
