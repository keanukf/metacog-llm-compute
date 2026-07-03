"""
Logical Reasoning tasks — Extension. Stub only.

DEPRECATED — not part of the 2×3×2 design; kept for reference on branch
``legacy/deprecated-environments`` only.
"""

from __future__ import annotations

from typing import Any


def generate_logic_tasks(n: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Generate n logic puzzle instances. Extension stub."""
    return [{"id": f"logic_{i}"} for i in range(n)]
