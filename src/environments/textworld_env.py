"""
TextWorld wrapper: reset(), step(action), .observation, .done.
Optional dependency guard for 'textworld' so stub works without install.
"""
from __future__ import annotations

from typing import Any

try:
    import textworld  # noqa: F401
    TEXTWORLD_AVAILABLE = True
except ImportError:
    TEXTWORLD_AVAILABLE = False


class TextWorldEnv:
    """
    Environment interface: reset() -> first observation, step(action) -> next observation.
    Attributes: observation (str), done (bool).
    """

    def __init__(self, game_file: str | None = None, max_steps: int = 20) -> None:
        self._game_file = game_file
        self._max_steps = max_steps
        self.observation = ""
        self.done = False
        self._step_count = 0

    def reset(self) -> str:
        """Reset environment; return initial observation."""
        self.done = False
        self._step_count = 0
        self.observation = "You are in a small room. Exits: north."
        return self.observation

    def step(self, action: str) -> str:
        """Apply action; return next observation."""
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self.done = True
            self.observation = "Max steps reached."
            return self.observation
        self.observation = "You moved. Exits: north, south."
        return self.observation


def make_textworld_env(**kwargs: Any) -> TextWorldEnv:
    """Factory for TextWorld env. When textworld is installed, can load a real game."""
    return TextWorldEnv(**kwargs)
