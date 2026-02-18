"""
TextWorld wrapper: reset(), step(action), .observation, .done.
Optional dependency guard for 'textworld' so stub works without install.
When game_file is set and textworld is installed, loads and plays real games.
"""
from __future__ import annotations

from pathlib import Path
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
    When game_file is set and textworld is available, uses textworld.gym; else stub.
    """

    def __init__(self, game_file: str | None = None, max_steps: int = 20) -> None:
        self._game_file = game_file
        self._max_steps = max_steps
        self.observation = ""
        self.done = False
        self._step_count = 0
        self._gym_env = None
        self._use_real = (
            TEXTWORLD_AVAILABLE
            and game_file is not None
            and Path(game_file).exists()
        )
        if self._use_real:
            self._init_gym_env()

    def _init_gym_env(self) -> None:
        try:
            import gym  # noqa: F401
            import textworld.gym
            env_id = textworld.gym.register_game(
                self._game_file,
                max_episode_steps=self._max_steps,
                name=f"tw_{id(self)}",
            )
            self._gym_env = gym.make(env_id)
        except Exception:
            self._use_real = False
            self._gym_env = None

    def reset(self) -> str:
        """Reset environment; return initial observation."""
        self.done = False
        self._step_count = 0
        if self._use_real and self._gym_env is not None:
            result = self._gym_env.reset()
            if isinstance(result, (list, tuple)):
                obs = result[0] if result else ""
            else:
                obs = result
            self.observation = obs if isinstance(obs, str) else str(obs)
            return self.observation
        self.observation = "You are in a small room. Exits: north."
        return self.observation

    def step(self, action: str) -> str:
        """Apply action; return next observation."""
        if self._use_real and self._gym_env is not None:
            result = self._gym_env.step(action)
            # step returns (obs, reward, done, info) or (obs, reward, terminated, truncated, info) in newer gym
            obs = result[0]
            self.observation = obs if isinstance(obs, str) else str(obs)
            if len(result) >= 4:
                self.done = bool(result[2]) or bool(result[3])  # terminated or truncated
            elif len(result) >= 3:
                self.done = bool(result[2])
            return self.observation
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self.done = True
            self.observation = "Max steps reached."
            return self.observation
        self.observation = "You moved. Exits: north, south."
        return self.observation


def make_textworld_env(**kwargs: Any) -> TextWorldEnv:
    """Factory for TextWorld env. When textworld is installed and game_file set, loads real game."""
    return TextWorldEnv(**kwargs)
