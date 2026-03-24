"""
TextWorld wrapper: reset(), step(action), .observation, .done.
Optional dependency guard for 'textworld' so stub works without install.
When game_file is set and textworld is installed, loads and plays real games.

Per-step records in ``step_results`` mirror TowerOfHanoiEnv (``step_index``, ``action_raw``,
``action_parsed``, ``correctness``, ``state_before``, ``state_after``) plus optional
``reward``, ``score_*`` when the engine exposes them. TextWorld uses ``correctness`` in
``{"legal", "illegal"}`` only (no ``optimal`` without an oracle).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import textworld  # noqa: F401
    TEXTWORLD_AVAILABLE = True
except ImportError:
    TEXTWORLD_AVAILABLE = False

# Feedback substrings suggesting the parser rejected the command (best-effort fallback).
_UNKNOWN_CMD_HINTS = (
    "don't understand",
    "dont understand",
    "not sure what you mean",
    "nothing happens",
    "no such thing",
    "can't do that",
    "cannot do that",
    "invalid command",
    "unknown command",
)


def _unpack_gym_step(result: tuple[Any, ...]) -> tuple[Any, float, bool, dict[str, Any]]:
    """Normalize gym / gymnasium step() return to (obs, reward, done, info)."""
    info: dict[str, Any] = {}
    if len(result) == 5:
        obs, reward, terminated, truncated, raw_info = result
        done = bool(terminated) or bool(truncated)
        if isinstance(raw_info, dict):
            info = raw_info
        return obs, float(reward), done, info
    if len(result) >= 4:
        obs, reward, done, raw_info = result[0], result[1], result[2], result[3]
        if isinstance(raw_info, dict):
            info = raw_info
        return obs, float(reward), bool(done), info
    obs, reward, done = result[0], result[1], result[2]
    return obs, float(reward), bool(done), info


def _feedback_text(info: dict[str, Any], observation: str) -> str:
    fb = info.get("feedback")
    if isinstance(fb, str) and fb.strip():
        return fb.lower()
    return observation.lower()


def _suggests_unknown_command(info: dict[str, Any], observation: str) -> bool:
    text = _feedback_text(info, observation)
    return any(h in text for h in _UNKNOWN_CMD_HINTS)


def _normalize_command_key(cmd: str) -> str:
    return " ".join(cmd.strip().split())


def _action_in_admissible(action: str, admissible: Any) -> tuple[str | None, bool]:
    """Return (parsed_stripped_or_none, is_legal)."""
    parsed = _normalize_command_key(action) if action.strip() else None
    if parsed is None:
        return None, False
    if admissible is None:
        return parsed, False
    try:
        candidates = list(admissible)
    except TypeError:
        candidates = [admissible]
    normalized = {_normalize_command_key(str(c)) for c in candidates}
    if parsed in normalized:
        return parsed, True
    return parsed, False


def _infer_correctness_real(
    reward: float,
    score_before: float | None,
    score_after: float | None,
    info: dict[str, Any],
    observation: str,
) -> str:
    if admissible is not None:
        _, ok = _action_in_admissible(action, admissible)
        return "legal" if ok else "illegal"
    if reward < 0:
        return "illegal"
    if _suggests_unknown_command(info, observation):
        return "illegal"
    inter = info.get("intermediate_reward")
    try:
        if inter is not None and float(inter) > 0:
            return "legal"
    except (TypeError, ValueError):
        pass
    if score_before is not None and score_after is not None and score_after > score_before:
        return "legal"
    if reward > 0:
        return "legal"
    # Ambiguous (many valid moves yield zero reward); default legal.
    return "legal"


class TextWorldEnv:
    """
    Environment interface: reset() -> first observation, step(action) -> next observation.
    Attributes: observation (str), done (bool), step_results (list of per-step dicts).
    When game_file is set and textworld is available, uses textworld.gym; else stub.
    """

    def __init__(self, game_file: str | None = None, max_steps: int = 20) -> None:
        self._game_file = game_file
        self._max_steps = max_steps
        self.observation = ""
        self.done = False
        self._step_count = 0
        self._gym_env = None
        self.step_results: list[dict[str, Any]] = []
        self.current_step = 0
        self.task_success = False
        self._last_score: float | None = None
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
            try:
                from textworld import EnvInfos

                infos = EnvInfos(admissible_commands=True, feedback=True, score=True)
                env_id = textworld.gym.register_game(
                    self._game_file,
                    max_episode_steps=self._max_steps,
                    name=f"tw_{id(self)}",
                    request_infos=infos,
                )
            except Exception:
                env_id = textworld.gym.register_game(
                    self._game_file,
                    max_episode_steps=self._max_steps,
                    name=f"tw_{id(self)}",
                )
            self._gym_env = gym.make(env_id)
        except Exception:
            self._use_real = False
            self._gym_env = None

    def _parse_score(self, info: dict[str, Any]) -> float | None:
        if not info:
            return None
        s = info.get("score")
        if s is None:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    def reset(self) -> str:
        """Reset environment; return initial observation."""
        self.done = False
        self._step_count = 0
        self.current_step = 0
        self.step_results = []
        self.task_success = False
        self._last_score = None
        if self._use_real and self._gym_env is not None:
            result = self._gym_env.reset()
            info: dict[str, Any] = {}
            if isinstance(result, tuple) and len(result) == 2:
                obs, raw_info = result
                if isinstance(raw_info, dict):
                    info = raw_info
            else:
                obs = result
            self.observation = obs if isinstance(obs, str) else str(obs)
            self._last_score = self._parse_score(info)
            return self.observation
        self.observation = "You are in a small room. Exits: north."
        return self.observation

    def step(self, action: str) -> str:
        """Apply action; return next observation."""
        state_before = self.observation

        if self._use_real and self._gym_env is not None:
            result = self._gym_env.step(action)
            obs, reward, done, info = _unpack_gym_step(result)
            self.observation = obs if isinstance(obs, str) else str(obs)
            self.done = done
            score_before = self._last_score
            score_after = self._parse_score(info)
            if score_after is None:
                score_after = score_before
            score_delta = (
                None
                if score_before is None or score_after is None
                else score_after - score_before
            )
            admissible = info.get("admissible_commands")
            use_admissible = admissible is not None
            if use_admissible:
                try:
                    use_admissible = len(admissible) > 0  # type: ignore[arg-type]
                except TypeError:
                    use_admissible = True

            if use_admissible:
                parsed, ok = _action_in_admissible(action, admissible)
                correctness = "legal" if ok else "illegal"
            else:
                parsed = _normalize_command_key(action) if action.strip() else None
                correctness = _infer_correctness_real(
                    reward,
                    score_before,
                    score_after,
                    info,
                    self.observation,
                )

            if info.get("won") is True or info.get("game_won") is True:
                self.task_success = True
            self._last_score = score_after

            rec: dict[str, Any] = {
                "step_index": self.current_step,
                "action_raw": action,
                "action_parsed": parsed,
                "correctness": correctness,
                "state_before": state_before,
                "state_after": self.observation,
                "reward": reward,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": score_delta,
            }
            self.step_results.append(rec)
            self.current_step += 1
            return self.observation

        # Stub: non-empty action counts as legal.
        correctness = "legal" if action.strip() else "illegal"
        parsed = _normalize_command_key(action) if action.strip() else None
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self.done = True
            self.observation = "Max steps reached."
        else:
            self.observation = "You moved. Exits: north, south."
        self.step_results.append(
            {
                "step_index": self.current_step,
                "action_raw": action,
                "action_parsed": parsed,
                "correctness": correctness,
                "state_before": state_before,
                "state_after": self.observation,
                "reward": 0.0,
                "score_before": None,
                "score_after": None,
                "score_delta": None,
            }
        )
        self.current_step += 1
        return self.observation


def make_textworld_env(**kwargs: Any) -> TextWorldEnv:
    """Factory for TextWorld env. When textworld is installed and game_file set, loads real game."""
    return TextWorldEnv(**kwargs)
