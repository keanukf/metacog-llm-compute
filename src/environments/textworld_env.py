"""
TextWorld wrapper: reset(), step(action), .observation, .done.
Optional dependency guard for 'textworld' so stub works without install.
When game_file is set and textworld is installed, loads and plays real games.

Per-step records in ``step_results`` mirror TowerOfHanoiEnv (``step_index``, ``action_raw``,
``action_parsed``, ``correctness``, ``state_before``, ``state_after``) plus optional
``reward``, ``score_*`` when the engine exposes them. TextWorld uses ``correctness`` in
``{"optimal", "legal", "illegal", "unlabeled"}`` with quest-distance progress via
TextWorld's ``policy_commands`` (quest solver rest plan, not walkthrough oracle):
- optimal: executable move that strictly reduces ``len(policy_commands)``
- legal: executable, quest distance unchanged or increased
- illegal: parser error or unrecognized action
- unlabeled: quest distance not computable (see ``label_reason``; excluded from calibration)

``score_progress_step`` (bool) records score increase as a descriptive side variable only.

Labeling edge cases (preregistered):
- Illegal detection uses the **pre-step** admissible-command cache: an action not in that
  cache is illegal. When the cache is unavailable or empty (``admissible_commands`` absent
  or ``[]`` in ``info``), illegal detection falls back to parser-feedback text heuristics
  (``_suggests_unknown_command``) **or** a negative step reward (``reward < 0.0``).
- **Synonym resolution (confidence-neutral):** ``look inventory`` / ``check inventory`` → ``inventory``
  when ``inventory`` is pre-step admissible. ``action_raw`` preserves model output; ``action_executed``
  is what the engine receives. No rewrite for non-admissible canonical forms or planning errors.
- Winning step: ``policy_commands == []`` with ``won=True`` is optimal when ``dist_after < dist_before``.
- ``policy_commands == []`` with ``not won`` → ``unlabeled`` / ``quest_distance_empty_unwon``.
- Missing ``policy_commands`` in info → ``unlabeled`` / ``quest_distance_unavailable``.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.utils.errors import EnvironmentError

try:
    import textworld  # noqa: F401

    TEXTWORLD_AVAILABLE = True
except ImportError:
    TEXTWORLD_AVAILABLE = False

# TextWorld/TatSu game logic parsing is not thread-safe; serialize env init + reset load.
_TEXTWORLD_LOAD_LOCK = threading.Lock()


@contextmanager
def _textworld_load_section() -> Iterator[None]:
    with _TEXTWORLD_LOAD_LOCK:
        yield


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


def _append_admissible_to_observation(observation: str, info: dict[str, Any]) -> str:
    """
    When TextWorld exposes ``admissible_commands`` in ``info``, append a single line so the
    policy can align with parser-legal actions (similar to Tower of Hanoi ``Valid moves``).
    """
    adm = info.get("admissible_commands")
    if adm is None:
        return observation
    try:
        cmds = [str(c).strip() for c in adm if str(c).strip()]
    except TypeError:
        return observation
    if not cmds:
        return observation
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    suffix = ", ".join(ordered)
    base = observation.rstrip()
    return f"{base}\n\nValid commands this turn: {suffix}"


def _normalize_command_key(cmd: str) -> str:
    return " ".join(cmd.strip().split())


# Confidence-neutral surface forms → canonical parser commands (TextWorld only).
# Rewrites apply only when the canonical form is pre-step admissible.
_TEXTWORLD_COMMAND_SYNONYMS: dict[str, str] = {
    "look inventory": "inventory",
    "check inventory": "inventory",
}


def _resolve_synonym_for_admissible(
    action: str,
    admissible: Any,
) -> tuple[str, str | None]:
    """
    Return (action_to_execute, canonical_if_synonym_applied).

    Maps model surface forms to parser-known commands only when the canonical
    command is legal in the pre-step state. Does not alter generative choice.
    """
    normalized = _normalize_command_key(action)
    if not normalized:
        return action, None
    canonical = _TEXTWORLD_COMMAND_SYNONYMS.get(normalized)
    if canonical is None:
        return action, None
    parsed_canon, ok = _action_in_admissible(canonical, admissible)
    if ok and parsed_canon:
        return parsed_canon, canonical
    return action, None


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


def _experiment_meta_sidecar_path(game_path: Path) -> Path:
    """Experiment metadata lives in ``{stem}.meta.json`` (not ``{stem}.json``, which is TextWorld's Game dump)."""
    return game_path.parent / f"{game_path.stem}.meta.json"


def _load_sidecar(game_file: str | None) -> dict[str, Any] | None:
    if not game_file:
        return None
    p = Path(game_file)
    meta = _experiment_meta_sidecar_path(p)
    if meta.exists():
        try:
            with open(meta) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    # Legacy: experiment sidecar used to overwrite ``{stem}.json`` (wrong — broke TextWorld's Game.load).
    legacy = p.with_suffix(".json")
    if legacy.exists():
        try:
            with open(legacy) as f:
                data = json.load(f)
            if isinstance(data, dict) and "generation_parameters" in data:
                return data
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def _quest_distance_from_info(info: dict[str, Any]) -> int | None:
    """Rest quest distance from TextWorld ``policy_commands`` (solver plan length)."""
    raw = info.get("policy_commands")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return len([str(x) for x in raw if str(x).strip()])


def _score_progress_step(
    score_before: float | None,
    score_after: float | None,
) -> bool:
    return score_before is not None and score_after is not None and score_after > score_before


def _classify_quest_correctness(
    *,
    dist_before: int | None,
    dist_after: int | None,
    won: bool,
) -> tuple[str, str | None]:
    """
    Distance-based step label using TextWorld quest solver rest plan.

    Optimal when dist strictly decreases (not necessarily by exactly 1).
    Empty policy after a win is the terminal optimal step (dist_before > dist_after).
    """
    if dist_before is None or dist_after is None:
        return "unlabeled", "quest_distance_unavailable"
    if dist_after == 0 and not won:
        return "unlabeled", "quest_distance_empty_unwon"
    if dist_after < dist_before:
        return "optimal", None
    return "legal", None


class TextWorldEnv:
    """
    Environment interface: reset() -> first observation, step(action) -> next observation.
    Attributes: observation (str), done (bool), step_results (list of per-step dicts).
    When game_file is set and textworld is available, uses textworld.gym; else stub.
    """

    def __init__(
        self,
        game_file: str | None = None,
        max_steps: int = 20,
        *,
        include_admissible_commands: bool = False,
    ) -> None:
        self._game_file = game_file
        self._max_steps = max_steps
        self._include_admissible_commands = bool(include_admissible_commands)
        self.observation = ""
        self.done = False
        self._step_count = 0
        self._gym_env = None
        self.step_results: list[dict[str, Any]] = []
        self.current_step = 0
        self.task_success = False
        self.task_lost = False
        self._last_score: float | None = None
        # Cache admissible commands from the *previous* state so we can assess whether
        # the submitted action was parser-legal at the time it was chosen.
        self._last_admissible: Any = None
        self._last_quest_distance: int | None = None
        self.sidecar_metadata: dict[str, Any] | None = _load_sidecar(game_file)
        self.walkthrough: list[str] = []
        if isinstance(self.sidecar_metadata, dict):
            wt = self.sidecar_metadata.get("walkthrough")
            if isinstance(wt, list):
                self.walkthrough = [str(x) for x in wt]
        self._use_real = TEXTWORLD_AVAILABLE and game_file is not None and Path(game_file).exists()
        if self._use_real:
            self._init_gym_env()

    def _init_gym_env(self) -> None:
        try:
            with _textworld_load_section():
                import textworld.gym

                try:
                    from textworld import EnvInfos

                    infos = EnvInfos(
                        admissible_commands=True,
                        feedback=True,
                        score=True,
                        max_score=True,
                        won=True,
                        lost=True,
                        policy_commands=True,
                        intermediate_reward=True,
                    )
                    env_id = textworld.gym.register_game(
                        self._game_file,
                        max_episode_steps=self._max_steps,
                        name=f"tw_{id(self)}",
                        request_infos=infos,
                    )
                except (ImportError, AttributeError, TypeError, ValueError):
                    env_id = textworld.gym.register_game(
                        self._game_file,
                        max_episode_steps=self._max_steps,
                        name=f"tw_{id(self)}",
                    )
                self._gym_env = textworld.gym.make(env_id)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._use_real = False
            self._gym_env = None
            self._init_error = EnvironmentError(f"Failed to initialize TextWorld gym env: {exc}")

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
        self.task_lost = False
        self._last_score = None
        self._last_admissible = None
        self._last_quest_distance = None
        if self._use_real and self._gym_env is not None:
            with _textworld_load_section():
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
            self._last_admissible = info.get("admissible_commands")
            self._last_quest_distance = _quest_distance_from_info(info)
            if self._include_admissible_commands:
                self.observation = _append_admissible_to_observation(self.observation, info)
            return self.observation
        self.observation = (
            "You are in a small room. Exits: north.\n\n"
            "(Enter one parser command per turn, e.g. go north.)"
        )
        return self.observation

    def step(self, action: str) -> str:
        """Apply action; return next observation."""
        state_before = self.observation

        if self._use_real and self._gym_env is not None:
            pre_admissible = self._last_admissible
            executed_action, synonym_canonical = _resolve_synonym_for_admissible(
                action, pre_admissible
            )
            result = self._gym_env.step(executed_action)
            obs, reward, done, info = _unpack_gym_step(result)
            self.observation = obs if isinstance(obs, str) else str(obs)
            if self._include_admissible_commands:
                self.observation = _append_admissible_to_observation(self.observation, info)
            self.done = done
            score_before = self._last_score
            score_after = self._parse_score(info)
            if score_after is None:
                score_after = score_before
            score_delta = (
                None if score_before is None or score_after is None else score_after - score_before
            )
            score_progress = _score_progress_step(score_before, score_after)
            dist_before = self._last_quest_distance
            dist_after = _quest_distance_from_info(info)
            admissible = pre_admissible
            use_admissible = admissible is not None
            if use_admissible:
                try:
                    use_admissible = len(admissible) > 0
                except TypeError:
                    use_admissible = True

            if use_admissible:
                parsed, ok = _action_in_admissible(executed_action, admissible)
                illegal = not ok
            else:
                parsed = (
                    _normalize_command_key(executed_action) if executed_action.strip() else None
                )
                illegal = _suggests_unknown_command(info, self.observation) or reward < 0.0

            won_now = info.get("won") is True or info.get("game_won") is True
            lost_now = info.get("lost") is True or info.get("game_lost") is True

            if illegal:
                correctness = "illegal"
                label_reason: str | None = None
            else:
                correctness, label_reason = _classify_quest_correctness(
                    dist_before=dist_before,
                    dist_after=dist_after,
                    won=won_now,
                )

            if won_now:
                self.task_success = True
            if lost_now:
                self.task_lost = True
            self._last_score = score_after
            self._last_admissible = info.get("admissible_commands")
            if dist_after is not None:
                self._last_quest_distance = dist_after

            rec: dict[str, Any] = {
                "step_index": self.current_step,
                "action_raw": action,
                "action_executed": executed_action,
                "action_parsed": parsed,
                "action_synonym_canonical": synonym_canonical,
                "correctness": correctness,
                "label_reason": label_reason,
                "quest_distance_before": dist_before,
                "quest_distance_after": dist_after,
                "optimal_moves_remaining": dist_after,
                "score_progress_step": score_progress,
                "state_before": state_before,
                "state_after": self.observation,
                "reward": reward,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": score_delta,
                "won": bool(won_now),
                "lost": bool(lost_now),
                "walkthrough_step_hint": (
                    self.walkthrough[self.current_step]
                    if self.current_step < len(self.walkthrough)
                    else None
                ),
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
            self.observation = (
                "You moved. Exits: north, south.\n\n"
                "(Enter one parser command per turn, e.g. go north.)"
            )
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
