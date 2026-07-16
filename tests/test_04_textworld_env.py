"""
Pilot Test 4 — TextWorld Mini-Environment.
Test env interface: reset(), step(), .observation, .done.
Uses stub TextWorldEnv so no TextWorld install required.
"""

from __future__ import annotations

import threading

from src.environments import textworld_env
from src.environments.textworld_env import TextWorldEnv


def test_textworld_load_lock_exists_for_parallel_init():
    assert isinstance(textworld_env._TEXTWORLD_LOAD_LOCK, type(threading.Lock()))


def test_textworld_env_reset_returns_observation():
    env = TextWorldEnv(max_steps=5)
    obs = env.reset()
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert env.observation == obs
    assert "Valid commands this turn:" not in obs


def test_textworld_env_step_returns_next_observation():
    env = TextWorldEnv(max_steps=5)
    env.reset()
    obs = env.step("go north")
    assert isinstance(obs, str)
    assert env.observation == obs


def test_textworld_env_done_after_max_steps():
    env = TextWorldEnv(max_steps=2)
    env.reset()
    env.step("go north")
    assert not env.done
    env.step("go south")
    assert env.done


def test_textworld_env_interface():
    """Agent loop can use reset() and step(action) and read .observation, .done."""
    env = TextWorldEnv(max_steps=3)
    obs = env.reset()
    assert obs
    obs = env.step("go north")
    assert obs
    obs = env.step("go south")
    assert env.done or obs


def test_textworld_stub_step_results_cleared_on_reset():
    env = TextWorldEnv(max_steps=5)
    env.reset()
    env.step("go north")
    assert len(env.step_results) == 1
    env.reset()
    assert env.step_results == []


def test_textworld_stub_step_results_non_empty_action_legal():
    env = TextWorldEnv(max_steps=5)
    env.reset()
    env.step("  go north  ")
    assert env.step_results[-1]["correctness"] == "legal"
    assert env.step_results[-1]["action_parsed"] == "go north"
    assert "state_before" in env.step_results[-1]
    assert "state_after" in env.step_results[-1]


def test_textworld_stub_empty_action_illegal():
    env = TextWorldEnv(max_steps=5)
    env.reset()
    env.step("")
    assert env.step_results[-1]["correctness"] == "illegal"
    assert env.step_results[-1]["action_parsed"] is None


def test_textworld_stub_step_index_increments():
    env = TextWorldEnv(max_steps=5)
    env.reset()
    env.step("a")
    env.step("b")
    assert env.step_results[0]["step_index"] == 0
    assert env.step_results[1]["step_index"] == 1


def test_unpack_gym_step_five_tuple():
    from src.environments.textworld_env import _unpack_gym_step

    obs, r, done, info = _unpack_gym_step(("x", 0.5, False, True, {"score": 1}))
    assert obs == "x"
    assert r == 0.5
    assert done is True
    assert info["score"] == 1


def test_action_in_admissible():
    from src.environments.textworld_env import _action_in_admissible

    parsed, ok = _action_in_admissible("  go north  ", ["go north", "look"])
    assert ok
    assert parsed == "go north"
    _, ok2 = _action_in_admissible("take apple", ["go north"])
    assert not ok2


def test_append_admissible_to_observation():
    from src.environments.textworld_env import _append_admissible_to_observation

    out = _append_admissible_to_observation(
        "A room.", {"admissible_commands": ["go north", "look"]}
    )
    assert out.startswith("A room.")
    assert "Valid commands this turn:" in out
    assert "go north" in out and "look" in out

    assert _append_admissible_to_observation("x", {}) == "x"
    assert _append_admissible_to_observation("x", {"admissible_commands": []}) == "x"


def test_textworld_env_opt_in_includes_admissible_in_observation():
    class _FakeGymEnv:
        def __init__(self) -> None:
            self._reset_result: tuple[str, dict] | str = (
                "reset",
                {"admissible_commands": ["go north"]},
            )
            self._step_result: tuple[str, float, bool, dict] = (
                "step",
                0.0,
                False,
                {"admissible_commands": ["look"]},
            )

        def reset(self):
            return self._reset_result

        def step(self, action: str):
            return self._step_result

    fake = _FakeGymEnv()
    env = TextWorldEnv(game_file=None, max_steps=5, include_admissible_commands=True)
    env._use_real = True
    env._gym_env = fake
    obs = env.reset()
    assert "Valid commands this turn:" in obs


class _FakeGymEnv:
    def __init__(self) -> None:
        self._reset_result: tuple[str, dict] | str = ("reset", {})
        self._step_result: tuple[str, float, bool, dict] = ("step", 0.0, False, {})
        self.last_action: str | None = None

    def reset(self):
        return self._reset_result

    def step(self, action: str):
        self.last_action = action
        return self._step_result


def _make_realish_env(fake: _FakeGymEnv) -> TextWorldEnv:
    env = TextWorldEnv(game_file=None, max_steps=5)
    env._use_real = True
    env._gym_env = fake
    return env


def test_correctness_uses_pre_step_admissible():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {
            "admissible_commands": ["eat lettuce", "look"],
            "policy_commands": ["go north"],
            "score": 0,
        },
    )
    fake._step_result = (
        "*** You lost! ***",
        0.0,
        True,
        {"admissible_commands": ["look"], "policy_commands": ["go north"], "score": 0},
    )

    env = _make_realish_env(fake)
    env.reset()
    env.step("eat lettuce")
    assert env.step_results[-1]["correctness"] == "legal"


def test_info_lost_sets_task_lost_and_step_record():
    fake = _FakeGymEnv()
    fake._reset_result = ("reset", {"admissible_commands": ["eat lettuce"], "score": 0})
    fake._step_result = (
        "*** You lost! ***",
        0.0,
        True,
        {"lost": True, "admissible_commands": [], "score": 0},
    )

    env = _make_realish_env(fake)
    env.reset()
    env.step("eat lettuce")
    assert env.task_lost is True
    assert env.task_success is False
    assert env.step_results[-1]["lost"] is True
    assert env.step_results[-1]["won"] is False


def test_info_won_sets_task_success_and_step_record():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["eat meal"], "policy_commands": ["eat meal"], "score": 0},
    )
    fake._step_result = (
        "*** You won! ***",
        1.0,
        True,
        {"won": True, "admissible_commands": [], "policy_commands": [], "score": 1},
    )

    env = _make_realish_env(fake)
    env.reset()
    env.step("eat meal")
    assert env.task_success is True
    assert env.task_lost is False
    assert env.step_results[-1]["won"] is True
    assert env.step_results[-1]["lost"] is False


def test_synonym_look_inventory_when_admissible():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {
            "admissible_commands": ["inventory", "go north"],
            "policy_commands": ["inventory"],
            "score": 0,
        },
    )
    fake._step_result = (
        "You are carrying nothing.",
        0.0,
        False,
        {"admissible_commands": ["go north"], "policy_commands": ["go north"], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("look inventory")
    rec = env.step_results[-1]
    assert fake.last_action == "inventory"
    assert rec["action_raw"] == "look inventory"
    assert rec["action_executed"] == "inventory"
    assert rec["action_synonym_canonical"] == "inventory"
    assert rec["correctness"] == "legal"


def test_synonym_not_applied_when_canonical_not_admissible():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["go north"], "policy_commands": ["go north"], "score": 0},
    )
    fake._step_result = (
        "You can't see any such thing.",
        0.0,
        False,
        {"admissible_commands": ["go north"], "policy_commands": ["go north"], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("look inventory")
    rec = env.step_results[-1]
    assert fake.last_action == "look inventory"
    assert rec["action_executed"] == "look inventory"
    assert rec["action_synonym_canonical"] is None
    assert rec["correctness"] == "illegal"


def test_hallucinated_fry_not_remapped():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {
            "admissible_commands": ["prepare meal", "inventory"],
            "policy_commands": ["prepare meal"],
            "score": 0,
        },
    )
    fake._step_result = (
        "Nothing happens.",
        0.0,
        False,
        {"admissible_commands": ["prepare meal"], "policy_commands": ["prepare meal"], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("fry yellow onion with stove")
    rec = env.step_results[-1]
    assert fake.last_action == "fry yellow onion with stove"
    assert rec["action_synonym_canonical"] is None
    assert rec["correctness"] == "illegal"
