"""
Pilot Test 4 — TextWorld Mini-Environment.
Test env interface: reset(), step(), .observation, .done.
Uses stub TextWorldEnv so no TextWorld install required.
"""
from __future__ import annotations

from src.environments.textworld_env import TextWorldEnv


def test_textworld_env_reset_returns_observation():
    env = TextWorldEnv(max_steps=5)
    obs = env.reset()
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert env.observation == obs


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

    out = _append_admissible_to_observation("A room.", {"admissible_commands": ["go north", "look"]})
    assert out.startswith("A room.")
    assert "Valid commands this turn:" in out
    assert "go north" in out and "look" in out

    assert _append_admissible_to_observation("x", {}) == "x"
    assert _append_admissible_to_observation("x", {"admissible_commands": []}) == "x"
