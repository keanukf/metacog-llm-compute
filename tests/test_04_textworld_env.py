"""
Pilot Test 4 — TextWorld Mini-Environment.
Test env interface: reset(), step(), .observation, .done.
Uses stub TextWorldEnv so no TextWorld install required.
"""
from __future__ import annotations

import pytest

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
