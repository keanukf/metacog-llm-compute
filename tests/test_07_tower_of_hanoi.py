"""
Pilot Test 7 — Tower of Hanoi environment.
Tests instance generation, env interface, correctness tracking, parsing, and terminal conditions.
"""
from __future__ import annotations

from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances


def _task(num_disks: int = 3) -> dict:
    return generate_instances(1, seed=7, num_disks_range=(num_disks, num_disks), partial_start_range=(0, 0))[0]


def test_generate_instances_count():
    instances = generate_instances(10, seed=1)
    assert len(instances) == 10


def test_generate_instances_deterministic():
    a = generate_instances(5, seed=42)
    b = generate_instances(5, seed=42)
    assert a == b


def test_generate_instances_schema():
    inst = generate_instances(1, seed=0)[0]
    for key in (
        "id",
        "num_disks",
        "initial_state",
        "goal_state",
        "optimal_solution",
        "optimal_steps",
        "max_steps",
        "partial_start_moves",
    ):
        assert key in inst


def test_env_reset_returns_observation():
    env = TowerOfHanoiEnv(task=_task(), max_steps=20)
    obs = env.reset()
    assert isinstance(obs, str)
    assert "Peg A" in obs
    assert "Valid moves:" not in obs


def test_env_reset_can_include_valid_moves_opt_in():
    env = TowerOfHanoiEnv(task=_task(), max_steps=20, include_valid_moves=True)
    obs = env.reset()
    assert "Valid moves:" in obs


def test_env_step_returns_observation():
    task = _task()
    env = TowerOfHanoiEnv(task=task, max_steps=20)
    env.reset()
    src, dst = task["optimal_solution"][0]
    obs = env.step(f"{src} to {dst}")
    assert isinstance(obs, str)
    assert len(obs) > 0


def test_env_optimal_move_tracked():
    task = _task()
    env = TowerOfHanoiEnv(task=task, max_steps=20)
    env.reset()
    src, dst = task["optimal_solution"][0]
    env.step(f"Move disk from {src} to {dst}")
    assert env.step_results[-1]["correctness"] == "optimal"
    assert isinstance(env.step_results[-1].get("optimal_moves_remaining"), int)


def test_env_legal_move_tracked():
    env = TowerOfHanoiEnv(task=_task(), max_steps=20)
    env.reset()
    env.step("A to B")
    assert env.step_results[-1]["correctness"] == "legal"


def test_env_illegal_move_tracked():
    env = TowerOfHanoiEnv(task=_task(), max_steps=20)
    env.reset()
    before = env.step_results[:]  # keep style similar with direct state check below
    state_before = env._state.copy()
    env.step("B to A")
    assert len(env.step_results) == len(before) + 1
    assert env.step_results[-1]["correctness"] == "illegal"
    assert env.step_results[-1]["state_before"] == env.step_results[-1]["state_after"]
    assert env._state == state_before


def test_env_done_on_goal():
    task = _task()
    env = TowerOfHanoiEnv(task=task, max_steps=100)
    env.reset()
    for src, dst in list(task["optimal_solution"]):
        env.step(f"{src} -> {dst}")
        if env.done:
            break
    assert env.done is True
    assert env.task_success is True


def test_env_done_on_max_steps():
    env = TowerOfHanoiEnv(task=_task(), max_steps=1)
    env.reset()
    env.step("invalid")
    assert env.done is True
    assert env.task_success is False


def test_action_parsing_formats():
    env = TowerOfHanoiEnv(task=_task(), max_steps=20)
    formats = [
        "Move disk from A to C",
        "A to C",
        "A -> C",
        "A→C",
        "A C",
        "a c",
        "move A C",
    ]
    for action in formats:
        env.reset()
        env.step(action)
        assert env.step_results[-1]["action_parsed"] == ("A", "C")


def test_difficulty_scaling():
    easy = generate_instances(1, seed=5, num_disks_range=(3, 3), partial_start_range=(0, 0))[0]
    hard = generate_instances(1, seed=5, num_disks_range=(4, 4), partial_start_range=(0, 0))[0]
    assert easy["optimal_steps"] < hard["optimal_steps"]
