"""Tower of Hanoi optimal labeling (distance criterion)."""

from __future__ import annotations

from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances


def test_optimal_move_reduces_distance_by_one():
    task = generate_instances(1, seed=99, num_disks_range=(3, 3), partial_start_range=(0, 0))[0]
    env = TowerOfHanoiEnv(task, max_steps=20)
    env.reset()
    # Take first optimal move from BFS path
    from src.environments.tower_of_hanoi import _format_move, _shortest_path_to_goal

    mv = _shortest_path_to_goal(env._state, env._num_disks)[0]
    env.step(_format_move(mv))
    assert env.step_results[-1]["correctness"] == "optimal"
    rem = env.step_results[-1]["optimal_moves_remaining"]
    before_rem = len(_shortest_path_to_goal(task["initial_state"], task["num_disks"]))
    assert before_rem - rem == 1
