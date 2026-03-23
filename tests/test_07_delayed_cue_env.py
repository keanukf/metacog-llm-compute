"""
Pilot Test 7 — Delayed-Cue Recall environment.
Tests generate_tasks, DelayedCueEnv interface, correctness, temporal bins, task_success.
"""
from __future__ import annotations

from src.agent.base_agent import run_episode
from src.environments.delayed_cue import (
    DelayedCueEnv,
    answer_matches_expected,
    generate_tasks,
)


def _minimal_task(
    *,
    num_distractors: int = 2,
    recall_answer: str = "Paris",
    recall_parts: list[str] | None = None,
) -> dict:
    d = []
    for i in range(num_distractors):
        d.append({"question": f"Distractor Q{i}?", "answer": str(10 + i)})
    t: dict = {
        "id": "test_task",
        "seed": 0,
        "critical_fact": {"type": "association", "subject": "Alice", "value": recall_answer},
        "encoding_prompt": "Remember: Alice lives in Paris.",
        "distractors": d,
        "recall_cue": "Which city?",
        "recall_answer": recall_answer,
        "num_distractors": num_distractors,
        "complexity": "low",
    }
    if recall_parts is not None:
        t["recall_expected_parts"] = recall_parts
    return t


def test_generate_tasks_returns_correct_count():
    tasks = generate_tasks(10, seed=1)
    assert len(tasks) == 10


def test_generate_tasks_deterministic_with_seed():
    a = generate_tasks(5, seed=99)
    b = generate_tasks(5, seed=99)
    assert a == b


def test_generate_tasks_different_seeds():
    a = generate_tasks(3, seed=1)
    b = generate_tasks(3, seed=2)
    assert a != b


def test_task_instance_schema():
    t = generate_tasks(1, seed=0)[0]
    for key in (
        "id",
        "seed",
        "critical_fact",
        "encoding_prompt",
        "distractors",
        "recall_cue",
        "recall_answer",
        "num_distractors",
        "complexity",
    ):
        assert key in t
    assert len(t["distractors"]) == t["num_distractors"]


def test_env_reset_returns_encoding_observation():
    env = DelayedCueEnv(_minimal_task())
    obs = env.reset()
    assert isinstance(obs, str)
    assert "Paris" in obs
    assert env.observation == obs


def test_env_step_returns_next_observation():
    env = DelayedCueEnv(_minimal_task(num_distractors=2))
    env.reset()
    obs = env.step("ok")
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert env.observation == obs


def test_env_done_after_all_steps():
    env = DelayedCueEnv(_minimal_task(num_distractors=2))
    env.reset()
    assert not env.done
    env.step("ok")  # encoding
    assert not env.done
    env.step("10")  # distractor 0 -> answer 10
    assert not env.done
    env.step("11")  # distractor 1
    assert not env.done
    env.step("Paris")  # recall
    assert env.done


def test_env_not_done_mid_episode():
    env = DelayedCueEnv(_minimal_task(num_distractors=3))
    env.reset()
    env.step("x")
    env.step("10")
    assert not env.done


def test_step_correctness_tracking():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("ack")
    env.step("10")
    env.step("Paris")
    assert len(env.step_results) == 3
    assert env.step_results[0]["correct"] is True
    assert env.step_results[1]["correct"] is True
    assert env.step_results[2]["correct"] is True


def test_step_incorrect_answer():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("ack")
    env.step("wrong")
    assert env.step_results[1]["correct"] is False


def test_encoding_step_always_correct():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("nonsense xyz")
    assert env.step_results[0]["phase"] == "encoding"
    assert env.step_results[0]["correct"] is True


def test_binary_temporal_phases():
    env = DelayedCueEnv(_minimal_task(num_distractors=2))
    env.reset()
    env.step("a")
    env.step("10")
    env.step("11")
    env.step("Paris")
    assert env.step_results[0]["temporal_bin"] == "pre-distractor"
    for r in env.step_results[1:]:
        assert r["temporal_bin"] == "post-distractor"


def test_variable_distractor_count():
    tasks = generate_tasks(80, seed=7, num_distractors_range=(3, 8))
    counts = {t["num_distractors"] for t in tasks}
    assert len(counts) >= 2
    for t in tasks:
        assert 3 <= t["num_distractors"] <= 8


def test_recall_answer_matches_critical_fact():
    t = generate_tasks(50, seed=11, complexity="low")
    for task in t:
        cf = task["critical_fact"]
        if cf.get("type") == "association":
            assert task["recall_answer"] == cf["value"]


def test_task_success_correct_recall():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("ok")
    env.step("10")
    env.step("The city is Paris.")
    assert env.task_success is True


def test_task_success_incorrect_recall():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("ok")
    env.step("10")
    env.step("London")
    assert env.task_success is False


def test_done_after_recall_regardless():
    env = DelayedCueEnv(_minimal_task(num_distractors=1))
    env.reset()
    env.step("ok")
    env.step("10")
    env.step("WrongAnswer")
    assert env.done is True
    assert env.task_success is False


def test_negation_rejection():
    assert answer_matches_expected("not Paris", "Paris") is False
    assert answer_matches_expected("It is Paris.", "Paris") is True


def test_run_episode_attaches_step_correctness():
    task = _minimal_task(num_distractors=1)

    def step_fn(obs, history, model):
        if "Remember" in obs:
            return "ok", None, None, 0
        if "Distractor" in obs:
            return "10", None, None, 0
        return "Paris", None, None, 0

    env = DelayedCueEnv(task)
    out = run_episode(env, None, "C0", step_fn=step_fn, max_steps=20)
    assert "step_correctness" in out
    assert len(out["step_correctness"]) == 3
    assert out["task_success"] is True


def test_answer_matches_expected_empty_expected():
    assert answer_matches_expected("anything", "") is True
