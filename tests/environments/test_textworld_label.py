"""TextWorld step labeling by quest-distance reduction (the DV).

Verifies a step is optimal only on a strict quest-distance reduction, legal when distance holds or
increases, winning steps count as optimal, and that an empty policy or missing distance stays
explicitly unlabeled rather than silently defaulting to legal. This label IS the dependent
variable for TextWorld, so the conservative unlabeled fallback is a DV-protection measure: a
spuriously-labeled step would inject noise straight into every RQ1/RQ2 result.
"""

from __future__ import annotations

from src.environments.textworld_env import (
    TextWorldEnv,
    _classify_quest_correctness,
    _quest_distance_from_info,
)


class _FakeGymEnv:
    def __init__(self) -> None:
        self._reset_result: tuple[str, dict] | str = ("reset", {})
        self._step_result: tuple[str, float, bool, dict] = ("step", 0.0, False, {})

    def reset(self):
        return self._reset_result

    def step(self, action: str):
        return self._step_result


def _make_realish_env(fake: _FakeGymEnv) -> TextWorldEnv:
    env = TextWorldEnv(game_file=None, max_steps=20)
    env._use_real = True
    env._gym_env = fake
    return env


def test_classify_optimal_on_strict_distance_reduction():
    assert _classify_quest_correctness(dist_before=4, dist_after=3, won=False) == ("optimal", None)


def test_classify_optimal_when_distance_drops_by_two():
    """Correction 1: any strict decrease counts, not only -1."""
    assert _classify_quest_correctness(dist_before=5, dist_after=3, won=False) == ("optimal", None)


def test_classify_legal_when_distance_unchanged():
    assert _classify_quest_correctness(dist_before=4, dist_after=4, won=False) == ("legal", None)


def test_classify_legal_when_distance_increases():
    assert _classify_quest_correctness(dist_before=4, dist_after=5, won=False) == ("legal", None)


def test_classify_winning_step_optimal():
    """Correction 2: terminal win step (1->0, won) is optimal, not unlabeled."""
    assert _classify_quest_correctness(dist_before=1, dist_after=0, won=True) == ("optimal", None)


def test_classify_empty_policy_unwon_is_unlabeled():
    assert _classify_quest_correctness(dist_before=2, dist_after=0, won=False) == (
        "unlabeled",
        "quest_distance_empty_unwon",
    )


def test_classify_missing_distance_is_unlabeled():
    assert _classify_quest_correctness(dist_before=None, dist_after=3, won=False) == (
        "unlabeled",
        "quest_distance_unavailable",
    )


def test_quest_distance_from_info():
    assert _quest_distance_from_info({"policy_commands": ["a", "b"]}) == 2
    assert _quest_distance_from_info({"policy_commands": []}) == 0
    assert _quest_distance_from_info({}) is None


def test_optimal_move_reduces_quest_distance():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {
            "admissible_commands": ["go north"],
            "policy_commands": ["go north", "cook apple with stove"],
            "score": 0,
        },
    )
    fake._step_result = (
        "moved",
        0.0,
        False,
        {
            "admissible_commands": ["go east"],
            "policy_commands": ["cook apple with stove"],
            "score": 0,
        },
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("go north")
    rec = env.step_results[-1]
    assert rec["correctness"] == "optimal"
    assert rec["quest_distance_before"] == 2
    assert rec["quest_distance_after"] == 1
    assert rec["score_progress_step"] is False


def test_legal_move_holds_quest_distance():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["look"], "policy_commands": ["go north"], "score": 0},
    )
    fake._step_result = (
        "looked",
        0.0,
        False,
        {"admissible_commands": ["look", "go north"], "policy_commands": ["go north"], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("look")
    rec = env.step_results[-1]
    assert rec["correctness"] == "legal"
    assert rec["quest_distance_before"] == 1
    assert rec["quest_distance_after"] == 1


def test_illegal_move():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["go north"], "policy_commands": ["go north"], "score": 0},
    )
    fake._step_result = (
        "unknown",
        0.0,
        False,
        {"admissible_commands": ["go north"], "policy_commands": ["go north"], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("xyzzy")
    rec = env.step_results[-1]
    assert rec["correctness"] == "illegal"
    assert rec.get("label_reason") is None


def test_winning_step_labeled_optimal():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["eat meal"], "policy_commands": ["eat meal"], "score": 3},
    )
    fake._step_result = (
        "*** You won! ***",
        1.0,
        True,
        {"won": True, "admissible_commands": [], "policy_commands": [], "score": 4},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("eat meal")
    rec = env.step_results[-1]
    assert rec["correctness"] == "optimal"
    assert rec["quest_distance_before"] == 1
    assert rec["quest_distance_after"] == 0
    assert rec["won"] is True
    assert rec.get("label_reason") is None


def test_empty_policy_unwon_not_silent_legal():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {"admissible_commands": ["look"], "policy_commands": ["go north"], "score": 0},
    )
    fake._step_result = (
        "stuck",
        0.0,
        False,
        {"admissible_commands": ["look"], "policy_commands": [], "score": 0},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("look")
    rec = env.step_results[-1]
    assert rec["correctness"] == "unlabeled"
    assert rec["label_reason"] == "quest_distance_empty_unwon"


def test_score_progress_step_side_variable():
    fake = _FakeGymEnv()
    fake._reset_result = (
        "reset",
        {
            "admissible_commands": ["cook x with stove"],
            "policy_commands": ["cook x with stove", "eat meal"],
            "score": 0,
        },
    )
    fake._step_result = (
        "cooked",
        0.0,
        False,
        {"admissible_commands": ["eat meal"], "policy_commands": ["eat meal"], "score": 1},
    )
    env = _make_realish_env(fake)
    env.reset()
    env.step("cook x with stove")
    rec = env.step_results[-1]
    assert rec["score_progress_step"] is True
    assert rec["correctness"] == "optimal"
