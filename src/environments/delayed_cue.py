"""
Delayed-Cue Recall: encoding → N distractors → recall (delayed match-to-sample).
Same surface as TextWorldEnv: reset(), step(action), .observation, .done.
"""
from __future__ import annotations

import random
from typing import Any, Literal

Complexity = Literal["low", "medium", "high"]

# Words that immediately before the expected span indicate a negated answer (methodological heuristic).
_NEGATION_LAST_WORDS = frozenset(
    {"not", "no", "never", "neither", "nor", "without"}
)


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def answer_matches_expected(agent_response: str, expected: str) -> bool:
    """
    True if normalized expected is contained in normalized response and no negation
    token immediately precedes the match. Encoding steps use expected="" and always match elsewhere.
    """
    if not expected:
        return True
    a = _normalize_text(agent_response)
    e = _normalize_text(expected)
    if not e:
        return True
    idx = a.find(e)
    if idx < 0:
        return False
    before = a[:idx].rstrip()
    if not before:
        return True
    last_token = before.split()[-1]
    if last_token in _NEGATION_LAST_WORDS:
        return False
    if last_token.endswith("n't") or last_token == "don't" or last_token == "doesn't" or last_token == "isn't":
        return False
    return True


# --- Critical fact templates (content generation) ---

_ASSOCIATION_SUBJECTS = [
    "Alice",
    "Bob",
    "Chen",
    "Dana",
    "Elena",
    "Frank",
    "Gina",
    "Hassan",
    "Iris",
    "Jon",
]

_ASSOCIATION_PLACES = [
    "Paris",
    "Berlin",
    "Oslo",
    "Lima",
    "Tokyo",
    "Delhi",
    "Cairo",
    "Perth",
    "Dublin",
    "Quebec",
]

_CODES = ["7421", "9038", "5510", "2284", "6619", "4407", "1192", "8830"]

_KEYS = ["silver", "brass", "iron", "oak", "jade", "ivory", "copper", "marble"]


def _build_association_fact(rng: random.Random, complexity: Complexity) -> dict[str, Any]:
    subject = rng.choice(_ASSOCIATION_SUBJECTS)
    place = rng.choice(_ASSOCIATION_PLACES)
    encoding = f"Remember this for later: {subject} lives in {place}."
    recall_cue = f"In which city does {subject} live?"
    return {
        "critical_fact": {"type": "association", "subject": subject, "attribute": "city", "value": place},
        "encoding_prompt": encoding,
        "recall_cue": recall_cue,
        "recall_answer": place,
    }


def _build_numeric_fact(rng: random.Random, complexity: Complexity) -> dict[str, Any]:
    code = rng.choice(_CODES)
    encoding = f"Remember this security code exactly: {code}. You will need it later."
    recall_cue = "What was the security code you memorized?"
    return {
        "critical_fact": {"type": "numeric", "value": code},
        "encoding_prompt": encoding,
        "recall_cue": recall_cue,
        "recall_answer": code,
    }


def _build_key_fact(rng: random.Random, complexity: Complexity) -> dict[str, Any]:
    key = rng.choice(_KEYS)
    encoding = f"Remember: the safe key is made of {key}."
    recall_cue = "What material is the safe key made of?"
    return {
        "critical_fact": {"type": "key", "value": key},
        "encoding_prompt": encoding,
        "recall_cue": recall_cue,
        "recall_answer": key,
    }


def _build_multi_fact(rng: random.Random) -> dict[str, Any]:
    s1, s2 = rng.sample(_ASSOCIATION_SUBJECTS, 2)
    p1, p2 = rng.sample(_ASSOCIATION_PLACES, 2)
    encoding = f"Remember both: {s1} lives in {p1}. {s2} lives in {p2}."
    recall_cue = (
        f"Where does {s1} live, and where does {s2} live? "
        "Answer with both cities (order does not matter)."
    )
    recall_answer = f"{p1} and {p2}"
    return {
        "critical_fact": {"type": "multi", "pairs": [(s1, p1), (s2, p2)]},
        "encoding_prompt": encoding,
        "recall_cue": recall_cue,
        "recall_answer": recall_answer,
        "recall_expected_parts": [p1, p2],
    }


def _sample_critical_fact(rng: random.Random, complexity: Complexity) -> dict[str, Any]:
    if complexity == "high":
        return _build_multi_fact(rng)
    if complexity == "medium":
        choice = rng.choice(["association", "numeric", "key"])
        if choice == "association":
            return _build_association_fact(rng, complexity)
        if choice == "numeric":
            return _build_numeric_fact(rng, complexity)
        return _build_key_fact(rng, complexity)
    # low
    return _build_association_fact(rng, complexity)


# --- Distractors (question -> single canonical answer substring) ---

_ARITHMETIC = [
    ("What is 7 * 8?", "56"),
    ("What is 13 * 7?", "91"),
    ("What is 144 / 12?", "12"),
    ("What is 15 + 27?", "42"),
    ("What is 100 - 37?", "63"),
    ("What is 9 * 6?", "54"),
    ("What is 81 / 9?", "9"),
    ("What is 11 * 11?", "121"),
]

_CATEGORY = [
    ("Name a primary color; answer with one word.", "red"),
    ("Name a primary color; answer with one word.", "blue"),
    ("Name a primary color; answer with one word.", "yellow"),
    ("Name a citrus fruit; answer with one word.", "orange"),
    ("Name a citrus fruit; answer with one word.", "lemon"),
    ("What is the chemical symbol for water? Answer H2O or similar.", "h2o"),
    ("How many sides does a triangle have?", "3"),
    ("How many days are in a leap year February?", "29"),
]

_LOGIC = [
    (
        "If all cats are animals, and Felix is a cat, what is Felix? One word: animal or cat.",
        "cat",
    ),
    (
        "If it rains, the street is wet. The street is wet. Is it necessarily raining? Answer yes or no.",
        "no",
    ),
    (
        "What is the capital of France? One word.",
        "paris",
    ),
]


def _sample_distractors(rng: random.Random, n: int) -> list[dict[str, str]]:
    pool = list(_ARITHMETIC) + list(_CATEGORY) + list(_LOGIC)
    rng.shuffle(pool)
    out: list[dict[str, str]] = []
    for i in range(n):
        q, a = pool[i % len(pool)]
        out.append({"question": q, "answer": a})
    return out


def generate_tasks(
    n: int,
    seed: int | None = None,
    *,
    num_distractors_range: tuple[int, int] | list[int] = (3, 8),
    complexity: Complexity | str = "medium",
) -> list[dict[str, Any]]:
    """
    Generate n delayed-cue task instances (replayable dicts).

    Each instance samples its own num_distractors from num_distractors_range (inclusive).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    lo, hi = int(num_distractors_range[0]), int(num_distractors_range[1])
    if lo > hi or lo < 0:
        raise ValueError("num_distractors_range must be [lo, hi] with 0 <= lo <= hi")
    comp: Complexity
    if complexity in ("low", "medium", "high"):
        comp = complexity  # type: ignore[assignment]
    else:
        comp = "medium"

    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for i in range(n):
        # Per-instance sub-RNG so order is stable under fixed seed
        sub_seed = rng.randint(0, 2**31 - 1)
        sub = random.Random(sub_seed)
        num_distractors = sub.randint(lo, hi)
        core = _sample_critical_fact(sub, comp)
        distractors = _sample_distractors(sub, num_distractors)
        task_seed = sub_seed
        entry: dict[str, Any] = {
            "id": f"delayed_cue_{i}",
            "seed": task_seed,
            "critical_fact": core["critical_fact"],
            "encoding_prompt": core["encoding_prompt"],
            "distractors": distractors,
            "recall_cue": core["recall_cue"],
            "recall_answer": core["recall_answer"],
            "num_distractors": num_distractors,
            "complexity": comp,
        }
        if "recall_expected_parts" in core:
            entry["recall_expected_parts"] = core["recall_expected_parts"]
        tasks.append(entry)
    return tasks


Phase = Literal["encoding", "distractor", "recall"]


class DelayedCueEnv:
    """
    reset() -> first observation; step(action) -> next observation.
    Attributes: observation (str), done (bool), task_success (bool after episode ends).
    step_results: per-step dicts for analysis; also surfaced as step_correctness via run_episode.
    """

    def __init__(self, task: dict[str, Any], max_steps: int = 20) -> None:
        self._task = task
        self._max_steps = max_steps
        self.observation = ""
        self.done = False
        self.task_success = False
        self.step_results: list[dict[str, Any]] = []
        self._phase: Phase = "encoding"
        self._distractor_index = 0
        self._steps_taken = 0
        self._recall_correct: bool | None = None

    def reset(self) -> str:
        self.done = False
        self.task_success = False
        self.step_results = []
        self._phase = "encoding"
        self._distractor_index = 0
        self._steps_taken = 0
        self._recall_correct = None
        self.observation = str(self._task["encoding_prompt"])
        return self.observation

    def _temporal_bin(self, step_index: int) -> str:
        return "pre-distractor" if step_index == 0 else "post-distractor"

    def _append_result(
        self,
        step_index: int,
        phase: Phase,
        expected: str,
        agent_answer: str,
        correct: bool,
    ) -> None:
        self.step_results.append(
            {
                "step": step_index,
                "phase": phase,
                "temporal_bin": self._temporal_bin(step_index),
                "expected_answer": expected,
                "agent_answer": agent_answer,
                "correct": correct,
            }
        )

    def step(self, action: str) -> str:
        if self.done:
            return self.observation

        self._steps_taken += 1
        if self._steps_taken > self._max_steps:
            self.done = True
            self.observation = "Max steps reached."
            return self.observation

        n_dist = int(self._task["num_distractors"])
        distractors: list[dict[str, str]] = self._task["distractors"]

        if self._phase == "encoding":
            self._append_result(0, "encoding", "", action, True)
            if n_dist > 0:
                self._phase = "distractor"
                self._distractor_index = 0
                self.observation = distractors[0]["question"]
            else:
                self._phase = "recall"
                self.observation = str(self._task["recall_cue"])
            return self.observation

        if self._phase == "distractor":
            step_idx = 1 + self._distractor_index
            d = distractors[self._distractor_index]
            expected = d["answer"]
            ok = answer_matches_expected(action, expected)
            self._append_result(step_idx, "distractor", expected, action, ok)
            if self._distractor_index < n_dist - 1:
                self._distractor_index += 1
                self.observation = distractors[self._distractor_index]["question"]
            else:
                self._phase = "recall"
                self.observation = str(self._task["recall_cue"])
            return self.observation

        # recall
        recall_step_index = 1 + n_dist
        expected_recall = str(self._task["recall_answer"])
        parts = self._task.get("recall_expected_parts")
        if isinstance(parts, list) and parts:
            ok = all(answer_matches_expected(action, str(p)) for p in parts)
        else:
            ok = answer_matches_expected(action, expected_recall)
        self._append_result(recall_step_index, "recall", expected_recall, action, ok)
        self._recall_correct = ok
        self.task_success = ok
        self.done = True
        self.observation = "Episode complete."
        return self.observation
