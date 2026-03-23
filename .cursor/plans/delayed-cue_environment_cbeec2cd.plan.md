---
name: Delayed-Cue Environment
overview: Replace the placeholder `delayed_cue.py` with a full episodic environment class (`DelayedCueEnv`) implementing the Delayed-Match-to-Sample paradigm from the thesis design, plus a `generate_tasks()` instance generator and comprehensive unit tests.
todos:
  - id: env-class
    content: Implement DelayedCueEnv class with reset()/step()/.done/.observation, step_results tracking, and task_success property (True iff recall step answered correctly) in src/environments/delayed_cue.py. done=True after recall step regardless of correctness.
    status: completed
  - id: content-gen
    content: "Build content generators: critical fact pools, distractor pools (arithmetic, category, logic), cue formatters with complexity scaling. Direct cue type only (no contextual)."
    status: completed
  - id: generate-tasks
    content: Rewrite generate_tasks(n, seed) to produce full task instance dicts. Sample num_distractors per instance from num_distractors_range (default [3,8]) for within-domain variance. Configurable complexity.
    status: completed
  - id: correctness
    content: "Implement step-level correctness checking: normalize whitespace/case, check expected answer is contained in response, reject if negation word immediately precedes it. Encoding step always correct."
    status: completed
  - id: base-agent-integration
    content: "Update run_episode in base_agent.py: after episode loop, check for env.step_results and attach as step_correctness in return dict."
    status: completed
  - id: config
    content: "Add delayed_cue config block to configs/experiment_core.yaml with num_distractors_range: [3, 8] and complexity: medium."
    status: completed
  - id: script-integration
    content: Update _make_env in run_phase1.py and run_phase2.py to construct DelayedCueEnv from task instances
    status: completed
  - id: unit-tests
    content: Create tests/test_07_delayed_cue_env.py with tests covering generation (count, determinism, variable distractor counts), env interface, correctness (including negation rejection), phases, task_success, and config.
    status: completed
isProject: false
---

# Delayed-Cue Recall Environment

## Context

The thesis design (`[blueprints/thesis_design.md](blueprints/thesis_design.md)` lines 183-187) specifies **Domain 2 -- Delayed-Cue Recall** as:

> Tasks where critical information is given early, followed by distractors, and the information is only needed later. Direct operationalization of the Delayed-Match-to-Sample paradigm from cognitive psychology. Tests whether metacognitive signals detect memory errors.

Currently `[src/environments/delayed_cue.py](src/environments/delayed_cue.py)` is a placeholder returning stub dicts. `[scripts/run_phase1.py](scripts/run_phase1.py)` falls back to a `TextWorldEnv(game_file=None)` stub for delayedcue domains.

## Design

### Episode structure (Delayed-Match-to-Sample)

Each episode has three phases across `2 + num_distractors` total steps (5--10 steps with default range):

```
Step 0:  ENCODING  -- present critical fact(s), agent acknowledges
Steps 1..N:  DISTRACTOR  -- unrelated questions with known answers
Step N+1:  RECALL  -- cue requiring recall of the critical fact
```

`num_distractors` varies **per instance** (sampled from `num_distractors_range`, default `[3, 8]`). This gives within-domain variance in episode length / context length, which strengthens H4 analysis and provides a secondary check for H3.

**Binary temporal tagging** (for H3/H4 analysis):

- Step 0 = `"pre-distractor"`
- Steps 1..N+1 = `"post-distractor"`

`**done` and `task_success` semantics:**

- `done = True` after the recall step, **regardless of whether the answer was correct**
- `task_success` property: `True` iff the recall step was answered correctly (analogous to TextWorld quest completion)

### Task instance schema (returned by `generate_tasks`)

```python
{
    "id": "delayed_cue_42",
    "seed": 12345,
    "critical_fact": {"subject": "Alice", "attribute": "city", "value": "Paris"},
    "encoding_prompt": "Remember this: Alice lives in Paris.",
    "distractors": [
        {"question": "What is 7 * 8?", "answer": "56"},
        {"question": "What color do you get mixing red and blue?", "answer": "purple"},
        ...
    ],
    "recall_cue": "In which city does Alice live?",
    "recall_answer": "Paris",
    "num_distractors": 5,       # sampled from num_distractors_range per instance
    "complexity": "medium",
}
```

### `DelayedCueEnv` class -- same interface as `TextWorldEnv`

```python
class DelayedCueEnv:
    observation: str       # current step's text
    done: bool
    task_success: bool     # True iff recall step answered correctly
    step_results: list[dict]  # per-step records for analysis
    
    def __init__(self, task: dict, max_steps: int = 20) -> None: ...
    def reset(self) -> str:       # returns encoding observation
    def step(self, action: str) -> str:  # returns next observation
```

`step_results` entries: `{"step": int, "phase": str, "temporal_bin": str, "expected_answer": str, "agent_answer": str, "correct": bool}`

- `phase`: `"encoding"` / `"distractor"` / `"recall"`
- `temporal_bin`: `"pre-distractor"` (step 0) / `"post-distractor"` (steps 1+)

**Correctness checking**: Normalize whitespace/case, check if expected answer is contained in agent's response AND no negation word ("not", "don't", "isn't", "no") immediately precedes it. Encoding step accepts any response as correct. This is a known methodological limitation (noted, not over-engineered).

### Content generators (for `generate_tasks`)

Three categories of critical facts, scaling with `complexity`:

- **Associations**: "X lives in Y", "The key is Z" (low/medium)
- **Numeric**: codes, sequences (medium/high)
- **Multi-fact**: 2-3 facts that must all be recalled (high)

Distractor pools:

- Arithmetic (e.g., "What is 13 * 7?")
- Word/category (e.g., "Name a fruit that starts with P")
- Simple logic (e.g., "If all cats are animals, and Felix is a cat, what is Felix?")

Cue type: `"direct"` only (straightforward question, e.g. "What was the code?"). Contextual cues (indirect references) are out of scope for the core experiment -- they introduce a confound by testing inference, not just recall. Noted as a future direction.

### Config integration

Add delayedcue-specific defaults to `[configs/experiment_core.yaml](configs/experiment_core.yaml)` under a new `delayed_cue` key:

```yaml
delayed_cue:
  num_distractors_range: [3, 8]   # sampled per instance for within-domain variance
  complexity: "medium"             # low | medium | high
```

### `base_agent.py` integration

After the episode loop in `run_episode` (`[src/agent/base_agent.py](src/agent/base_agent.py)`), check for `env.step_results` and include it in the return dict as `"step_correctness"`. This keeps the step function signature stable and works generically for any environment that exposes `step_results`.

### Script integration

Update `_make_env` in `[scripts/run_phase1.py](scripts/run_phase1.py)` (line 84-97) to construct `DelayedCueEnv` instead of a TextWorldEnv stub. Similarly update `[scripts/run_phase2.py](scripts/run_phase2.py)` if it has the same pattern.

### Unit tests

New file `[tests/test_07_delayed_cue_env.py](tests/test_07_delayed_cue_env.py)` following existing conventions (numbered, `from __future__ import annotations`, docstring):

- `test_generate_tasks_returns_correct_count` -- n=10, check length
- `test_generate_tasks_deterministic_with_seed` -- same seed = same output
- `test_generate_tasks_different_seeds` -- different seeds = different output
- `test_task_instance_schema` -- check required keys in generated dict
- `test_env_reset_returns_encoding_observation` -- contains critical info
- `test_env_step_returns_observation` -- type/non-empty checks
- `test_env_done_after_all_steps` -- done is True after encoding + distractors + recall
- `test_env_not_done_mid_episode` -- done is False during distractors
- `test_step_correctness_tracking` -- correct answers produce `correct=True` in stepresults
- `test_step_incorrect_answer` -- wrong answer produces `correct=False`
- `test_encoding_step_always_correct` -- any response to encoding is correct
- `test_binary_temporal_phases` -- stepresults phases are pre-distractor / post-distractor
- `test_variable_distractor_count` -- instances from same generator have varying num_distractors within range
- `test_recall_answer_matches_critical_fact` -- recall_answer field matches value
- `test_task_success_correct_recall` -- task_success is True when recall answered correctly
- `test_task_success_incorrect_recall` -- task_success is False when recall answered incorrectly
- `test_done_after_recall_regardless` -- done is True after recall step even with wrong answer
- `test_negation_rejection` -- "not Paris" is marked incorrect when expected answer is "Paris"

## File changes


| File                               | Change                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `src/environments/delayed_cue.py`  | Replace placeholder with `DelayedCueEnv` class + `generate_tasks()`          |
| `src/agent/base_agent.py`          | Attach `env.step_results` as `step_correctness` in `run_episode` return dict |
| `tests/test_07_delayed_cue_env.py` | New: unit tests (18 tests)                                                   |
| `configs/experiment_core.yaml`     | Add `delayed_cue:` config block with `num_distractors_range`                 |
| `scripts/run_phase1.py`            | Update `_make_env` to use `DelayedCueEnv`                                    |
| `scripts/run_phase2.py`            | Update env creation if applicable                                            |
| `src/environments/__init__.py`     | Optional: add export                                                         |


