---
name: TextWorld sweep and dataset
overview: Implement a sweep-first TextWorld Cooking pipeline for thesis-grade instance generation, fixed holdout split, and sidecar-backed step-correctness support in TextWorldEnv, without external benchmark anchoring.
todos:
  - id: build_textworld_generator
    content: Implement `scripts/generate_textworld_games.py` with parameterized TextWorld Cooking generation, per-instance sub-seeds, `.ulx` output, and rich `.json` sidecars.
    status: completed
  - id: build_textworld_player
    content: Implement `scripts/play_textworld.py` interactive CLI for manual game debugging with score and termination status output.
    status: completed
  - id: extend_textworld_env_sidecar
    content: Update `src/environments/textworld_env.py` to load sidecar metadata and compute step-level correctness from score/admissibility/error feedback (not walkthrough matching).
    status: completed
  - id: build_textworld_sweep
    content: Implement `scripts/sweep_textworld_difficulty.py` to run grid sweep on the updated real TextWorld env, execute C0 episodes, and emit `sweep_results.json` with success/length statistics and ranked candidates.
    status: completed
  - id: create_final_manifest_pipeline
    content: Add deterministic final dataset manifest flow producing `difficulty_manifest.json` with metadata, difficulty tier, and fixed 5/45 holdout split.
    status: completed
isProject: false
---

# TextWorld Cooking Thesis Plan (Revised)

## Scope Lock

- Use exactly two domains for core experiments: TextWorld Cooking (exploration) and Tower of Hanoi (planning).
- Remove external benchmark anchoring entirely (no appendix subset, no external validation games).
- TextWorld parameters are selected only after sweep results; no hardcoded final combination before sweep.

## Files to Implement / Update

- New generator: `[/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/generate_textworld_games.py](/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/generate_textworld_games.py)`
- New sweep runner: `[/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/sweep_textworld_difficulty.py](/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/sweep_textworld_difficulty.py)`
- New human debugger: `[/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/play_textworld.py](/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/scripts/play_textworld.py)`
- Update env wrapper: `[/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/src/environments/textworld_env.py](/Users/keanuprivatbenutzer/Documents/git_repos/metacog-llm-compute/src/environments/textworld_env.py)`
- Add dataset manifest output under: `data/tasks/textworld/difficulty_manifest.json`

## Implementation Plan

### 1) TextWorld instance generation script

Implement `generate_textworld_games.py` to generate `.ulx` + `.json` sidecar per instance into `data/tasks/textworld/`.

Required CLI flags:

- `--num-rooms`
- `--num-ingredients`
- `--cut`
- `--cook`
- `--open`
- `--num-instances`
- `--seed`
- Optional output path defaulting to `data/tasks/textworld`

Behavior:

- Derive deterministic per-instance sub-seeds from master seed.
- Generate each game with `max_steps=50` at generation time.
- Save sidecar JSON with:
  - generation parameters
  - master seed + instance seed
  - walkthrough from `game.metadata["walkthrough"]`
  - entity list
  - max_score
  - expected step count (walkthrough length)
- Ensure file naming is stable for downstream runners (e.g., `textworld_0.ulx`, `textworld_0.json`).

### 2) Interactive game player

Implement `play_textworld.py` for manual validation:

- Input: `.ulx` path
- Start interactive loop with text commands
- After each command print:
  - observation
  - current score / max score
- On termination print one of:
  - `YOU WON`
  - `YOU LOST`
  - `MAX STEPS REACHED`

### 3) TextWorldEnv sidecar + step-correctness support

Update `textworld_env.py` so real TextWorld mode:

- Loads `.ulx` as already done.
- Loads sidecar JSON (`same_stem.json`) when present.
- Exposes walkthrough metadata for reference/debugging (not as correctness oracle).

Step-correctness policy (for thesis consistency):

- `optimal`: score increased after the action (progress toward recipe goal).
- `legal`: action admissible / no parser error, but score unchanged.
- `illegal`: action not recognized or game feedback indicates an error.
- Keep walkthrough in sidecar for diagnostics and post-hoc analysis only.
- Continue logging `step_results` with `step_index`, action fields, correctness, and state snapshots.

### 4) Difficulty sweep script

Implement `sweep_textworld_difficulty.py` to evaluate a parameter grid and identify the operating point, after TextWorldEnv real-mode correctness tracking is in place.

Grid:

- rooms in `{3, 5, 7}`
- ingredients in `{1, 2, 3}`
- operations in `{take-only, take+cook, take+cut+cook}`

Per combination:

- Generate a small batch (configurable 5–10 instances).
- Run C0 episodes via existing agent loop (reuse `run_episode`/`get_step_fn` and current model wrapper flow).
- Record:
  - mean success rate
  - mean episode length
  - step-count distribution
- Write `sweep_results.json` and print ranked console summary.

Selection criterion:

- Choose the combination closest to target window:
  - C0 success rate: `30–50%`
  - episode length: `8–15` steps

### 5) Final 50-instance dataset + fixed holdout split

After sweep selects parameters:

- Run generator once with `--num-instances 50` and chosen settings.
- Treat outputs as immutable dataset artifacts.
- Generate `difficulty_manifest.json` containing all 50 with:
  - instance id/path
  - full generation metadata
  - difficulty tier
  - `holdout: true/false`

Split policy:

- Fixed before any phase runs.
- 5 holdout instances (10%) for Phase 1 threshold tuning.
- Remaining 45 for Phase 2.
- Deterministic holdout assignment (documented rule in manifest generation).

## Validation & Acceptance Checks

- Generator creates 50 paired `.ulx` + `.json` files reproducibly from seed.
- Sidecar includes non-empty walkthrough and expected step count for all instances.
- Sweep produces `sweep_results.json` and clear best-configuration selection rationale.
- Selected configuration satisfies or is nearest to `30–50%` C0 and `8–15` steps.
- `TextWorldEnv` uses score/admissibility/error feedback for step correctness in real-game mode.
- `difficulty_manifest.json` contains fixed 5/45 split and required metadata.

## Out-of-Scope (explicitly removed)

- Any external/validated benchmark-game subset.
- Any benchmark anchoring appendix workflow.

