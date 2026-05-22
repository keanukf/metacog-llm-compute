# TextWorld Cooking dataset

Core experiments use **TextWorld Cooking** (partial observability / exploration) and **Tower of Hanoi** (full observability / planning). TextWorld games are **not** bundled inside the library: you **generate** compiled story files (Inform 7 **`.z8`**, or legacy **`.ulx`**) and load them by path. This repo builds a reproducible dataset under `data/tasks/textworld/`.

**Related:** [`docs/scripts.md`](scripts.md), [`configs/experiment_core.yaml`](../configs/experiment_core.yaml) (`paths.tasks_dir`).

## Generation (`scripts/generate_textworld_games.py`)

Generates TextWorld Cooking games via `python -m textworld.challenges.tw_cooking` with:

- **CLI:** `--num-rooms`, `--num-ingredients`, `--cut`, `--cook`, `--open`, `--num-instances`, `--seed` (master seed; each instance gets a deterministic sub-seed).
- **Output:** `textworld_{i}.z8`, TextWorld’s **serialized game** `textworld_{i}.json` (required at play time — do not overwrite), and experiment metadata **`textworld_{i}.meta.json`** in `--output-dir` (default `data/tasks/textworld/`).
- **`.meta.json`** includes generation parameters, seeds, walkthrough from game metadata (reference/debugging), entity list, `max_score`, and expected step count (walkthrough length).
- **Generation-time** `max_steps` is **50** (generous engine limit). The **agent** caps episodes via `episode.max_steps_per_episode` in config (e.g. 20–25); do not rely on the game file alone for the experiment cap.

Optional: `--write-manifest` writes `difficulty_manifest.json` in the output directory (useful after the final 50-instance run).

Example (from repo root; requires `textworld` installed):

```bash
python scripts/generate_textworld_games.py \
  --num-rooms 5 \
  --num-ingredients 2 \
  --cook \
  --seed 42 \
  --num-instances 3
```

This writes `data/tasks/textworld/textworld_0.z8` … `textworld_2.z8`, matching `textworld_{i}.json` (game dump) and `textworld_{i}.meta.json`. Add `--cut` for cutting, or omit `--cook` for a take-only style run, depending on the sweep cell you are testing.

## Difficulty sweep (`scripts/sweep_textworld_difficulty.py`)

Runs a grid over rooms `{3,5,7}` × ingredients `{1,2,3}` × operations `{take-only, take+cook, take+cut+cook}`, generates a small batch per cell, runs **C0** episodes via the existing agent loop, and writes **`sweep_results.json`** plus a ranked console summary. Goal: land near **30–50% C0 task success** and **8–15 steps** mean episode length before freezing parameters for the final 50 instances. Use `--real` with `configs/experiment_core.yaml` (or your sweep config) so the model matches the thesis run (e.g. Qwen2.5-3B-Instruct on GPU).

## Manual play (`scripts/play_textworld.py`)

Interactive terminal play for a single compiled story (`.z8` or `.ulx`): observations, score vs max score, and termination as `YOU WON` / `YOU LOST` / `MAX STEPS REACHED`. Use this to sanity-check generated games before long sweeps. Requires `textworld` (same stack as generation).

Example (from repo root; path matches instances produced by `generate_textworld_games.py`):

```bash
python scripts/play_textworld.py data/tasks/textworld/textworld_0.z8
```

Use `--max-steps N` to change the interactive cap for the session (default 50).

## Final manifest (`scripts/build_textworld_manifest.py`)

After generating the **50** immutable instances with chosen parameters, build **`data/tasks/textworld/difficulty_manifest.json`**: metadata per instance, difficulty tier, and **`holdout: true/false`** for **5 / 45** split (Phase 1 threshold tuning vs Phase 2). Policies: `--holdout-policy first-n` or `mod-10`.

## `TextWorldEnv` and step correctness

`src/environments/textworld_env.py` loads a `.z8`/`.ulx` and, when present, the **`textworld_{i}.meta.json` sidecar** (walkthrough kept for reference only; legacy `textworld_{i}.json` experiment-only files are still detected if they contain `generation_parameters`). Step labels use the **game engine**, not walkthrough matching (Cooking allows multiple valid orderings):

- **`optimal`:** score increased after the step.
- **`legal`:** admissible / no parser error, score unchanged.
- **`illegal`:** unrecognized command or error feedback.

Phase runners resolve games from `paths.tasks_dir` in `experiment_core.yaml`: either `data/tasks/textworld_{i}.z8`/`.ulx` or **`data/tasks/textworld/textworld_{i}.z8`/`.ulx`** (see `src/utils/experiment_env.py`).
