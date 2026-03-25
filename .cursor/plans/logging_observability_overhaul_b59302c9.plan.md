---
name: Logging observability overhaul
overview: Enrich episode JSONs with per-step detail, correct LM-call accounting, add run-level metadata + resilient progress/error reporting for overnight GPU runs, and add analysis utilities + loaders (backward compatible) with updated tests.
todos:
  - id: step-schema
    content: Update compute stages to return 5-tuple incl. lm_calls_this_step; add steps_detail + per-step timing + total_lm_calls semantics in base_agent.
    status: completed
  - id: episode-metadata
    content: Add episode-level compute/timestamp fields and keep backward-compatible aliases.
    status: completed
  - id: run-metadata-summary
    content: Implement write_run_metadata + progress reporter + error resilience + run_summary.json in phase scripts.
    status: completed
  - id: analysis-utils
    content: Extend calibration.py with AUROC, step-position calibration, discrimination report, efficiency utilities.
    status: completed
  - id: loaders
    content: Implement load_episodes/load_steps with backward compat and optional DataFrame output.
    status: completed
  - id: toh-optimal-remaining
    content: Add optimal_moves_remaining to Tower of Hanoi step_results.
    status: completed
  - id: tests-update
    content: Update existing tests/fixtures and add test_08_analysis_utils.py to cover new utilities and schema.
    status: completed
  - id: verification
    content: Run pytest and iterate until green.
    status: completed
isProject: false
---

# Logging, Observability & Analysis Utilities Upgrade

## Goals

- Add **step-level structured logs** (`steps_detail`) that support H1/H2/H3/H4 analyses (ECE/Brier by step position, AUROC discrimination, ToH correctness-level distributions, Phase 2 allocation patterns).
- Add **episode-level compute semantics** (`total_lm_calls`, `normalized_compute_cost`, `efficiency_score`) and timestamps.
- Add **run-level metadata** (`run_metadata.json`) and **end-of-run summaries** (`run_summary.json`) plus robust progress/error reporting for overnight GPU runs.
- Add **analysis utilities** (AUROC, step-position calibration, discrimination report, strategy efficiency) without heavy deps.
- Add **loaders** to read episodes and flatten steps into a DataFrame for downstream stats/MEM.
- Keep **backward compatibility** with existing episode JSONs lacking `steps_detail`.

## Current state (key findings)

- `src/agent/base_agent.py` currently counts `lm_calls += 1` per step regardless of compute stage, and returns only flat lists `tle_per_step`/`vc_per_step` plus optional `step_correctness` (copied from env).
- `src/agent/compute_stages.py` step fns return a 4-tuple `(action, tle, vc, tokens_used)` and `c1_step` is explicitly a stub (single call).
- Phase scripts (`scripts/run_phase1.py`, `scripts/run_phase2.py`) write checkpoint JSONs via `src/utils/checkpointing.py` → `src/utils/logging_utils.py` and currently print progress only every 50 new episodes.
- Tower of Hanoi env already computes `_shortest_path_to_goal(...)` and stores step results with `correctness`.

## Design decisions

- **New canonical step record**: `steps_detail: list[dict]` in every agent episode result. Keep `tle_per_step` / `vc_per_step` as redundant compatibility lists.
- **LM call accounting** comes from compute-stage step fns returning `lm_calls_this_step`. Base agent sums these into `total_lm_calls`.
- **Tokens accounting**: store `total_tokens_generated` (alias old `tokens`). Per step: `tokens_generated`.
- **Correctness per step**: sourced from env `step_results` when present, joined by `step_index`. If unavailable, store `None`.
- **Observation length**: measured on the observation passed *into* the step fn (proxy for context growth).
- **Normalization for old JSONs**: loaders will synthesize `steps_detail` from legacy fields when missing.

## Implementation steps

### 1) Step-level detail + correct LM call semantics

- Update `src/agent/compute_stages.py`:
  - Change `c0_step`, `c1_step`, `c2_step` return signature to 5-tuple:
    - `(action, tle, vc, tokens_used, lm_calls_this_step)`.
  - Set `lm_calls_this_step`:
    - `c0_step`: `1` (note: VC currently parsed from same response, no extra prompt).
    - `c1_step`: `1` (current implementation is a single model call). Add a short comment that this should become `2` when C1 is fully implemented as **CoT + verify** (two calls).
    - `c2_step`: `n_samples`.
- Update `src/agent/base_agent.py`:
  - Extend `_normalize_step_result` to accept 3/4/5 tuples and return `(action, tle, vc, tokens_used, lm_calls_this_step)` with sensible defaults.
  - Wrap every `step_fn(...)` with `time.perf_counter()` to compute `step_wall_time_s`.
  - Build `steps_detail` entries with the schema you provided.
  - Replace `lm_calls` meaning:
    - Keep `steps` as step count.
    - Add `total_lm_calls` (sum of `lm_calls_this_step`).
    - Preserve `lm_calls` in outputs/scripts as legacy but repoint it to `steps` OR keep as-is and add explicit `episode_length_steps` + `total_lm_calls` (we’ll align scripts/tests to the new explicit keys).

### 2) Episode-level metadata additions

- In `src/agent/base_agent.py` return dicts (both `run_episode` and `run_adaptive_episode`):
  - Add: `episode_length_steps`, `total_lm_calls`, `total_tokens_generated`, `normalized_compute_cost`, `efficiency_score`, `timestamp_utc`.
  - Keep: `tokens` as alias of `total_tokens_generated` for backward compat.
  - Compute `normalized_compute_cost = total_lm_calls / (max_steps * 3)` (guard `max_steps==0`).
  - Compute `efficiency_score = task_success / normalized_compute_cost` (None if cost is 0).

### 3) Run-level metadata utility

- Create/extend `src/utils/logging_utils.py` with:
  - `write_run_metadata(checkpoint_dir: Path, config: dict, *, script: str, config_path: str|Path, pilot_mode: str, model_name: str, model_dtype: str, domains: list[str], total_episodes_planned: int, resumed_from: int) -> Path`
  - Implementation details:
    - `run_id`: UUID4 or timestamp-based string.
    - `config_hash`: SHA256 of config file bytes.
    - `git_commit`: best-effort via `subprocess` (return None on failure).
    - `python_version`, `hostname`.
    - `gpu_name`, `vram_total_gb`: best-effort via `torch.cuda` if available; else None.
    - `timestamp_start_utc`: ISO-8601 UTC.

### 4) Console observability + error resilience in phase scripts

- Update `scripts/run_phase1.py` and `scripts/run_phase2.py`:
  - Add configurable reporting:
    - Default `progress_every_episodes = 10` (configurable from YAML and/or CLI).
    - Also emit a report at least every 300 seconds.
  - Track rolling window (last 10 episodes) for: success, avg steps, avg time/ep, avg TLE (mean_entropy), avg VC.
  - Print domain/stage/strategy and instance range for the last batch.
  - Wrap each episode execution in try/except:
    - On exception: append JSON line to `errors.jsonl` in checkpoint dir with `episode_id`, `domain`, `instance`, `stage_or_strategy`, `run`, `timestamp_utc`, and full traceback.
    - Print warning and continue.
  - At completion:
    - Write and print `run_summary.json` with requested aggregates **plus explicit episode outcome counts** so reruns are obvious after an overnight run:
      - `episodes_attempted`: number of episode executions attempted by this process (excludes episodes skipped due to `--resume`).
      - `episodes_completed`: number of attempted episodes that finished and were checkpointed successfully.
      - `episodes_failed`: number of attempted episodes that raised and were logged to `errors.jsonl`.
      - `errors`: keep as alias of `episodes_failed` for backward/UX convenience.
      - `new_episodes_this_run`: should equal `episodes_completed` (not attempted).
      - `total_episodes`: total checkpoints in directory at end of run (completed prior + completed now).
  - At start:
    - Call `write_run_metadata(...)` once to create `run_metadata.json`.

### 5) Analysis utilities

- Extend `src/analysis/calibration.py`:
  - `compute_auroc(scores, labels)` using Mann–Whitney U statistic (pure Python; optional SciPy validation if present, but no sklearn).
  - `calibration_by_step_position(...)`:
    - Bin steps by relative position into `n_bins` (labels like `bin_0..bin_{n_bins-1}` or `early/mid/late` when `n_bins==3`; we’ll keep deterministic labels).
    - Extract per-step signal:
      - `signal='tle'` → use `steps_detail[i]['tle']['mean_entropy']`.
      - `signal='vc'` → use `steps_detail[i]['vc']/100`.
    - Extract per-step correctness as binary (correct=1):
      - Tower of Hanoi: `correctness in {'optimal','legal'}` counts as correct (illegal incorrect) unless you prefer only `optimal` as correct; we’ll default to **legal-or-optimal = correct** to match “action validity” calibration and keep this configurable.
      - TextWorld: `correctness == 'legal'` correct, `illegal` incorrect.
    - Return ECE and Brier per position bin.
  - `signal_discrimination_report(episodes, signal)` computing AUROC + Cohen’s d + group means (correct vs incorrect steps).
  - `compute_efficiency(success_rate, normalized_compute_cost)` and `compute_strategy_efficiency(episodes)` grouped by `compute_stage` (Phase 1) or `strategy` (Phase 2) depending on presence of keys.

### 6) Aggregate episode + step loaders

- Add to `src/utils/logging_utils.py` (or create `src/analysis/data_loader.py` if you prefer analysis isolation):
  - `load_episodes(checkpoint_dir, as_dataframe=False)`:
    - Load all `ep_*.json` from a directory.
    - Backward compat: if an episode lacks `steps_detail`, synthesize it from `tle_per_step`, `vc_per_step`, `stage_per_step` (if present), `step_correctness` (if present), and fill unknown fields (`tokens_generated`, `step_wall_time_s`, etc.) with `None/0`.
  - `load_steps(checkpoint_dir) -> pd.DataFrame`:
    - Flatten `steps_detail` into one row per step, joined with episode-level columns (`episode_id`, `domain`, `instance`, `strategy/compute_stage`, `run`, `task_success`, etc.).
    - Keep optional dependency: pandas is already used elsewhere? If pandas is not in requirements, we’ll implement `as_dataframe=True` only when pandas import succeeds (else raise a clear error).

### 7) Tower of Hanoi: `optimal_moves_remaining`

- Update `src/environments/tower_of_hanoi.py`:
  - In each `step_results` entry add `optimal_moves_remaining: int` computed as `len(_shortest_path_to_goal(self._state, self._num_disks))` after applying/attempting the move (or recompute from `state_after`).

### 8) Tests

- Update fixtures and existing tests:
  - `tests/conftest.py` `sample_episode_data` adds `steps_detail` (minimal 1–2 step dicts) and new episode-level keys.
  - `tests/test_05_e2e_mini_experiment.py` asserts:
    - `steps_detail` present.
    - Each entry has required keys (`compute_stage`, `lm_calls_this_step`, `step_wall_time_s`, etc.).
  - `tests/test_07_tower_of_hanoi.py` asserts `optimal_moves_remaining` exists in `env.step_results[-1]` and is an int.
- Add `tests/test_08_analysis_utils.py`:
  - `compute_auroc`: known small cases (perfect separation => 1.0; equal scores => 0.5).
  - `calibration_by_step_position`: smoke test with synthetic `steps_detail` and correctness.
  - `signal_discrimination_report`: returns expected keys; AUROC in [0,1].
  - `load_episodes` and `load_steps`: backward-compat with legacy episode dicts; flattening produces expected row counts.

## Verification

- Run `python -m pytest tests/ -v` and fix any regressions.

## Notes / small clarifications baked into implementation

- `compute_stage` will be stored per step for **both** Phase 1 and Phase 2. In Phase 2 it comes from `stage_per_step`; in Phase 1 it’s constant per episode.
- `c1_step` LM-call count will reflect **current reality (1 call)** until CoT+Verify is actually implemented as two calls; at that point it will be updated to `2` in code and any compute-normalization expectations should be revisited accordingly.

