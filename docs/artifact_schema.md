# Artifact Schema

## Episode JSON (`episode.v1`)

Episode files written by `log_episode(...)` include `schema_version: "episode.v1"`.

### Required high-level keys

- `episode_id`
- `compute_stage` (Phase 1) or `strategy` (Phase 2)
- `task_success`

### Analysis fields (Phase 1 / Phase 2)

- `domain`, `instance`, `run`
- `holdout` (bool) — from task manifest
- `difficulty_tier` — from task manifest
- `stage_per_step` (Phase 2 adaptive runs)
- `tle_per_step`, `vc_per_step`, `step_correctness`

### Compact storage (`compact=True`, default)

Full `steps_detail`, `vc_detail_per_step`, and `logprob_raw_per_step` may be omitted from the
main JSON to save space. **Minimal per-step records** are retained for analysis joins:

- `step_index`, `compute_stage`, `tle`, `vc`, `tokens_generated`, `lm_calls`, `correctness`

Optional sidecars (when `logging.save_*` enabled):

- `logprobs/{episode_id}_logprobs.json`
- `vc/{episode_id}_vc.json`

### Loader compatibility

`src/analysis/datasets.py` validates core fields and synthesizes `steps_detail` from legacy
per-step arrays when the full detail list is absent.

Golden fixture: `tests/fixtures/episode_schema_v1.json`.
