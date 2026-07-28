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
- `total_prompt_tokens` — input-token side of "Total Tokens Processed" (secondary DV, revision_audit
  P1-stat-7). Backend-reported (`GenerateResult.prompt_tokens`), booked per candidate the same way
  `total_tokens_generated` (output) already is. **Absent on Phase 1 episodes** (added 2026-07-28,
  after Phase 1 collection; a deliberate economy decision, not missing data — no Phase 1 analysis
  depends on it) and **present from Phase 2 on**. `0`/absent means "backend couldn't report it"
  (e.g. `VLLMWrapper`/`LMStudioWrapper` don't wire this yet — only `ServerBackend`, the actual
  production backend, does), not "zero tokens were used."

### Compact storage (`compact=True`, default)

Full `steps_detail`, `vc_detail_per_step`, and `logprob_raw_per_step` may be omitted from the
main JSON to save space. **Minimal per-step records** are retained for analysis joins:

- `step_index`, `compute_stage`, `tle`, `vc`, `tokens_generated`, `prompt_tokens`, `lm_calls`,
  `correctness`

Optional sidecars (when `logging.save_*` enabled):

- `logprobs/{episode_id}_logprobs.json`
- `vc/{episode_id}_vc.json`

### Loader compatibility

`src/analysis/datasets.py` validates core fields and synthesizes `steps_detail` from legacy
per-step arrays when the full detail list is absent.

Golden fixture: `tests/fixtures/episode_schema_v1.json`.
