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

### Full raw-text traces (`trace_{episode_id}.jsonl`) — undocumented until 2026-08-04

**Not opt-in in practice**: `logging.write_debug_views: true` (`experiment_core.yaml`, the real
Phase 1/2 config) forces `save_step_traces=True` internally
(`src/utils/trace_debug_view.py::resolve_step_trace_flags`), regardless of the separate
`save_step_traces: false` written in the same file. One line per step, written directly under the
checkpoint dir (not gitignored `data/results/` exclusion-exempt -- these are real per-run
artifacts, not committed to the repo). Each record carries the **full, untruncated** prompt and
model response for every candidate generated at that step -- including the complete `<think>`
block for C1/C2, and every C2 self-consistency candidate individually (`call_detail["subcalls"]`),
even a candidate whose `<think>` block never closes (`reject_reason="thinking_unclosed"`). This is
the only artifact that preserves the exact reasoning text a reader would need to quote a real
example (e.g. a runaway/never-closing reasoning trace) in the thesis discussion -- the compact
episode JSON and the debug views below both truncate or omit it.

Empirically on the real Phase 1 data (1500 episodes): 2.71 GB total, ~49.3 KB/step average. Cheap
relative to available storage; no retention/pruning policy is applied or needed at this scale.

`debug_views/episode_*.json` (built post-run by `finalize_run_debug_views`,
`src/utils/run_output_layout.py`) is the head/tail-truncated (800 chars each,
`trace_debug_view.py::DEFAULT_HEAD_CHARS`/`DEFAULT_TAIL_CHARS`) human-browsable summary generated
*from* the `trace_*.jsonl` files -- meant for quick skimming across many episodes, not as a
replacement for the full trace when the exact text matters. Both files coexist; neither is deleted
after the other is built.

### Adaptive-allocation decision fields (Phase 2, `adaptive_tle`/`adaptive_vc`/`eager_style` only)

Per step, alongside `compute_stage` (the discrete C0/C1/C2 decision): `allocator_uncertainty_score`
(the continuous [0,1] percentile-based score `AllocationPolicy.stage()` computes before
thresholding into a stage -- previously computed and discarded, now retained for the "when does
the agent choose to deliberate" analysis, Ch.7.4) and `allocator_theta1`/`allocator_theta2` (the
frozen policy's threshold values, for context). Absent for `always_c0`/`always_c2`/`random` (no
policy-driven decision to log) and for step 0 of any adaptive strategy (`signal=None`, defaults to
C0 without a policy lookup).

### Loader compatibility

`src/analysis/datasets.py` validates core fields and synthesizes `steps_detail` from legacy
per-step arrays when the full detail list is absent.

Golden fixture: `tests/fixtures/episode_schema_v1.json`.
