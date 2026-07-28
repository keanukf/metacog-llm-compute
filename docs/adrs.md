# ADR Log (Key Harness Decisions)

## ADR-001: C1 CoT->Verify split — SUPERSEDED by ADR-005

- **Decision:** Keep C1 as two explicit calls (draft CoT parse + verification pass).
- **Rationale:** Improves controllability, explicit error modes, and traceability of draft correction.
- **Status (added 2026-07-21, never previously noted):** this does not describe the current
  implementation and has not since commit `c9bdfba` ("Refactor C1/C2 compute stages for
  single-axis TLE and pinned Qwen3 revisions"), well before this note was added. C1 has been a
  single call with native thinking (`enable_thinking=True`, no separate verify pass) for a long
  time; this ADR was simply never updated when that changed. Kept here only as historical record
  — do not treat it as current. See ADR-005.

## ADR-002: LM Studio logprobs via `/v1/responses`

- **Decision:** Use LM Studio `/v1/responses` for token logprobs when available.
- **Rationale:** OpenAI-compat chat/completions often omit usable logprobs for TLE.

## ADR-003: Episode artifact schema version marker

- **Decision:** Add `schema_version` to episode JSON writes (`episode.v1` baseline).
- **Rationale:** Enables safe loader evolution and explicit migration semantics.

## ADR-004: Unified `top_logprobs` across inference backends

- **Decision:** Single config key `inference.top_logprobs` (default 20); vLLM and LM Studio normalize to the same per-token record schema (`token`, `logprob`, `top_logprobs[]`); Shannon TLE stays in `token_entropy.py`.
- **Rationale:** EAGER-aligned top-k entropy without duplicating normalization or entropy logic per backend; `lmstudio_top_logprobs` remains a deprecated alias.

## ADR-005: C1 as a single native-thinking call (supersedes ADR-001)

- **Decision:** C1 is one LM call with `enable_thinking=True` — the model reasons inside a native
  `<think>...</think>` block, then commits one action on the first non-empty line after it. No
  separate draft-then-verify pass.
- **Rationale:** Matches the single, stage-agnostic TLE measurement window (committed-action tokens
  only, identical across C0/C1/C2) that the thesis design requires — a two-call draft+verify split
  would make "the committed action" ambiguous between calls. Predates this ADR entry by a long
  margin (commit `c9bdfba`); documented now only because ADR-001 was found still describing the old
  design and nobody had corrected it.
- **2026-07-21 addendum:** C1 and C2 now share one reasoning engine
  (`src/agent/stages/shared.py::reasoning_step_core`) — C1 is that engine called with
  `n_samples=1` (no vote), C2 with `n_samples=3` (self-consistency + majority vote). Both require a
  closed `</think>` block before a candidate is admissible; an unclosed one used to get its literal
  `"<think>"` text parsed as the action in C1 specifically (found and fixed 2026-07-20/21, see
  `docs/consistency_log.md`) — C2 already rejected this case correctly, C1 didn't share the check
  until this unification.

## ADR-006: Stage-conditional z-standardization in `fit_h3_model`

- **Decision:** the H3 GEE model's signal (TLE or VC) is z-standardized (mean 0, SD 1) *within
  each `compute_stage` group* (C0/C1/C2) before fitting, not pooled across stages. Position
  (`position_norm`) is unaffected — still domain-wide mean-centered only, per the original design.
- **Context:** found during a pre-Phase-1-analysis audit cross-checking the thesis prose against
  the code (`../metacog-thesis/notes/revision_audit_2026-07.md`, item P0-5). Ch.5 §5.2.1/§5.8 say
  the signal enters the model "z-standardised"; `fit_h3_model` only mean-centered (`z - z.mean()`),
  never divided by SD, and pooled across all three compute stages when computing that mean.
- **Rationale:**
  - **Non-commensurability across stages is already the thesis's own argument (§5.3):** C0/C1/C2
    use different decoding temperatures (0.3/0.5/0.7) and reasoning-token budgets (C1/C2 add an
    entire `<think>` block TLE never sees in C0), so raw TLE/VC scale is not comparable across
    stages independent of the underlying construct. Pooling the standardization across stages lets
    a stage-driven scale difference masquerade as signal.
  - **Consistent with the allocator's own normalization.** The Phase 2 allocator normalizes TLE/VC
    stage-wise (per-stage ECDF, not pooled) for exactly this reason (see `CLAUDE.md` terminology
    cheat sheet — pooled-ECDF is explicitly a rejected legacy pattern there). Standardizing the H3
    confirmatory model pooled-across-stage while the allocator itself is stage-wise would be
    internally inconsistent between two parts of the same thesis.
  - **GEE significance is invariant to this choice, only interpretation changes.** Rescaling a
    covariate by a constant scales its coefficient and standard error by the same factor, so the
    Wald statistic, p-value, and significance decision at any given stage's data are unchanged by
    switching from centering-only to z-standardizing *within that stage*. The two things this
    *does* change: (a) whether cross-stage-pooled variance differences leak into the fitted
    coefficient (fixed by standardizing per stage, independent of z vs. z/SD), and (b) coefficient
    interpretability (per-SD effect sizes, APA 7 convention) — both real gains, no downside.
  - **Coherence with the H3 power simulation.** `docs/gate_e_h3_power_simulation.md` already
    expresses attenuation thresholds per-SD (β_z, SD=1 synthetic signals). A production model that
    only mean-centers yields raw-unit coefficients not on the same scale as the reported power
    thresholds — standardizing puts the fitted model and the pre-registered power analysis on the
    same scale.
  - **Cross-domain / cross-signal comparability.** TLE (nats, SD≈0.11) and VC (0–100, SD≈20–30)
    are not comparable in raw units; per-SD standardization is what makes "does TLE or VC degrade
    faster" (H3) a well-posed question in the first place.
  - **Domain-wide-but-not-stage-conditional was considered and rejected** (the alternative
    surfaced during the audit: keep domain-wide standardization, add an explicit `compute_stage`
    main effect/interaction instead). Rejected because it doesn't fix the underlying
    non-commensurability of the *input* to the interaction term (`z_c * p_c` would still mix
    stage-scaled units before the stage effect could linearly correct for it), and it would add a
    coefficient to a model the thesis's own confirmatory pre-registration (Ch.5 Table 5.2) doesn't
    include, changing what's being tested rather than just how the input is scaled.
- **Implementation:** `src/analysis/inference.py::fit_h3_model` — `compute_stage` is required per
  row (rows missing it are dropped, same as missing signal/`y_optimal`); z is standardized via
  `groupby("stage")` mean/SD before the existing `z_c * p_c` GEE fit (unchanged: exchangeable
  `instance_key` clustering, pooled across stages for the regression itself — only the input
  scaling changed). A stage group with zero or undefined variance now fails loudly
  (`converged: False`) instead of silently producing a degenerate coefficient. Regression tests:
  `tests/analysis/test_inference.py::test_fit_h3_model_standardizes_per_stage_not_pooled` and
  `::test_fit_h3_model_converges_with_multi_stage_data`.
- **Date:** 2026-07-27.

## ADR-007: `h2_paired()` decides on the bootstrap CI bound, not the point estimate

- **Decision:** `h2_paired` now cluster-bootstraps `mean_success_diff` and `mean_log_token_diff`
  over resampled instances (5000 reps, the same engine used everywhere else) and bases
  `non_inferiority_holds`/`token_superiority_holds` on the **lower** bootstrap CI bound, not the
  raw paired-mean point estimate.
- **Context:** found during the pre-Phase-1-analysis cross-check against
  `../metacog-thesis/notes/praeregistrierung_auswertungsplan.md` and the current Ch.5 prose (§5.8:
  "H2 holds ... only when both one-sided bootstrap intervals satisfy their bounds"). The prior
  implementation compared the point estimate directly against the threshold and was never
  composed with `cluster_bootstrap` anywhere in the codebase — confirmed by grep, it wasn't called
  at all outside its own definition and one unit test. A point estimate crossing a threshold is a
  materially weaker claim than a CI bound crossing it, given the paired-cluster non-independence
  the rest of the inference engine exists to handle.
- **Sign convention:** both `succ_diff` (policy success − baseline success) and `log_tok_diff`
  (log baseline tokens − log policy tokens) are oriented larger-is-better, so both decision rules
  use the lower CI bound (`ci_low > -delta` / `ci_low > 0`), not one lower and one upper.
- **Regression test:** `tests/analysis/test_inference.py::
  test_h2_paired_decides_on_ci_bound_not_point_estimate` constructs a 10-instance case with
  `mean_success_diff == 0.0` (point estimate would say "holds", since `0.0 > -0.05`) but high
  instance-to-instance variance (arms alternate which one wins) that pushes the CI lower bound
  well below `-0.05`, so the correct decision is "does not hold."
- **Date:** 2026-07-28. Full suite green (376 tests at the time of this fix).

## ADR-008: Prompt/input-token tracking (`total_prompt_tokens`), Phase 2 onward

- **Decision:** track backend-reported prompt/input tokens per LM call, booked per candidate the
  same way output tokens (`total_tokens_generated`) already are, surfaced as `prompt_tokens` per
  step and `total_prompt_tokens` per episode. **Deliberately not retrofitted onto the already-
  collected Phase 1 data** (debug-view prompts are head/tail-truncated to 800 chars, not full, so
  it can't be reconstructed) — present from Phase 2 (Run 2) on.
- **Context:** `notes/praeregistrierung_auswertungsplan.md` §2.2 names "Total Tokens Processed pro
  Episode (Input + Output)" as a secondary DV; the episode schema only ever tracked the output
  side. Ch.5 §5.3 prose now states the phase asymmetry explicitly (see revision_audit P1-stat-7).
- **Where the number actually comes from:** the vLLM OpenAI-compatible server response already
  carries `usage.prompt_tokens` at the top level; `ServerBackend._post_chat` was reading `data` but
  only ever pulling `choices`, discarding `usage` entirely. Now extracted via
  `_prompt_tokens_from_usage` and threaded through.
- **Backward compatibility (the actual engineering problem here):** `generate()`/`generate_many()`
  are called via a frozen `text, logprobs = model.generate(...)` 2-tuple-unpacking contract at
  every call site, and ~45 test mocks across the suite return a bare 2-tuple. Changing the return
  arity would have broken all of them. Solution: `src/utils/inference/generate_result.py::
  GenerateResult` — an object whose `__iter__`/`__len__`/`__getitem__` all behave exactly like a
  2-tuple (so `text, logprobs = result` and every existing mock are unaffected), but which also
  carries `.prompt_tokens`, read via `getattr(result, "prompt_tokens", None) or 0` at call sites
  that want it. A mock returning a plain tuple has no such attribute, so `getattr` gracefully
  yields `None`/`0` — prompt-token tracking is backend-real-only by construction, never fabricated.
- **Per-candidate booking, not shared-prefix accounting:** a batched `n>1` request (C2 draws 3
  candidates in one HTTP call) reports one shared `usage.prompt_tokens` for the whole request, but
  it is attached **in full to each of the n candidates**, not divided — symmetric with how
  `reasoning_step_core` already books output tokens in full per candidate ("Compute is nevertheless
  booked for all three generations, which keeps the cost axis unbiased," Ch.5 §5.9). Getting this
  inconsistent between input and output would make batching look artificially cheaper on the input
  side than the output side for no principled reason.
- **`StepReturn` grew an 11th field** (`prompt_tokens_used`), appended at the end;
  `normalize_step_result` defaults it to `0` for any shorter/older tuple. Appending (not inserting)
  was deliberate: a few tests use `*_mid, call_detail = step(...)` catchall-unpacking that assumes
  `call_detail` is the *last* element — inserting in the middle would have silently broken that
  invariant instead of just changing arity (an explicit, loud `ValueError` at the 3 affected call
  sites, all fixed).
- **Compact storage gap caught and fixed in the same pass:** `src/utils/logging_utils.py::
  _MINIMAL_STEP_KEYS` is an explicit allowlist the production (default `compact=True`) storage path
  filters `steps_detail` through — `prompt_tokens` had to be added there too, or the field would be
  computed correctly in memory and then silently stripped before ever reaching disk. Caught by a
  dedicated regression test, `tests/utils/test_logging_utils_compact.py`, not by accident.
- **Scope boundary (deliberate, not an oversight):** `VLLMWrapper` and `LMStudioWrapper` (the
  in-process and LM-Studio backends, used for Gate C parity checks and local dev, never for
  production Phase 1/2 `--real` collection, which is `execution.backend_mode: server` only) are
  untouched — they still return plain tuples, so `prompt_tokens` is `0`/absent through them. Also
  untouched: `src/execution/metrics.py`'s run-level throughput reporting (`avg_tokens_by_stage`,
  `tokens_per_sec`) — that would need scheduler-level accumulation across a whole run, a separate
  feature from the per-episode DV this ADR is about.
- **Tests:** `tests/utils/test_generate_result.py` (the carrier object), 3 new cases in
  `tests/execution/test_execution_server.py` (usage extraction, missing-usage, batched-shared
  attribution), 3 new cases in `tests/agent/test_token_accounting.py` (C0/C2 per-episode sums,
  zero-default for non-reporting backends), `tests/utils/test_logging_utils_compact.py` (the
  compact-storage allowlist gap). Full suite green (387 tests) after 6 pre-existing tests were
  updated for the new (backward-compatible) arity — see commit for the list.
- **Date:** 2026-07-28.
