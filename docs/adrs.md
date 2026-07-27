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
