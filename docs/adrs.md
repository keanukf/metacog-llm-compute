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
