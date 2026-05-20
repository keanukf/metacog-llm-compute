# ADR Log (Key Harness Decisions)

## ADR-001: C1 CoT->Verify split

- **Decision:** Keep C1 as two explicit calls (draft CoT parse + verification pass).
- **Rationale:** Improves controllability, explicit error modes, and traceability of draft correction.

## ADR-002: LM Studio logprobs via `/v1/responses`

- **Decision:** Use LM Studio `/v1/responses` for token logprobs when available.
- **Rationale:** OpenAI-compat chat/completions often omit usable logprobs for TLE.

## ADR-003: Episode artifact schema version marker

- **Decision:** Add `schema_version` to episode JSON writes (`episode.v1` baseline).
- **Rationale:** Enables safe loader evolution and explicit migration semantics.
