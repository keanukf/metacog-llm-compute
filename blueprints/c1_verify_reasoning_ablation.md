# C1 Verify reasoning-visible ablation (optional)

This repository’s default C1 design keeps the verify call **action-centered** to protect TLE measurement integrity at Call 2. In particular, the verify prompt does **not** include the chain-of-thought (CoT) from Call 1, to avoid anchoring / continued-reasoning confounds.

If you want to study *process-verification* as an ablation (reasoning-visible verification), treat it as a **separate experimental branch** and do not mix the resulting runs with the main Phase-1 calibration data.

## Proposed config flag

Add a YAML flag (default off):

```yaml
c1:
  verify_show_reasoning: false
  verify_reasoning_max_chars: 500
```

## Intended behavior (when `verify_show_reasoning: true`)

- The verify prompt includes an additional block:

```text
<draft_reasoning>
  ... bounded / truncated reasoning from Call 1 ...
</draft_reasoning>
```

- The instruction explicitly labels it as diagnostic-only context and forbids copying it.
- The final output format remains unchanged and continues to use the single-line output instruction constant:
  `Output exactly one command on a single line. No reasoning, no tags, no preamble.`

## Reporting guidance

When this ablation is enabled, report results separately, because:

- The verify call is no longer an independent verification pass.
- Anchoring can reduce action-token uncertainty (TLE) via prompt scaffolding rather than calibrated metacognition.
- C0 vs. C1 comparability at the measurement point becomes ambiguous.

