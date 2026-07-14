# FREEZE-REVIEW: §5.4 / §5.6 — stage-wise ECDF allocator

**Status:** Approved for implementation (2026-07-14)  
**Scope:** Code + consistency log only; thesis chapter prose updated separately.

## Decision

Replace the single pooled Phase-1 holdout ECDF with **three stage-wise ECDFs** (C0, C1, C2), one per compute stage. Thresholds θ₁ and θ₂ remain on the **percentile scale** and stay **identical across stages**; grid search logic is unchanged except that each step's raw signal is mapped through the ECDF of the stage that **produced** that signal.

This is a **design correction** (constructing a valid allocator), not an empirical fallback. If adaptive allocation underperforms in Phase 2, that remains a signal finding — not an artifact of comparing incomparable raw scales.

## Mechanistic rationale (preregistration-safe)

After a native reasoning trace (C1/C2), the token distribution over the committed action is **collapsed**: the decision was largely fixed in the thinking block. TLE in C1/C2 therefore measures **confidence in verbalizing an already-chosen action**; in C0 it measures **uncertainty over the action itself**. These are different quantities on different scales.

A pooled ECDF violates the cross-stage comparability assumption underlying §5.4 percentile thresholds: C1/C2 values sit near ~1e-6 while C0 sits at ~0.02–0.07 (descriptive entropy distribution on `105004`; not used as a hypothesis test). Stage-wise ECDF restores comparability by ranking each step **within its own stage's holdout distribution**. The percentile is then stage-independent by construction.

**Not covered by raw_logprobs / temperature invariance:** The scale shift arises in **reasoning**, not in decoding temperature.

## §5.4 specification change (allocator contract)

| Element | Before | After |
|---------|--------|-------|
| ECDF reference | One pooled holdout sample per domain/signal | Three holdout ECDFs: `ecdf_by_stage["C0"|"C1"|"C2"]` |
| Percentile at runtime | `F_pooled(x)` | `F_{stage}(x)` where `stage` = compute stage of the **previous** step |
| θ₁, θ₂ | Percentile cutoffs | Unchanged (same values, now meaningful cross-stage) |
| Grid search | Pooled ECDF | Stage-appropriate ECDF per holdout row's `compute_stage` |

Artifact schema: policy JSON uses `ecdf_by_stage`. Legacy `ecdf_ref`-only artifacts **fail to load** unless `allow_legacy_pooled_ecdf=True` (pilot-only opt-in with warning; Phase 2 path never sets this flag).

## §5.6 note (measurement mechanism)

Document in thesis §5.6: post-reasoning action entropy collapse is a **measurement-structure** effect. AUROC remains rank-based and can discriminate within each stage even when magnitudes collapse; a pooled ECDF threshold cannot, because it compares stages on a common raw axis where C1/C2 sit at the floor.

## Empirical context (descriptive only — does not justify the design change)

Within-stage AUROC on `phase1_20260714_105004` (`optimal_only`, pilot n):

- Signal **order** is preserved in every stage (TW/ToH C1/C2 TLE ≈ 0.59–0.72; ranks discriminate despite ~1e-6 medians).
- Pooled ToH TLE AUROC (0.744) vs C2 within-stage (0.594) illustrates pooling confound — not used to select this fix.
- `n_positive` per cell is 19–28 in five of six cells → AUROC CIs ~±0.10; treat table as “signal exists, order preserved”, not fine-grained stage effects. TW C0 TLE 0.534 is **not** secured as near-chance.
- **H1a tension (result, not action):** VC beats TLE at C0 in both domains (TW 0.607 vs 0.534; ToH 0.790 vs 0.678). No design change triggered — note for Phase 1 confirmatory analysis.

## Open point — §5.9 sensitivity arm

ToH C2 under `legal_or_optimal` collapse: `n_neg = 1` on 72 pilot episodes. Preregistered sensitivity analysis runs under both collapses; if a cell degenerates under the secondary collapse, that arm is not evaluable. Re-check with full Phase-1 instance count. If ToH-C2 rarely plays illegal moves, the secondary collapse may be **structurally degenerated** for that domain/stage — document in §5.9.

## Implementation (this repo)

- `src/agent/allocation_policy.py` — `ecdf_by_stage`, `stage(x, source_stage=...)`
- `src/analysis/thresholds.py` — `build_ecdf_ref_by_stage`, grid search uses row `compute_stage`
- `src/agent/allocator.py` — `signal_source_stage` from previous step
- `src/agent/base_agent.py` — tracks `signal_source_stage` in adaptive loop

## Gate sequence after merge

1. Unit tests green  
2. Re-freeze **K = 20** (C-6 reconciled; independent of ECDF fix)  
3. Close Gate C
