# Instrument validation session log (Gate C)

Live log for RunPod 5090 instrument validation. Updated during the session.

| Field | Value |
|-------|-------|
| Host | RunPod 5090 (`213.173.111.21`, online 2026-07-13) |
| Started | 2026-07-12 |
| Branch (code) | `perf/vllm-server-concurrency` @ `3be18f0` (pod pulled post C-5/C-6 pre-declaration) |
| Results root | `/workspace/metacog-llm-compute/data/results/instrument_validation` |
| Python | `/root/venv-metacog/bin/python` |
| Production N | **N=32 (frozen)** — C-1 PASS; downstream gates run 2026-07-13 night |

## 2026-07-14 — C-5/C-6 sequence (post quest-DV, pre-freeze)

**Pre-declaration (before `signal_smoke` re-run):** H1a-style AUROC from this run is visible pre-freeze. **Permitted uses only:** (1) choose frozen **K** for TLE (C-6, both domains, same K); (2) diagnose whether TW TLE/VC AUROC is **no longer inverted** vs pre-DV `211029` (TW tle≈0.19 under score labels). **Not permitted:** threshold tuning, signal-definition changes, collapse switch, domain decisions, or Phase-1 go/no-go from smoke AUROCs alone.

**C-5 scope:** `phase1_20260713_211029` = **pipeline/sidecars Done**; TW signal discrimination **invalid** (score labels). Re-run @ 8192 + quest labels replaces signal arm and feeds C-6.

**In flight:** ~~`signal_smoke` @ `phase1_20260714_105004`~~ **Done** (72 ep, quest-DV, 8192, 47m 42s wall, ~93 ep/h @ max_in_flight=32, git `3be18f0`).

**C-5 `105004`:** Pipeline + sidecars **Done**. Signal arm **Done** (quest-DV labels).

**C-6:** **Blocked — AUROC reconciliation.** Alter Sweep (first Sidecar-Token, kein Action-Slice) wich von H1a ab (TW Δ≈0.06). Fix in Code; Re-Sweep pending. **K=20 unfreeze** bis Re-Sweep ≡ H1a.

**H1a AUROC (kanonisch: score = −TLE mean entropy; `signal_discrimination_report` nach Fix):**

| Domain | Collapse | Signal | AUROC | n_steps | n_positive |
|--------|----------|--------|------:|--------:|-----------:|
| TextWorld | optimal_only | TLE | **0.619** | 703 | **81** |
| TextWorld | optimal_only | VC | 0.611 | 703 | 81 |
| TextWorld | legal_or_optimal | TLE | 0.649 | 703 | 600 |
| TextWorld | legal_or_optimal | VC | 0.468 | 703 | 600 |
| ToH | optimal_only | TLE | 0.744 | 659 | 178 |
| ToH | optimal_only | VC | 0.701 | 659 | 178 |
| ToH | legal_or_optimal | TLE | 0.953 | 659 | 450 |
| ToH | legal_or_optimal | VC | 0.714 | 659 | 450 |

**DV-repair diagnostic:** TW `n_positive` 8→81; TW TLE H1a > 0.5 post-fix. `preanalysis_screen` TLE = raw entropy (legacy display); H1a uses **−entropy**.

**AUROC path reconciliation (2026-07-14):** Drei Pfade erklärt — (1) preanalysis raw + flip; (2) alter C-6 first-token; (3) H1a −mean_entropy action-window. „0.19" = ToH raw auf `211029` (~0.209), nicht C-6 TW. Fix: `slice_action_logprob_tokens` + sweep aligned to H1a.

**N @ `105004`:** `max_concurrent_episodes=32` (= parity freeze, `max_in_flight_observed=32`).

**C-1 freeze metadata:** **Done** — `backend_parity_20260714T104959Z.json`, `run_metadata.json` (`frozen_execution_params`: N=32, eps=0.05).

## 2026-07-13 — Gate C downstream complete @ N=32 (pod)

**VLLM:** OK | **signal_smoke throughput:** ~132 ep/h

| Gate | Result | Notes |
|------|--------|-------|
| C-1 parity | **PASS** | `backend_parity_20260713T205633Z.json`; gating max_dtle=0.008 |
| C-2/C-4 format_vc_probe | **C-2 pilot OK @ 8192** | Budget raster + admissibility fix; `phase1_20260714_083538` @ `7b1ef9f` — see 2026-07-14 section |
| C-3 toh_parse_probe | **PASS** | parse_rate=1.0 |
| C-5 signal_smoke | **Done** | `211029`: sidecars only (pre-DV); **`105004`**: full signal arm |
| C-6 topk sweep | **blocked — reconcile** | Fix deployed; Re-Sweep pending; K unfrozen |

**Open:** FREEZE-REVIEW §5.3 C2 (thesis repo); local disk full — 7/72 logprob sidecars not rsync'd (analysis ran on pod).

**Artifacts:** `phase1_20260713_211029`, `phase1_20260714_083538`, `phase1_20260714_100023`, **`phase1_20260714_105004`**, `backend_parity_20260714T104959Z.json`

## 2026-07-14 — TW quest-distance DV repair + label re-run

**Change:** `textworld_env.py` — `correctness` from `len(policy_commands)` (TextWorld quest solver), not score increase. `optimal` iff executable and `dist_after < dist_before` (strict). `score_progress_step` retained as descriptive side variable.

**Re-run:** `phase1_20260714_100023` — same `format_vc_probe` @ 8192, max_steps=10, 18 ep, 13 min. **Code anchor:** `9d994b8` / `6eb0f5d`. Run pre-commit on scp overlay semantically identical to commit (only `ruff format` whitespace in `_score_progress_step`; MD5 `ef39ff27…` vs `b37e9e9f…`).

| Domain | optimal | legal | illegal | unlabeled | n | **n_positive** |
|--------|--------:|------:|--------:|----------:|--:|-----------------:|
| TextWorld (new) | 37.8% | 34.4% | 27.8% | 0 | 90 | **34** |
| TextWorld (old score @ 083538) | 2.2% | 71.1% | 26.7% | — | 90 | **2** |
| ToH (unchanged) | 26.7% | 48.9% | 24.4% | 0 | 90 | **24** |

Run-to-run ToH shift vs 083538 (34.8% on n=89) is stochastic episode mix, not env change.

**Freeze notes (Thesis §5.9):** TW info-gathering actions (`look`/`inventory`/`examine`) are distance-neutral → `legal` under primary collapse; action class absent in ToH (full observability). Defensible H4 confound; motivates preregistered `optimal_or_legal` sensitivity. **Episode length:** §5.5 target 8–15 steps; probe suggests 15–20+ for TW — Phase-0 calibration, affects `position_norm` (H3). **Instance heterogeneity:** reset quest distances 7/4/7 across probe instances.

## 2026-07-14 — Budget raster + C2 admissibility fix (`7b1ef9f`)

**Pre-declared criterion:** ≥90% parseable thinking-closure per cell/domain on format_vc_probe (n≈30 steps/cell); no H2 metrics in decision path.

**Budget raster:** TW+ToH `cot_max_tokens` 1024 → 2048 → 4096 → **8192 (frozen)**. VC prompt fix (`711785e`: drop trailing `Confidence:`). C2 vote admissibility fix (`eac97cd`: vote only over closed+post_think samples).

**Final run:** `phase1_20260714_083538` (~10.5 min, 18 ep, 179 steps — one ToH C2 ep ended at 9 steps).

| Metric | TextWorld | Tower of Hanoi | Notes |
|--------|-----------|----------------|-------|
| `truncation_no_action` (C2) | **0/30** | **0/29** | Post-admissibility; meaningful gate metric |
| C1 thinking closed | **93.3%** (28/30) | **100%** (30/30) | Wilson ~78–98% @ n=30 — not separable from 90% |
| avg `n_samples_admissible` (C2) | **2.87** | **2.97** | Requested N=3; rejected samples still billed |
| VC parsed | **179/179** (100%) | same run | |
| `winner_closed` (C2) | — | — | **By construction 100% after `eac97cd`** — not evidence |

**TW C1 failures (2/30):** Both in `textworld_2_C1` steps 8–9; **`tokens_generated=8192` exactly** — hard cap hits mid-reasoning, no `</think>`. Not a regression vs 4096; run-to-run variance at T=0.5 + batched fp16. Belastbare C1-closure-Rate → Phase 1 full sample.

**Step-label distribution (8192 run, all stages):**

| Domain | optimal | legal | illegal | n |
|--------|--------:|------:|--------:|--:|
| TextWorld | 2.2% | 71.1% | 26.7% | 90 |
| ToH | 34.8% | 40.4% | 24.7% | 89 |

ToH is **not** collapsed near 90–95% optimal (y+ ≈35%); TW optimal rate very low in this probe — AUROC smoke here is format/ plumbing only. Episode success: ToH 2/9 (inst 1 under C1+C2); not a Gate-C-2 criterion.

**Prose follow-up (freeze-relevant):** §5.3 C2 must state: N=3 generations always emitted and token-billed; majority vote over **admissible** candidates only; effective N logged per step. Limitation → §5.9.

## 2026-07-14 — C2 thinking-closure metric comparison (pre-admissibility fix)

Source: `phase1_20260714_075754` @ 4096, **before** `eac97cd`.  
**Do not use `winner_closed` post-fix as empirical metric** (tautology after admissibility gate).

| Domain | Stage | step_any (≥1 sample) | winner_closed | all_samples (3/3) |
|--------|-------|---------------------:|--------------:|------------------:|
| TextWorld | C1 | 100% | 100% | 100% |
| TextWorld | C2 | 100% | 100% | 96.7% |
| ToH | C1 | 83.3% | 83.3% | 83.3% |
| ToH | C2 | 100% | **80.0%** | 60.0% |

Pre-fix ToH C2: 6/30 steps had unclosed **winner** despite another sample closed → motivated `eac97cd`.

## 2026-07-14 — C2 vote admissibility fix

**Bug:** Unclosed C2 samples could supply garbage vote keys (`_strip_think_blocks` no-op without close tag); winner could be non–committed-action → TLE measured at wrong tokens.

**Fix:** Vote only over admissible samples (closed thinking + `parse_method=post_think`). Log `n_samples_admissible`, `step_outcome`, `truncation_reason`. Rejected samples still count toward token/compute totals. Zero admissible → `truncation_no_action`, empty action, no VC follow-up.

**Limitation (§5.9):** C2 effective N is data-dependent after fix.

## 2026-07-13 — Backend perf fix (local)

**Branch:** `perf/vllm-server-concurrency` (`9987431`)

| Change | Detail |
|--------|--------|
| ServerBackend lock | Removed global HTTP lock — parallel episodes can hit vLLM concurrently |
| C2 `generate_many` | Single OpenAI `n=` request with sequential fallback |
| Default timeout | 600s (`execution.server_timeout_s` override in YAML) |
| vLLM serve flags | Documented: `--gpu-memory-utilization 0.92`, `--max-num-seqs 32`, `--max-num-batched-tokens 8192`, `--enable-prefix-caching` |
| Tests | 278 passed locally |

**Invalidates:** C-1 parity PASS from 2026-07-12 (batch probe ran through serialized client). Re-run parity after pod pull + re-sweep.

**Pod command (when online):**

```bash
bash scripts/run_instrument_validation_after_perf.sh
```

Or stepwise: see `docs/runpod.md` § Instrument validation session.

## 2026-07-13 — C-1 scoped waiver + N=32 freeze (design sign-off)

**Decision (explicit user sign-off):** Adopt **N=32** as frozen production `max_concurrent_episodes` and scope the batch-invariance gate to committed-action-representative probes. `minimal_3` is retained as a reported diagnostic but no longer gates.

**Rationale (evidence-based):**

- TLE is measured **only at committed-action tokens** (signal contract). The probes that represent that window pass batch invariance by a wide margin at every N:

  | Probe | represents | worst `dtle_mean` @ N=32 | vs eps=0.05 |
  |-------|-----------|--------------------------|-------------|
  | `tw_short` | TextWorld action | 0.0002 | ~250× under |
  | `toh_short` | ToH move | 0.0078 | ~6× under |
  | `minimal_1` | constrained action | 0.0010 | far under |
  | `minimal_2` | constrained action | 0.0021 | far under |
  | `minimal_3` (diagnostic) | free-form ramble | 0.091 | over — **not gating** |

- `minimal_3`'s prompt (`"Action: inventory"`) is **underspecified**: the model emits free-form multi-token text instead of a single committed action. Its under-load `dtle_mean` (0.09–0.15, non-monotonic in N) is driven by **generation-sequence divergence**, not committed-action-token numerics — confirmed by `dtle_max` (peak-token deviation) staying ~0.005 while the sequence-mean shifts. This load pattern never occurs in the real agent loop, which always prompts for a single committed action (like `tw_short`/`toh_short`).
- Residual batch non-invariance in free generation is a known property of batched vLLM inference; forcing true batch invariance (`--enforce-eager` + batch-invariant kernels) would cost large throughput for no gain on the committed-action signal.

**Code change (reproducible gate, not a prose-only waiver):**

- `data/probes/parity_prompts.json`: added `gating`/`role`/`note` fields; `minimal_3` → `gating: false`.
- `src/execution/parity.py`: `run_batch_invariance_probe` now gates on `gating`-true probes only; reports all probes plus `diagnostic_max_dtle`/`diagnostic_worst_constellation` for transparency. Default `gating=True` (backward compatible).
- Tests: added `test_non_gating_probe_drift_does_not_fail_gate`; full suite 281 passed.

**Frozen params:** `(N=32, eps=0.05 bits)`, `eps_derived_under_load=True`. Both Phase 1 and Phase 2 run at N=32 (batch effect is N-dependent; a single frozen N keeps calibration and deployment consistent).

**Limitations note for thesis:** report `minimal_3` diagnostic transparently as evidence of batched-inference nondeterminism; state that the committed-action TLE window is invariant within eps under production concurrency.

## 2026-07-13 — Post-perf Gate C re-run (pod online)

**Host:** RunPod 5090 (`213.173.111.21:39260`)  
**Branch:** `perf/vllm-server-concurrency` @ `cd40dde`  
**vLLM:** running on `:8000` (health OK)

### Throughput re-sweep (post perf fix)

| N | ep/h | wall_s | max_in_flight | smoke |
|---|-----:|-------:|--------------:|-------|
| 8 | 169.6 | — | 12 | GO |
| 16 | 178.9 | — | 12 | GO |
| 24 | 186.8 | — | 12 | GO |
| 32 | 192.4 | — | 12 | GO |

Artifact: `data/results/instrument_validation/throughput_sweep_post_perf.json`

Sweep recommended N=32 (highest ep/h). **Not adopted** — batch invariance failed at all tested N.

### C-1 backend parity (post perf fix) — **FAIL / HARD STOP**

K-coverage and temperature invariance pass at all N. Batch invariance fails at every production N tested:

| N | max_dtle | eps | worst probe | worst constellation | Artifact |
|---|--------:|----:|-------------|---------------------|----------|
| 32 | 0.091 | 0.05 | `minimal_3` | `pool32_long` | `backend_parity_20260713T201008Z.json` |
| 24 | 0.154 | 0.05 | `minimal_3` | `pool24_long` | `backend_parity_20260713T202215Z.json` |
| 16 | 0.133 | 0.05 | `minimal_3` | `pool16_long` | `backend_parity_20260713T202234Z.json` |
| 8 | 0.130 | 0.05 | `minimal_3` | `pool8_long` | `backend_parity_20260713T202259Z.json` |

Domain probes (`tw_short`, `toh_short`) stay well under eps at all N. Failure is isolated to synthetic `minimal_3` under long-filler concurrent load (`pool*_long`, 96 filler tokens). Even `pool2_long` exceeds eps for `minimal_3` (dtle ≈ 0.037–0.087).

**Production N:** **blocked** — no N satisfies batch invariance at eps=0.05. Configs left at **N=16** (fallback attempt; best throughput/invariance tradeoff among failures is N=32 at max_dtle=0.091).

**Downstream gates skipped (hard stop):** format_vc_probe, audit_pilot_signals, preanalysis_screen, toh_parse_probe, signal_smoke, sweep_topk_sensitivity.

**Next actions to unblock C-1:**
1. Investigate vLLM batching / prefix-caching interaction under mixed-length concurrent requests (`minimal_3` + long fillers).
2. Re-run parity with prefix caching disabled or `--max-num-batched-tokens` adjusted.
3. Consider whether `minimal_3` long-filler constellation is representative of Phase 1 load (domain probes pass).
4. If acceptable as limitation: document and proceed at N=32 with max_dtle=0.091 (~1.8× eps) — requires explicit design sign-off.

## Steps (cumulative — authoritative)

| Step | Gate | Status | Notes |
|------|------|--------|-------|
| Pod setup | C-0 | **done** | vLLM :8000, Qwen3-8B, `experiment_core.yaml` |
| Throughput @ N=32 | — | **done** | ~192 ep/h post-perf (`throughput_sweep_post_perf.json`) |
| Backend parity | C-1 | **PASS + frozen** | `backend_parity_20260714T104959Z.json`; `run_metadata.json` N=32, eps=0.05 |
| format_vc_probe | C-2/C-4 | **PASS @ 8192** | `083538` budget/admissibility; `100023` post-DV labels |
| toh_parse_probe | C-3 | **PASS** | parse_rate=1.0 |
| signal_smoke | C-5 | **Done** | `105004` @ quest-DV + 8192; `211029` sidecars-only legacy |
| topk sweep | C-6 | **blocked — reconcile** | Re-Sweep after sweep/H1a alignment fix |

## Throughput sweep pre-fix (2026-07-12 — superseded)

| N | ep/h | wall_s | max_in_flight | smoke |
|---|-----:|-------:|--------------:|-------|
| 1 | 130.5 | 331 | 1 | GO |
| 3 | 129.5 | 334 | 3 | GO |
| 6 | 136.9 | 316 | 6 | GO |
| 8 | 138.8 | 311 | 8 | GO |

Flat ~88–92 tok/s indicated HTTP serialization, not GPU saturation.

## Gate F budget (preliminary, pending C-5 ep/h)

| Item | Value |
|------|-------|
| Phase 1 episodes | 1,500 |
| Phase 2 episodes | 3,000 |
| Total | 4,500 |
| Pre-fix ep/h (post C1/C2 token fix, 8-step smoke) | ~27–30 |
| Rough wall-time pre-fix | ~150–170 h |
| Target post perf-fix | Re-measure; expect multi× if tok/s scales with real batching |

Formula: `wall_hours ≈ total_episodes / measured_ep_per_hour` (from C-5 or throughput probe at production step count).

## 2026-07-12 session detail (archive)

### C-1 backend parity (invalidated)

```
K-coverage: PASS
Temperature invariance: PASS (eps=0.05 bits)
Batch invariance: PASS (max_dtle=0.033, eps=0.05 bits)
```

### C-2 / C-4 format_vc_probe (`phase1_20260712_123103`)

- Pre thinking-budget fix (`4ed7de6`); TextWorld C1 FAIL; VC 60.6%

### C-3 toh_parse_probe (`phase1_20260712_124245`)

- parse_rate=1.000 — PASS

## Artifacts

| Artifact | Path |
|----------|------|
| Throughput sweep (pre-fix) | `data/results/instrument_validation/throughput_sweep.json` |
| Backend parity (invalidated) | `data/results/instrument_validation/backend_parity_20260712T123057Z.json` |
| Post-perf sweep | `data/results/instrument_validation/throughput_sweep_post_perf.json` |
