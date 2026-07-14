# Instrument validation session log (Gate C)

Live log for RunPod 5090 instrument validation. Updated during the session.

| Field | Value |
|-------|-------|
| Host | RunPod 5090 (`213.173.111.21`, online 2026-07-13) |
| Started | 2026-07-12 |
| Branch (code) | `perf/vllm-server-concurrency` @ `f725139` |
| Results root | `/workspace/metacog-llm-compute/data/results/instrument_validation` |
| Python | `/root/venv-metacog/bin/python` |
| Production N | **N=32 (frozen)** — C-1 PASS; downstream gates run 2026-07-13 night |

## 2026-07-13 — Gate C downstream complete @ N=32 (pod)

**VLLM:** OK | **signal_smoke throughput:** ~132 ep/h

| Gate | Result | Notes |
|------|--------|-------|
| C-1 parity | **PASS** | `backend_parity_20260713T205633Z.json`; gating max_dtle=0.008 |
| C-2/C-4 format_vc_probe | **PASS** (2026-07-14) | VC prompt fix + unified `cot_max_tokens=4096` both domains; `phase1_20260714_075754`: vc=100%; TW C1/C2 think 100%; ToH C1 83%, C2 100% |
| C-3 toh_parse_probe | **PASS** | parse_rate=1.0 |
| C-5 signal_smoke | **PASS (run)** | 72 ep, 0 errors; AUROC interpretable; TW tle=0.71, ToH tle=0.21 |
| C-6 topk sweep | **blocked** | sidecar schema drift — fix in progress locally |

**Open:** C-1 freeze metadata re-run; C-6 topk on pod after pull.

**Note (2026-07-14):** In the 2026-07-13 format_vc_probe run, Tower of Hanoi instances were **solved under C1 and C2** on Qwen3-8B — a meaningful step up from local Qwen3-4B, which often stalled on final moves. Unified **`cot_max_tokens=4096`** for both domains after Gate C-2/C-4 re-runs.

**Artifacts:** `phase1_20260713_205837`, `phase1_20260713_210804`, `phase1_20260713_211029`

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

## Steps (cumulative)

| Step | Gate | Status | Notes |
|------|------|--------|-------|
| Pod setup | C-0 | **done** (2026-07-12) | Deploy key; venv on container disk; vLLM TRITON_ATTN :8000 |
| Plumbing smoke | C-0 | **GO** (2026-07-12) | ~14 min; max_in_flight=3 |
| Throughput N-sweep (pre-fix) | — | **done** | N=8 @ 138.8 ep/h — **superseded** by serialized-client artifact |
| Backend parity | C-1 | **PASS (scoped)** (2026-07-13) | Committed-action batch invariance PASS @ N=32; `minimal_3` diagnostic-only; K-coverage + temp PASS |
| format_vc_probe | C-2/C-4 | **FAIL vs 90%** | TW C1 clean 73%, VC 63%; thinking budget trunc |
| toh_parse_probe | C-3 | **PASS** | parse_rate=1.0 |
| signal_smoke | C-5 | **PASS (run)** | 72 ep; AUROC interpretable; ~132 ep/h |
| topk sweep | C-6 | **fix pending** | sidecar schema drift in sweep script |
| Throughput re-sweep | — | **done** (2026-07-13) | N=32 @ 192.4 ep/h; see post-perf table below |

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
