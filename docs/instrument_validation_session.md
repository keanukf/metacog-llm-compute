# Instrument validation session log (Gate C)

Live log for RunPod 5090 instrument validation. Updated during the session.

| Field | Value |
|-------|-------|
| Host | `only_emerald_roundworm` (offline 2026-07-13) |
| Started | 2026-07-12 |
| Branch (code) | `perf/vllm-server-concurrency` @ `9987431` |
| Results root | `/workspace/metacog-llm-compute/data/results/instrument_validation` |
| Python | `/root/venv-metacog/bin/python` |
| Production N | **TBD** — re-sweep after ServerBackend concurrency fix |

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

## Steps (cumulative)

| Step | Gate | Status | Notes |
|------|------|--------|-------|
| Pod setup | C-0 | **done** (2026-07-12) | Deploy key; venv on container disk; vLLM TRITON_ATTN :8000 |
| Plumbing smoke | C-0 | **GO** (2026-07-12) | ~14 min; max_in_flight=3 |
| Throughput N-sweep (pre-fix) | — | **done** | N=8 @ 138.8 ep/h — **superseded** by serialized-client artifact |
| Backend parity | C-1 | **INVALIDATED** | 2026-07-12 PASS; must re-run on real batched load |
| format_vc_probe | C-2/C-4 | **RE-RUN PENDING** | Pre `4ed7de6` thinking fix + pre perf fix |
| toh_parse_probe | C-3 | **RE-RUN PENDING** | Was 1.0 on 2026-07-12 |
| signal_smoke | C-5 | **pending** | After C-2/C-4 pass |
| Throughput re-sweep | — | **pending** | `8,16,24,32` after perf branch on pod |

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
| Post-perf sweep | `data/results/instrument_validation/throughput_sweep_post_perf.json` (pending) |
