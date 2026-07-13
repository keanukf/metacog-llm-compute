# Instrument validation session log (Gate C)

Live log for RunPod 5090 instrument validation. Updated during the session.

| Field | Value |
|-------|-------|
| Host | `only_emerald_roundworm` |
| Started | 2026-07-12 |
| Branch | `cursor/instrument-validation-prep` @ `664bf68` |
| Results root | `/workspace/metacog-llm-compute/data/results/instrument_validation` |
| Python | `/root/venv-metacog/bin/python` |
| Production N | **8** (from throughput sweep) |

## Steps

| Step | Gate | Status | Notes |
|------|------|--------|-------|
| Pod setup | C-0 | **done** | Deploy key; venv on container disk; vLLM TRITON_ATTN :8000 |
| Plumbing smoke | C-0 | **GO** | ~14 min; max_in_flight=3; ~91 tok/s |
| Throughput N-sweep | — | **done** | All candidates GO; **N=8** best (138.8 ep/h); artifact `throughput_sweep.json` |
| Config N=8 | — | **done** | `experiment_core.yaml` + dev configs updated on pod |
| Backend parity | C-1 | **PASS** | K-coverage, temp invariance, batch invariance (max_dtle=0.033, eps=0.05); `backend_parity_20260712T123057Z.json` |
| format_vc_probe | C-2/C-4 | **PARTIAL / BLOCKER** | 18 ep in `phase1_20260712_123103`; see blockers below |
| toh_parse_probe | C-3 | **PASS** | 20× C0 ToH; **parse_rate=1.0** (400/400 steps); `phase1_20260712_124245` |
| signal_smoke | C-5 | **skipped** | Blocked by C-2 TextWorld + C-4 VC rate |

## Throughput sweep (2026-07-12T12:28:51Z)

| N | ep/h | wall_s | max_in_flight | smoke |
|---|-----:|-------:|--------------:|-------|
| 1 | 130.5 | 331 | 1 | GO |
| 3 | 129.5 | 334 | 3 | GO |
| 6 | 136.9 | 316 | 6 | GO |
| 8 | **138.8** | 311 | 8 | GO |

Chosen production N=8. Token throughput ~88–92 tok/s flat across N (GPU not saturated; higher N improves wall-time via batching).

## C-1 backend parity

```
K-coverage: PASS
Temperature invariance: PASS (eps=0.05 bits)
Batch invariance: PASS (max_dtle=0.033, eps=0.05 bits)
```

## C-2 / C-4 format_vc_probe (`phase1_20260712_123103`)

- Run: 18 episodes (2×3 inst×3 stages), ~10 min, max_in_flight=8, ~88 tok/s
- **C-2 TextWorld C1: FAIL** — 0/30 C1 steps produced a parseable game action; model output truncated to opening `<think>` tag only (see `debug_views/episode_ep_textworld_*_C1_*.json`)
- **C-2 ToH C1: partial** — 12/30 steps parsed valid moves; 18/30 think-tag-only
- **C-4 VC: below target** — vc_rate **60.6%** (109/180); gate target ≥90%; validate_pilot_outputs FAIL
- preanalysis_screen (smoke-only AUROC flags): ToH tle_auroc=0.540, vc_auroc=0.693; TextWorld vc mostly missing (79%)
- C2 self-consistency: 60/60 steps with majority_vote, 3 samples each — OK

## C-3 toh_parse_probe (`phase1_20260712_124245`)

- 20 C0 episodes, 3 disks, ~56 s wall, max_in_flight=8
- **parse_rate=1.000** (400/400 steps with `action_parsed`) — **PASS** (>80%)

## Blockers before signal_smoke (C-5)

1. **TextWorld C1 format compliance** — thinking never completes; action slot receives `<think>` literal. Likely causes: `action_max_tokens` / stop sequence interaction, or missing `chat_template`/thinking config for TextWorld C1 on server backend.
2. **VC parse rate 60.6%** — below C-4 threshold (90%). Follow-up often returns `"Confidence:"` echo instead of integer.

**Recommended next actions (user GO required):**

- Diagnose TextWorld C1 truncation (inspect raw completions in traces; try raising C1 token budget or `chat_template: true` if not set)
- Re-run `format_vc_probe` after config fix; only then launch `signal_smoke.yaml` (~72 ep, extrapolate ~35 min at observed ~104 ep/h for 18-ep probe)

## Findings

- **Workspace volume ~10GB quota:** use `/root/venv-metacog` + `/root/.cache/huggingface`, not workspace `.venv`.
- **preanalysis_screen.py:** f-string bug fixed locally + on pod during session (invalid nested format specifiers in markdown/print paths).
- **Attention backend:** TRITON_ATTN; re-run parity if switched to FLASHINFER.

## Artifacts

| Artifact | Path |
|----------|------|
| Throughput sweep | `data/results/instrument_validation/throughput_sweep.json` |
| Backend parity | `data/results/instrument_validation/backend_parity_20260712T123057Z.json` |
| format_vc_probe | `data/results/instrument_validation/phase1_20260712_123103/` |
| toh_parse_probe | `data/results/instrument_validation/phase1_20260712_124245/` |
| Logs | `/root/logs/{throughput_sweep,backend_parity,format_vc_probe,toh_parse_probe}.log` |
