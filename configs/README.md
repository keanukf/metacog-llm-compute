# Configuration files

YAML configs live here. **Do not commit secrets** — use environment variables for API keys (`HF_TOKEN`, `LM_STUDIO_*`, `LANGFUSE_*`).

## File overview

| File | Purpose |
|------|---------|
| [`pilot.yaml`](pilot.yaml) | Pilot study (Tests 1–6, feasibility); primary file during pilot phase |
| [`experiment_core.yaml`](experiment_core.yaml) | Phase 1/2 core experiment (instances, domains, paths) |
| [`lmstudio_config.yaml`](lmstudio_config.yaml) | Optional deep-merge overlay when LM Studio enabled |
| [`dev/`](dev/) | Gate C/D diagnostic + smoke-run configs (parity, throughput, resume, format/VC probes) |

The single production model is `Qwen/Qwen3-8B`, hardcoded in `experiment_core.yaml`. The former
multi-model shortlist files (`models.yaml`, `models_runpod.yaml`), the `experiment_ext.yaml`
extension config, and the `prompt_variants/` A/B configs were removed in the 2026-07-21 refactor
along with the tooling that consumed them.

## `pilot.yaml` — keys you touch most

| Section | Keys | Notes |
|---------|------|-------|
| `model` | `name`, `dtype` | HF repo id; must match LM Studio API id in lmstudio mode. **Primary models:** RunPod/Phase 1/2 → `Qwen/Qwen3-8B` ([`experiment_core.yaml`](experiment_core.yaml)); local pilot → `Qwen/Qwen3-4B` ([`pilot.yaml`](pilot.yaml)) |
| `inference` | `backend`, `temperature`, `max_tokens`, `max_model_len`, `chat_template`, `enable_thinking` | `chat_template: true` required for instruct models; `max_model_len` lowers VRAM on 24GB GPUs |
| `inference` | `top_logprobs` | Top-k width for TLE (vLLM + LM Studio; default 20, EAGER-aligned) |
| `inference` | `lmstudio_base_url`, `lmstudio_api_key` | LM Studio: `reasoning.effort: none` (off) / `low` (on) on `/v1/responses`; `lmstudio_top_logprobs` is a deprecated alias for `top_logprobs` |
| `pilot` | `instances`, `compute_stages`, `runs_per_instance` | `compute_stages`: `3` or `C0` or `C0,C2` |
| `logging` | `logprob_sidecar_mode` (`off` \| `action_window` \| `full`), `logprob_sidecar_full_instances`, `save_step_traces`, `logprob_subdir`, `vc_subdir` | Sidecar volume; production default `action_window` |
| `tracing` | `langfuse_enabled` | Requires `pip install ".[tracing]"` (`langfuse>=3`) + env keys |
| `c1` / `c2` | `cot_*` (C1) / `n_samples`, `sample_temperature` (C2) | C1 single native thinking pass (`cot_*` caps); C2 self-consistency (`n_samples`, `sample_temperature`) |
| `domain_prompts` | `textworld`, `tower_of_hanoi` | Per-domain action prompts, stops, and optional `cot_max_tokens` (overrides `c1.cot_max_tokens`) |

**C1 `cot_max_tokens` (methodology):** Fixed per-domain caps (`domain_prompts.<domain>.cot_max_tokens`, fallback `c1.cot_max_tokens`) bound reasoning length for reproducibility. TextWorld and ToH use **2048** in `pilot.yaml` after RunPod smoke showed thinking blocks hitting the previous 1024 cap. Document the chosen caps and any truncation rate from pilot traces in thesis §5 (Methodology).
| `vc` | `followup_max_tokens`, … | VC follow-up call sizing |
| `episode` | `max_steps_per_episode` | Agent cap per episode (separate from game gen limit) |

## `execution` block (Phase 1/2 parallel runs)

| Key | Default | Notes |
|-----|---------|-------|
| `max_concurrent_episodes` | `1` | Thread-pool width; opt-in parallelism against shared `vllm serve` |
| `backend_mode` | `server` | `--real` uses HTTP server backend; `inprocess` reserved |
| `server_url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible base URL |
| `frozen_max_concurrent_episodes` | — | Freeze after TLE invariance validation; must match `max_concurrent_episodes` under `--real` |
| `frozen_tle_invariance_eps` | — | Paired with frozen N in `run_metadata.json` |

**Backpressure (RTX 5090, 32 GB):** effective concurrent sequences ≈ `max_concurrent_episodes` when `ServerBackend` posts are not serialized. C2 issues one `n=3` request per step when the server supports batched sampling. Size N so `N × KV footprint at max_model_len` fits KV budget; use throughput sweep + vLLM preemption logs to pick N.

See [`docs/runpod.md`](../docs/runpod.md) for plumbing smoke vs TLE invariance validation.

## `experiment_core.yaml` — Phase 1/2 RunPod keys

| Section | Keys | Notes |
|---------|------|-------|
| `model` | `name`, `revision`, `dtype` | Primary confirmatory model (`Qwen/Qwen3-8B`). Use `dtype: float16` — vLLM 0.19+ rejects `fp16`. |
| `inference` | `max_model_len`, `top_logprobs` | Cap context on 24 GB GPUs (default in file: `16384`). Required for `verify_backend_parity.py` and phase runners. |

Phase scripts and `scripts/instrument_validation/verify_backend_parity.py` load this file via `create_experiment_model()`; vLLM memory kwargs match `scripts/experiment/run_pilot.py` behaviour.
