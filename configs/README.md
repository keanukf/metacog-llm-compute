# Configuration files

YAML configs live here. **Do not commit secrets** — use environment variables for API keys (`HF_TOKEN`, `LM_STUDIO_*`, `LANGFUSE_*`).

## File overview

| File | Purpose |
|------|---------|
| [`pilot.yaml`](pilot.yaml) | Pilot study (Tests 1–6, feasibility); primary file during pilot phase |
| [`experiment_core.yaml`](experiment_core.yaml) | Phase 1/2 core experiment (instances, domains, paths) |
| [`experiment_ext.yaml`](experiment_ext.yaml) | Extended experiment settings |
| [`models.yaml`](models.yaml) | Default model list for `run_pilot_models.py` |
| [`models_runpod.yaml`](models_runpod.yaml) | RunPod model shortlist + `hf_model_card_gate.py` |
| [`lmstudio_config.yaml`](lmstudio_config.yaml) | Optional deep-merge overlay when LM Studio enabled |
| [`prompt_variants/`](prompt_variants/) | A/B prompt configs for `run_prompt_ab.py` |

## `pilot.yaml` — keys you touch most

| Section | Keys | Notes |
|---------|------|-------|
| `model` | `name`, `dtype` | HF repo id; must match LM Studio API id in lmstudio mode |
| `inference` | `backend`, `temperature`, `max_tokens`, `max_model_len`, `chat_template`, `enable_thinking` | `chat_template: true` required for instruct models; `max_model_len` lowers VRAM on 24GB GPUs |
| `inference` | `lmstudio_base_url`, `lmstudio_api_key`, `lmstudio_top_logprobs` | LM Studio: `reasoning.effort: none` (off) / `low` (on) on `/v1/responses` |
| `pilot` | `instances`, `compute_stages`, `runs_per_instance` | `compute_stages`: `3` or `C0` or `C0,C2` |
| `logging` | `save_logprob_distributions`, `save_step_traces`, `logprob_subdir`, `vc_subdir` | Sidecar volume; off for long runs unless needed |
| `tracing` | `langfuse_enabled` | Requires `pip install ".[tracing]"` (`langfuse>=3`) + env keys |
| `c1` / `c2` | `cot_*`, `verify_*`, `n_samples` | C1 CoT+verify; C2 self-consistency N |
| `domain_prompts` | `textworld`, `tower_of_hanoi` | Per-domain action prompts, stops, and optional `cot_max_tokens` (overrides `c1.cot_max_tokens`) |
| `vc` | `followup_max_tokens`, … | VC follow-up call sizing |
| `episode` | `max_steps_per_episode` | Agent cap per episode (separate from game gen limit) |

See [`docs/pilot.md`](../docs/pilot.md) for CLI flags and [`blueprints/infrastructureplan_pilot.md`](../blueprints/infrastructureplan_pilot.md) for pilot design rationale.
