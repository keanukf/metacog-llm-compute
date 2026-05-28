# Pilot runner (`run_pilot.py`)

The pilot runs Tests 1–6 in sequence and writes benchmark and calibration outputs. See [`configs/pilot.yaml`](../configs/pilot.yaml) and [`configs/README.md`](../configs/README.md).

**Related:** [`docs/runbook.md`](runbook.md) (mock checklist), [`docs/runpod.md`](runpod.md) (CUDA), [`docs/scripts.md`](scripts.md) (all scripts).

## Pilot modes

| Mode | Flag | Hardware | Backend | Use case |
|------|------|----------|---------|----------|
| **Pilot 0** | `--pilot-mode mock` (default) | None | Mock | Quick local sanity; CI; no model download. |
| **Pilot 1** | `--pilot-mode hf` | Mac M1/M2 (Apple Silicon) | HuggingFace + MPS | Test full pipeline locally before buying GPU. |
| **Pilot 2** | `--pilot-mode cuda` | NVIDIA GPU (e.g. RunPod) | vLLM (or HF) | Validate GPU setup and throughput before Phase 1/2. |
| **Pilot 3** | `--pilot-mode lmstudio` | LM Studio host (LAN or localhost) | OpenAI-compatible HTTP | Local or LAN LM Studio (`LM_STUDIO_BASE_URL`, default `http://localhost:1234/v1`). |

## Pilot 0 — Mock (default)

No real model, stub environments. Confirms the script and output format:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results
```

You get per-step JSON under `--output-dir` (e.g. `pilot_test1_inference.json`, `pilot_test2_tle.json`, …) plus `ep_textworld_*.json` / `ep_tower_of_hanoi_*.json` for episodes and `pilot_feasibility.json`. In **mock** mode, metrics are synthetic (e.g. unrealistic `tokens_per_sec`).

**Optional granular logprobs.** Set `logging.save_logprob_distributions: true` in `configs/pilot.yaml` to write per-env-step completion token rows (including `top_logprobs` when the backend returns them) as sidecar files under `logging.logprob_subdir` (default `logprobs/`), e.g. `{episode_id}_logprobs.json`, without bloating the main episode JSON. Use `logprob_export_format: csv` or `both` for a long tidy table; CSV includes `p_renorm_topk` (softmax over the returned top-k at export time). Size grows with steps × tokens × top-k—keep off for long runs unless needed. Test 2 can also emit `pilot_test2_tle_distributions.json` (and optional CSV) in that subdirectory. Phase 1 (`run_phase1.py`) honors the same `logging.*` keys next to checkpoints.

Run individual steps without the full pipeline:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --only test2
```

`--only` accepts one or more of: `sanity`, `test1`, `test2`, `test3`, `test4`, `test5`, `feasibility` (executed in that order). For `feasibility`, missing inputs are filled from JSON already present in `output_dir` when available.

After a run, aggregate TextWorld / ToH episode JSONs (success rate by stage, mean TLE, VC spread, optional ECE proxy):

```bash
PYTHONPATH=. python scripts/summarize_pilot_calibration.py data/results/pilot_<UTC>/
```

Optional validation:

```bash
python scripts/validate_pilot_outputs.py --pilot-dir data/results/pilot_<UTC>/
```

## Pilot 1 — HuggingFace on Apple Silicon (hf)

On a Mac with M1/M2/M3, run the same pipeline with the real model via HuggingFace on Metal (MPS). No CUDA or vLLM required:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode hf
```

`--pilot-mode m1` is accepted as a deprecated alias for `hf`. For the same machine, **lmstudio** is often faster because inference runs inside LM Studio instead of raw Transformers.

Requires `transformers`, `torch` with MPS support, and enough RAM/Unified Memory for the model (e.g. Qwen2.5-3B). Slower than Pilot 2 but catches setup and integration errors before spending on cloud GPU.

## Pilot 2 — Real CUDA GPU

On a machine with CUDA (e.g. RunPod RTX 3090), use vLLM for real inference and measured throughput:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode cuda
```

Or use `--real` to auto-detect: if CUDA is available → cuda; else if MPS (Mac) → hf; else mock.

Full RunPod workflow: [`docs/runpod.md`](runpod.md).

## Pilot 3 — LM Studio

LM Studio exposes an OpenAI-compatible API (often `http://localhost:1234/v1` or a LAN address). Set `model.name` to the **exact model identifier** shown in LM Studio for API requests:

```bash
export LM_STUDIO_BASE_URL="http://192.168.178.173:1234/v1"
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode lmstudio
```

Or use `inference.lmstudio_base_url` / `inference.lmstudio_api_key` in YAML; default API key is `lm-studio` if unset (`LM_STUDIO_API_KEY`). Requires the `openai` package.

**Thinking mode (C1 only):** This repo forces **model-native thinking ON only for the C1 CoT subcall**; C1 verify, C0, C2, and VC follow-up calls force thinking OFF.

**LM Studio + Qwen3 checklist** (if CoT/verify lack `` blocks or verify still “thinks” aloud):

1. **Model id** — Use hybrid **`Qwen/Qwen3-4B`** (or LM Studio `qwen/qwen3-4b`), not **`Qwen3-4B-Instruct-2507`** (non-thinking-only snapshot).
2. **API path** — C1 uses `POST /v1/responses` with `include: message.output_text.logprobs`. If logprobs are empty, the wrapper used to fall back to raw `/v1/completions` (no chat template); check traces for that warning.
3. **Response shape** — LM Studio often returns `output[].type == "reasoning"` plus `message`, not inline think tags. The wrapper reassembles `` + answer text for the C1 parser.
4. **`enable_thinking=False`** — On some LM Studio builds, verify still emits a `reasoning` block; toggle **Developer → Inference → Custom Fields → Enable Thinking** off in the app, or set `defaultValue: false` in the model’s `model.yaml` (`enable_thinking` Jinja variable).
5. **Smoke test** — To check CoT→verify parsing on your endpoint:


```bash
python scripts/run_c1_handoff_gate.py --config configs/pilot.yaml --pilot-mode lmstudio --real --domain textworld --n-episodes 3 --max-steps 5
```

### Output handling contract (C1)

For reproducibility and debugging, C1 uses a strict input/output contract:

1. **CoT call (`enable_thinking=true`)**
   - **Input:** `<task> + <history> + <state>` plus instruction to output one command after `</think>`.
   - **Preferred CoT format:** include `<command>...</command>` at the end of the think block.
   - **Output consumed:** `cot_parser.parse_cot_action(...)` extracts:
     - `<command>...</command>` (preferred),
     - else first plausible line after `</think>`,
     - else conservative fallbacks.
   - **Result:** `draft_action`, `draft_status`, `parse_method`.

2. **Verify call (`enable_thinking=false`)**
   - **Input:** same base context; optional `<draft_action>` hint when CoT parsed an action.
   - **Required output:** exactly one action line.
   - **Parsing:** `normalize_action_line(...)` strips think blocks, rejects instruction echoes (e.g. `Just the command.`), then tries conservative embedded-action recovery.
   - **Fallback policy (minimal):**
     - primary = parsed verify action
     - if verify is non-action and `draft_action` exists: use `draft_action`
     - no extra multi-stage verify retries

3. **VC follow-up (`enable_thinking=false`)**
   - **Input:** `<task_context>`, `<output_to_judge>`, and confidence instruction.
   - **Output consumed:** first line only (`stop=["\\n"]`) parsed as 0-100 confidence.

**TLE semantics:** TLE is computed from verify-call logprobs only. If LM Studio returns text but no logprobs, `tle` is `null` for that step by design. This affects telemetry quality, not action execution.

**Token-level entropy (TLE) with LM Studio:** The usual OpenAI-compat endpoints (`/v1/completions`, `/v1/chat/completions`) do not return logprobs in LM Studio. For `logprobs=True`, the code uses **`POST /v1/responses`** (LM Studio **0.4.x+**) with `include: ["message.output_text.logprobs"]` and `top_logprobs` (see `inference.lmstudio_top_logprobs` in config, default 5). TLE is Shannon entropy over the **renormalized top-k** distribution per token (approximation vs full vocabulary). vLLM/HF still use top-1 logprobs and the legacy binary-entropy path.

**Step-level observability:** With `logging.save_step_traces: true` (default in `configs/pilot.yaml`), each episode writes `trace_{episode_id}.jsonl` next to the episode JSON: one line per env step with full prompt, full raw model output, observations, and `history_snapshot` (including prior `ACTION: ...` lines so the model cannot “forget” what it did). For Langfuse cloud traces, install `pip install ".[tracing]"`, set `tracing.langfuse_enabled: true`, and export `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (optional `LANGFUSE_HOST` for EU). Phase 1/2 runners pass the same options when enabled in YAML.

### Debug views (`debug_views/`)

After each pilot, Phase 1, or Phase 2 run (and after `run_c1_handoff_gate.py`), the repo writes a **compact JSON summary** under `debug_views/` when `logging.write_debug_views: true` (default). This is a human-readable view of the inference pipeline per environment step — **including truncated prompts and responses** — without opening multi-megabyte `trace_*.jsonl` files.

| File | Purpose |
|------|---------|
| `debug_views/run_summary.json` | Run-level index: episode list, step counts, parse-method histogram, empty-action count |
| `debug_views/episode_{episode_id}.json` | Per-episode `steps[]` with `pipeline` blocks |

Each step’s `pipeline` may include:

- **`primary`** — C0 (or combined prompt/response for the step)
- **`cot`** / **`verify`** — C1 subcalls (truncated prompt/response, `gen` params, verify `parse` metadata such as `parse_method`, `draft_action`, `fallback_source`)
- **`vc_followup`** — VC confidence follow-up when present
- **`c2_samples`** / **`c2_vote`** — C2 self-consistency samples and majority-vote summary
- **`final`** — parsed action, truncated observations, last few `history_tail` lines

Truncation uses `logging.debug_view_head_chars` and `debug_view_tail_chars` (default 800 each): each string is stored as `{head, tail, length, sha256_prefix, truncated}`.

**Machine truth** remains in `trace_*.jsonl`. Debug views are for inspection and RunPod download review.

Regenerate for an existing run folder:

```bash
python scripts/build_debug_views.py --run-dir data/results/pilot_<UTC>/
```

If `write_debug_views` is true but `save_step_traces` is false, runners enable step traces automatically (debug views require trace JSONL as input).

- **Test 1:** Real prompts → measured tokens/s, latency, VRAM (CUDA only).
- **Tests 2–3:** With a real model, TLE and VC are evaluated on **real generations**; in mock mode they use synthetic/static checks.
- **Test 4:** TextWorld e2e episodes (stub env unless a real game file is wired in).
- **Test 5:** Tower of Hanoi parseability (C0; count from `configs/pilot.yaml`).
- **Feasibility:** Go/No-Go checklist and ECE (from TextWorld episodes), written to `pilot_feasibility.json`.

After a non-mock pilot, `pilot_test1_inference.json` includes realistic `tokens_per_sec` for that device/endpoint.

## Multi-model batch (`run_pilot_models.py`)

Run the full pilot once per model id with one command. Each run writes under `data/results/pilot_batch_<UTC>/pilot_<UTC>_<slug>/` and optional `pilot_batch_manifest.json` lists `model_id`, output path, exit code, and wall time. For **L0.3** (local LM Studio spot-checks of several GGUF/MLX candidates), use `--pilot-mode lmstudio --real` and load each model in LM Studio (or point `LM_STUDIO_BASE_URL` at the right server) before the corresponding subprocess. For **L0.4** (RunPod vLLM model-selection benchmark), use `--pilot-mode cuda --real` — this repo loads **one HF repo id per subprocess** (no separate vLLM server swap required). Example:

```bash
python scripts/run_pilot_models.py --config configs/pilot.yaml --pilot-mode lmstudio --real \
  --models "id1,id2"
# or: --models-file path/to/models.yaml  (list of strings or models: [ ... ])
# Default list: edit configs/models.yaml and omit --models / --models-file to use it
```

## Tower of Hanoi (manual play)

Interactive sanity check without a model: [`scripts/play_tower_of_hanoi.py`](../scripts/play_tower_of_hanoi.py). See [`docs/scripts.md`](scripts.md).

```bash
python scripts/play_tower_of_hanoi.py
```

Defaults: 3 disks, seed 42. Flags: `--num-disks`, `--seed`, `--partial-moves`, `--max-steps`.
