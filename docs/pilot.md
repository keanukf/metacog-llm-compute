# Pilot runner (`run_pilot.py`)

The pilot runs Tests 1–6 in sequence and writes benchmark and calibration outputs. See [`configs/pilot.yaml`](../configs/pilot.yaml) and [`configs/README.md`](../configs/README.md).

**Related:** [`docs/runbook.md`](runbook.md) (mock checklist), [`docs/runpod.md`](runpod.md) (CUDA), [`docs/scripts.md`](scripts.md) (all scripts).

## Pilot modes

| Mode | Flag | Hardware | Backend | Use case |
|------|------|----------|---------|----------|
| **Pilot 0** | `--pilot-mode mock` (default) | None | Mock | Quick local sanity; CI; no model download. |
| **Pilot 1** | `--pilot-mode lmstudio` | LM Studio host (LAN or localhost) | `POST /v1/responses` (TLE logprobs) | Local Mac or LAN LM Studio (`LM_STUDIO_BASE_URL`, default `http://localhost:1234/v1`). |
| **Pilot 2** | `--pilot-mode cuda` | NVIDIA GPU (e.g. RunPod) | vLLM | Validate GPU setup and throughput before Phase 1/2. |

## Pilot 0 — Mock (default)

No real model, stub environments. Confirms the script and output format:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results
```

You get per-step JSON under `--output-dir` (e.g. `pilot_test1_inference.json`, `pilot_test2_tle.json`, …) plus `ep_textworld_*.json` / `ep_tower_of_hanoi_*.json` for episodes and `pilot_feasibility.json`. In **mock** mode, metrics are synthetic (e.g. unrealistic `tokens_per_sec`).

**Optional granular logprobs.** Set `logging.logprob_sidecar_mode` in `configs/pilot.yaml` (`full` for pilot calibration, `action_window` for production Phase 1/2) to write per-env-step completion token rows as sidecar files under `logging.logprob_subdir` (default `logprobs/`), e.g. `{episode_id}_logprobs.json`, without bloating the main episode JSON. The deprecated key `save_logprob_distributions` is rejected — use `logprob_sidecar_mode` instead.

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

## Pilot 1 — LM Studio (lmstudio)

On a Mac or LAN host with LM Studio running, use the responses API path (required for TLE token logprobs).

**macOS:** Do not install `vllm` locally (`pip install -r requirements.txt` skips it on Darwin). Use `pip install -r requirements-local.txt` or `pip install -e ".[dev]"` plus the packages below.

```bash
pip install lmstudio httpx
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode lmstudio
```

`--pilot-mode hf` / `m1` were removed; use **lmstudio** locally. Step traces and `debug_views/` include per-call `lmstudio` diagnostics (`route`, `status`, logprobs presence).

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

**Thinking mode:** This repo forces **model-native thinking ON for C1 and C2 sample calls**; **C0** and **VC follow-up** calls force thinking OFF.

**LM Studio + Qwen3 checklist** (if reasoning/action parsing lacks `` blocks or thinking leaks into action lines):

1. **Model id** — Use hybrid **`Qwen/Qwen3-4B`** (or LM Studio `qwen/qwen3-4b`), not **`Qwen3-4B-Instruct-2507`** (non-thinking-only snapshot).
2. **API path** — C1 uses `POST /v1/responses` with `include: message.output_text.logprobs`. If logprobs are empty, the wrapper used to fall back to raw `/v1/completions` (no chat template); check traces for that warning.
3. **Response shape** — LM Studio often returns `output[].type == "reasoning"` plus `message`, not inline think tags. The wrapper reassembles `` + answer text for the C1 parser.
4. **Thinking off (API)** — The wrapper sends:
   - `reasoning: { "effort": "none" }` on the wire (Open Responses; LM Studio maps to model reasoning **off**)
   - `enable_thinking: false` and `chat_template_kwargs: { "enable_thinking": false }`
   For C1/C2 reasoning (`enable_thinking=true`): `reasoning.effort: "low"` (not `"medium"` — Qwen3 dev log warns that only model **on**/**off** exist and coerces `medium` → `on`). Do not send `"on"`/`"off"` in JSON; the HTTP API rejects them.
   Probe: `python scripts/probe_lmstudio_thinking_toggle.py --model qwen/qwen3-4b`
5. **GUI fallback** — **Developer → Inference → Custom Fields → Enable Thinking** off, or `defaultValue: false` in `model.yaml` for `enableThinking`.
6. **Smoke test** — To check single-call C1 action parsing on your endpoint:


```bash
python scripts/run_c1_handoff_gate.py --config configs/pilot.yaml --pilot-mode lmstudio --real --domain textworld --n-episodes 3 --max-steps 5
```

### Output handling contract (C1)

For reproducibility and debugging, C1 uses a strict single-call input/output contract:

1. **Reason call (`enable_thinking=true`)**
   - **Input:** `<task> + <history> + <state>` plus instruction to reason inside `<think>...</think>` and output one command after `</think>`.
   - **Preferred format:** include `<command>...</command>` at the end of the think block.
   - **Output consumed:** `cot_parser.parse_cot_action(...)` / `_normalize_action_line(...)` extracts:
     - `<command>...</command>` (preferred),
     - else first plausible line after `</think>`,
     - else conservative fallbacks.
   - **Result:** parsed action line; no `<draft_action>` hint, no verify fallback (single LM call only).

2. **VC follow-up (`enable_thinking=false`)**
   - **Input:** `<task_context>`, `<output_to_judge>`, and confidence instruction.
   - **Output consumed:** first line only (`stop=["\\n"]`) parsed as 0-100 confidence.


### C2: Self-Consistency (Majority Vote — not C1×3 + Verifier)

Per [`blueprints/thesis_design.md`](../blueprints/thesis_design.md), **C2 is not** “three CoT+Verify chains with an LLM judge.” It is:

1. **Three parallel reasoning samples** with thinking ON at `c2.sample_temperature` (same action prompt template as C0, but `enable_thinking: true`).
2. **Majority vote** over normalized action keys (`majority_vote` in `src/agent/stages/c2.py`).
3. **Optional VC follow-up** on the winning action (same modes as C0/C1).

There is **no** separate verify subcall in C2. Inspect traces: `call_detail.method == self_consistency_majority_vote`, three `subcalls` with `kind: sample`, and `c2_vote` in compact debug views.

**TLE semantics:** TLE is computed from the **action-token slice** of the stage call — C0/C1: the single stage call; C2: the **winning sample** (`extract_action_tle_from_response` in `src/signals/token_entropy.py`). If LM Studio returns text but no logprobs, `tle` is `null` for that step by design. This affects telemetry quality, not action execution.

**Token-level entropy (TLE):** Configure top-k width via `inference.top_logprobs` (default **20**, EAGER-aligned). When `logprobs=True`, both backends return per-token candidate lists; TLE is Shannon entropy over the **renormalized top-k** distribution (approximation vs full vocabulary).

- **LM Studio:** Uses **`POST /v1/responses`** (LM Studio **0.4.x+**) with `include: ["message.output_text.logprobs"]` and `top_logprobs` from config. The usual OpenAI-compat `/v1/completions` and `/v1/chat/completions` endpoints do not return usable logprobs.
- **vLLM (CUDA):** Uses `SamplingParams(logprobs=K)` for top-k width; pins `logprobs_mode="raw_logprobs"` on the **engine** (`LLM(...)`) for temperature-invariant TLE. Records are normalized to the same schema as LM Studio before `compute_tle`.

If only top-1 logprobs are available, TLE falls back to legacy binary entropy per token.

**Step-level observability:** With `logging.save_step_traces: true` (default in `configs/pilot.yaml`), each episode writes `trace_{episode_id}.jsonl` next to the episode JSON: one line per env step with full prompt, full raw model output, observations, and `history_snapshot` (including prior `ACTION: ...` lines so the model cannot “forget” what it did). For Langfuse cloud traces, install `pip install ".[tracing]"` (requires `langfuse>=3`), set `tracing.langfuse_enabled: true`, and export `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (optional `LANGFUSE_HOST` for EU). Phase 1/2 runners pass the same options when enabled in YAML.

**Langfuse observation tree (per env step):**

| Stage | Hierarchy under `step_{n}` |
|-------|----------------------------|
| C0 | `action_{n}_C0` → optional `vc_followup_{n}` (sibling under step when VC enabled) |
| C1 | `reason_{n}_C1` → optional `vc_followup_{n}` (VC only when follow-up runs) |
| C2 | `sample_{i}_{n}_C2` siblings under step → optional `vc_followup_{n}` under step |

Trace-level fields use `session_id`, `tags`, and `trace_name` (episode id). Step timing lives on the step span; TLE is a numeric **score** on the action/reason generation, VC on the VC generation, correctness on the step span. Token counts and temperatures are sent as `usage_details` / `model_parameters` on each generation, not only in metadata.

### Debug views (`debug_views/`)

After each pilot, Phase 1, or Phase 2 run (and after `run_c1_handoff_gate.py`), the repo writes a **compact JSON summary** under `debug_views/` when `logging.write_debug_views: true` (default). This is a human-readable view of the inference pipeline per environment step — **including truncated prompts and responses** — without opening multi-megabyte `trace_*.jsonl` files.

| File | Purpose |
|------|---------|
| `debug_views/run_summary.json` | Run-level index: episode list, step counts, parse-method histogram, empty-action count |
| `debug_views/episode_{episode_id}.json` | Per-episode `steps[]` with `pipeline` blocks |

Each step’s `pipeline` may include:

- **`primary`** — C0 (or combined prompt/response for the step)
- **`reason`** — C1 single-call reasoning (truncated prompt/response, `gen` params)
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

## Backend logprob parity (Phase 1 prerequisite)

Before confirmatory Phase 1 runs on vLLM, run:

```bash
python scripts/verify_backend_parity.py --backend vllm --config configs/experiment_core.yaml
```

Production TLE uses vLLM **engine** `logprobs_mode="raw_logprobs"` (temperature-invariant:
logprobs before temperature / top_k / top_p). LM Studio `/v1/responses` may use a
different logprob scale or quantization (e.g. local MLX 4-bit vs pod float16 8B) — cross-backend entropy
equality is only required when the **same** model and precision run on both backends. See
`data/results/backend_parity_*.json`.

**Parity gate (§5.7):** Pass when K-coverage ≥ 20 and first-token `|dTLE(T=0.3 vs T=1.0)| ≤ eps`
(bits), with `eps = max(0.05, 3 × same-T noise floor)`. The script also records a Same-T control
(T=1.0 vs T=1.0) and predicted temperature-scaling spans for diagnostics. Bit-identical logprob
JSON comparison is intentionally **not** used (fp16 non-determinism between separate requests).

**RunPod / vLLM config notes (required for the script above):**

- `model.dtype` must be `float16` (or `bfloat16`). vLLM 0.19+ rejects the alias `fp16` in
  `LLM(..., dtype=...)`.
- Set `inference.max_model_len` (e.g. `16384` in `configs/experiment_core.yaml`) on 24 GB GPUs.
  Qwen3-8B advertises 40960 context; without a cap, vLLM fails at init with a KV-cache OOM.
  Phase runners use the same keys via `create_experiment_model()` in `src/utils/experiment_env.py`.
