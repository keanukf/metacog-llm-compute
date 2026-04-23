---
name: runpod_smoke_test_and_model_list
overview: Prepare and run a medium-size RunPod smoke test (on-demand 4090) using vLLM in-process, with pinned dependency installs, and a curated shortlist of dense instruct models up to ~9B parameters (your literature definition treats <=10B as “small models”), excluding MoE/hybrid. Validate each candidate against Hugging Face model cards/metadata before GPU runs, then collect the same pilot artifacts per model and compare hard criteria (logprobs, parseability, VC parseability, signal discrimination proxies).
todos:
  - id: deps_versioning
    content: "Make RunPod install reproducible: use requirements.txt pins (avoid unpinned setup_cloud.sh installs) and verify vllm/transformers versions on the pod."
    status: in_progress
  - id: hf_model_card_gate
    content: For each candidate repo id, pull Hub metadata + README/model card via Hugging Face MCP (`hub_repo_details`, optional `hf_hub_query`) and record a short compliance checklist (dense vs MoE/hybrid, instruct/chat template, thinking-mode flags, license/gating, expected transformers/vLLM constraints).
    status: pending
  - id: model_shortlist
    content: Finalize the candidate model list (dense instruct, <=~9B for this smoke-test shortlist; literature SLM cap <=10B, no MoE/hybrid), including exact Hugging Face repo IDs for each, after the HF model-card gate passes.
    status: pending
  - id: single_model_gpu_sanity
    content: (You on RunPod) Run one full pilot on 4090 with a baseline model to validate vLLM logprobs + artifacts end-to-end; paste stdout + key JSON paths back into chat for review.
    status: pending
  - id: batch_run
    content: (You on RunPod) Run scripts/run_pilot_models.py for the shortlist at medium scale (TW 2×, ToH 10 eps) with continue-on-fail; keep the batch folder path for comparison.
    status: pending
  - id: results_review
    content: (Together) Review batch outputs (or you upload summaries) and select a winner for GATE 0; document why (hard criteria).
    status: pending
isProject: false
---

## Scope decisions (locked)

- **GPU**: On-demand/persistent (RTX 4090).
- **Scale**: Medium.
  - TextWorld: 2 instances × C0/C1/C2 × 1 run
  - Tower of Hanoi: 10 episodes (C0)

## Who runs what (important)

- **This plan is prep + checklists + exact commands.** I cannot rent/start a RunPod GPU from here, so I also cannot truthfully “execute” the 4090 smoke test end-to-end in this environment.
- **You run the GPU portion on RunPod** (SSH into the pod, venv, installs, `run_pilot.py` / `run_pilot_models.py`). After each step, paste the command + last ~30 lines of output + the new `data/results/...` folder name; I can then help debug/version-pin/logprob issues quickly.

## Constraints from codebase / blueprint

- `scripts/run_pilot.py` with `--pilot-mode cuda` uses **in-process vLLM** via `src/utils/model_wrapper.py::VLLMWrapper` (no vLLM server required).
- Requirements currently specify:
  - `vllm>=0.18.0`
  - `transformers>=4.56.0,<5`
  - `openai>=1.0.0` (used for LM Studio path)
- `scripts/setup_cloud.sh` currently does **unpinned** `pip install vllm transformers ...` and pre-downloads `Qwen/Qwen3.5-4B` (hybrid; should be excluded per dependency map for L0.4).

## Model shortlist (dense instruct; SLM <=10B in your literature; smoke-test focuses <=~9B; no MoE/hybrid)

Target: models that are plausibly strong enough for simple ToH + cooking TextWorld, while staying inside your literature definition of **“small models” as <=10B parameters**, and staying conservative on the RunPod smoke-test shortlist by focusing on **<=~9B** candidates (still dense, instruct-tuned, no MoE/hybrid).

Primary candidates (HF IDs; verify via model-card gate before running):

- `Qwen/Qwen3-8B-Instruct` (primary “Qwen3 @ 8B instruct” anchor)
- `Qwen/Qwen2.5-7B-Instruct` (dense Qwen2.5 instruct baseline; often very stable on vLLM)
- `meta-llama/Llama-3.1-8B-Instruct` (dense Llama instruct baseline)
- `google/gemma-2-9b-it` (9B instruct-tuned Gemma 2; included now that ~9B is explicitly desired)

Explicitly **exclude** (thesis + interpretability reasons unless you rewrite the rationale):

- MoE models (e.g. Mixtral family)
- Hybrid / non-standard attention stacks called out in `blueprints/thesis_dependency_map.html` (e.g. Qwen3.5 DeltaNet-style hybrids)
- Vision-language checkpoints for a text-only thesis setup (e.g. `Qwen/Qwen3-VL-8B-Instruct`)

## Hugging Face verification (before any GPU $)

Use the Hugging Face MCP server `plugin-huggingface-skills-huggingface-skills` (read-only) with the correct tool schemas:

- `hub_repo_details` with `repo_ids: ["org/name", ...]` (max 10 per call) to fetch canonical Hub metadata for each candidate.
- Optionally `hf_hub_query` with a tight natural-language question if you need relationship/discovery queries beyond raw repo metadata.

Minimum “model card gate” checklist to record per `repo_id` (copy/paste bullets into your run notes / GATE 0 appendix):

- **Architecture class**: dense decoder-only vs MoE vs hybrid (reject MoE/hybrid per thesis map)
- **Chat/instruct**: confirm it is instruction-tuned (not base) and has a chat template / recommended usage
- **Thinking / chain-of-thought toggles**: if the card mentions “thinking mode”, `/think`, etc., document the **non-thinking** invocation you will use for all pilot prompts (your pipeline is single-line actions; thinking models are a footgun)
- **Licensing / gating**: gated models need `HF_TOKEN` on the pod for weights download
- **Runtime constraints**: any README warnings about `transformers` min version, `trust_remote_code`, tokenizer quirks, or “requires flash-attn” (note: your `VLLMWrapper` already sets `trust_remote_code=True` in `src/utils/model_wrapper.py`)

Note: the earlier “Model Details” MCP error was simply missing the required string field; for Hub reads in this plan, prefer `hub_repo_details` (schema requires `repo_ids`).

## RunPod smoke-test procedure

### 1) Pod setup (repeatable + pinned)

- Clone repo and create a clean venv.
- Install dependencies using the repo pins (avoid unpinned `setup_cloud.sh` behavior):
  - `pip install -r requirements.txt`
  - Optionally `pip install -e .` if you want import consistency.
- Verify versions:
  - `python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"`

### 2) Single-model sanity on GPU

Run one pilot end-to-end for the primary anchor first:

- `python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode cuda --real --model-name "Qwen/Qwen3-8B-Instruct"`

If vLLM load fails (VRAM / kernel / unsupported arch), fall back in this order:

1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`

Acceptance criteria:
- `pilot_sanity.json`: `has_logprobs: true`
- `pilot_test1_inference.json`: reasonable tok/s for 4090 (order-of-magnitude check)
- `pilot_test2_tle.json`: non-null TLE; easy vs hard differs
- `pilot_test3_vc.json`: VC parse-rate acceptable
- ToH parse-rate ≥80%
- Sanity-check qualitative behavior vs the 4B run: ToH should show **non-trivial** movement toward goal on at least some episodes under diversified starts (your configs already support diversified ToH generation in `configs/pilot.yaml`).

### 3) Multi-model batch

Use `scripts/run_pilot_models.py`:

- Put the shortlist into `configs/models.yaml` (or pass `--models` inline).
- Run:
  - `python scripts/run_pilot_models.py --config configs/pilot.yaml --pilot-mode cuda --real --continue-on-fail`

This writes a `pilot_batch_*/pilot_batch_manifest.json` plus one subfolder per model.

### 4) Compare outputs and decide next step

For each model folder, extract and compare:
- `pilot_sanity.json` (logprobs availability)
- `pilot_test3_vc.json` (VC parseability)
- `pilot_test5_toh.json` (parse_rate + behavioral metrics)
- Presence of `logprobs/` sidecars and `trace_*.jsonl`

Pick top-1 model for GATE 0 documentation (dependency map), then proceed to the larger L0.4 benchmark if needed.

## Notes on the HuggingFace MCP error

If you still want a “Model Details”-style prompt, ensure the tool invocation includes the required string field(s) exactly as that tool’s JSON schema demands. For this workflow, **`hub_repo_details` is the canonical Hub read** (requires `repo_ids`).
