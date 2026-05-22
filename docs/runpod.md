# Running on RunPod Cloud GPU

The pilot with `--real` is designed for **~2 GPU-hours** on an **RTX 3090** (24 GB VRAM). Budget: about **$0.44** (see [`blueprints/infrastructureplan_pilot.md`](../blueprints/infrastructureplan_pilot.md) Section V).

**Related:** [`docs/pilot.md`](pilot.md) (pilot modes), [`docs/scripts.md`](scripts.md), [`configs/models_runpod.yaml`](../configs/models_runpod.yaml).

## Step 1 — Create & start the pod (dashboard settings)

- **GPU**: RTX 4090 (or 3090-class), **count = 1**
- **Template/Image**: RunPod **PyTorch 2.x**
  - PyTorch **2.4.0 vs 2.8.0**: either works; this repo installs pinned Python deps from `requirements.txt` anyway. If unsure, **2.4.0 is fine**.
- **Pricing**:
  - **On‑Demand**: not preemptible
  - **Spot** (if available): cheaper but can be preempted
- **Storage**:
  - **Container Disk** (ephemeral root disk): pick large enough for **one** 7–9B model download/cache at a time (rule of thumb **≥ 50 GB**, “no surprises” **~100 GB**).
  - **Network Volume** (persistent, mounted at `/workspace`): **10 GB is enough for results** (they’re MB-scale).

## Step 2 — Connect via SSH (interactive)

Use the SSH command shown in the RunPod UI (gateway SSH is fine for interactive work).

## Step 3 — Quick disk check (+ decide what persists)

On the pod:

```bash
df -h
ls -la /workspace
```

Expected:

- `/` reflects your **Container Disk** size (ephemeral)
- `/workspace` is mounted (your **Network Volume**, persistent)

If you want **results persistent** but **models ephemeral**, set:

```bash
# Keep results on the Network Volume:
export RESULTS_DIR="/workspace/metacog-llm-compute/data/results"

# Keep HF cache on the container disk (default behavior; make it explicit anyway):
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
```

## Step 4 — Hugging Face token (recommended)

Some checkpoints are **gated** on the Hub. On the pod:

```bash
export HF_TOKEN="..."   # read-only token is enough for downloads
```

This also improves rate limits for metadata scripts like `scripts/hf_model_card_gate.py`.

**Optional — Langfuse tracing (recommended if you want cloud traces)**

These secrets are **not committed** (don’t put them in git). Set them on the pod before running pilots:

```bash
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
# Optional (EU): export LANGFUSE_HOST="https://eu.cloud.langfuse.com"
```

If you prefer a file instead of re-exporting every session, create a local env file on the pod and source it:

```bash
cd /workspace/metacog-llm-compute
nano .env   # add HF_TOKEN / LANGFUSE_* exports (or KEY=VALUE lines)
set -a && source .env && set +a
```

## Step 5 — Clone the repo on the pod

This repo is `keanukf/metacog-llm-compute`. On RunPod, the most reliable way is cloning via **GitHub SSH**.

```bash
# On the pod:
cd /workspace

# 1) Create a new SSH key for this pod (one-time per pod)
ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_ed25519_runpod
cat ~/.ssh/id_ed25519_runpod.pub
# 2) Add that public key in GitHub → Settings → SSH and GPG keys

# 3) Sanity check (must say "Hi keanukf!")
ssh -T -i ~/.ssh/id_ed25519_runpod git@github.com

# 4) Clone using the key explicitly (important: forces the right key)
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_runpod -o IdentitiesOnly=yes' \
  git clone git@github.com:keanukf/metacog-llm-compute.git metacog-llm-compute

cd /workspace/metacog-llm-compute
```

Use `/workspace` so that code and results live on the network volume and persist across pod restarts.

## Step 6 — Set up the environment on the pod

On the pod, from the repo root:

```bash
cd /workspace/metacog-llm-compute
bash scripts/setup_cloud.sh
```

This installs the **pinned** dependency set from `requirements.txt` (includes `vllm`, `transformers`, `textworld`, `numpy`, `pandas`, `scipy`, `pyyaml`, test deps, etc.).

**Optional — pre-download the model** (saves time during the pilot; `scripts/setup_cloud.sh` does this unless `SKIP_MODEL_DOWNLOAD=1`. If `MODEL_NAME` is unset, it uses the **first** model in `configs/models_runpod.yaml`):

```bash
export MODEL_NAME="Qwen/Qwen3-8B"
export SKIP_MODEL_DOWNLOAD=0
bash scripts/setup_cloud.sh
```

Store the model on the network volume (e.g. under `/workspace`) so it persists.

## Step 7 — Hugging Face model-card gate (before burning GPU time)

Quick, read-only Hub scan for obvious exclusion flags (MoE-ish language, multimodal/VL language, “thinking mode” keywords):

```bash
cd /workspace/metacog-llm-compute
python scripts/hf_model_card_gate.py --models-file configs/models_runpod.yaml
```

## Step 8 — Run unit tests on the pod (no GPU needed)

Same as locally; all tests use mocks:

```bash
cd /workspace/metacog-llm-compute
pip install pytest pytest-cov  # if not already installed by setup_cloud.sh
python -m pytest tests/ -v
```

All tests should pass (run `pytest` locally to see the current count). This confirms the codebase and interfaces before you run the real pilot.

## Step 9 — Run the pilot with real model (GPU workload)

Run the pilot **with real inference** on the pod (Pilot 2):

```bash
cd /workspace/metacog-llm-compute
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir /workspace/metacog-llm-compute/data/results --pilot-mode cuda --real
```

Or use `--real` to auto-detect (on the pod this will select `cuda`).

### Troubleshooting: weird model outputs (blank actions / prompt echo / VC always null)

- **Model outputs empty actions (0 tokens) / stops immediately**
  - Ensure `configs/pilot.yaml` has `inference.chat_template: true` (required for instruct/chat models like Qwen3).
  - For TextWorld, avoid stopping on a single newline: use `domain_prompts.textworld.action_stop: ["\n\n"]` (or remove `action_stop` entirely).
- **Model echoes prompt fragments (e.g. “Do not use disk numbers.”)**
  - This is also a strong signal that the chat template is not applied. Keep `inference.chat_template: true`.
- **VC is always `null`**
  - Increase `vc.followup_max_tokens` (default is small). In `configs/pilot.yaml` we use **24**.
  - If the VC follow-up output contains words instead of a number, it will parse as `null`.
- **Thinking text floods the action**
  - Keep `inference.enable_thinking: false` for baseline runs; enable only in dedicated A/B variants.

**If vLLM fails during model init with a KV-cache / max-seq-len error (common on 24 GB GPUs):**

Some models advertise very large context lengths (e.g. 40960). On 24 GB GPUs, vLLM may fail with a message like “KV cache needed … larger than available …”. In that case set a smaller `inference.max_model_len` in `configs/pilot.yaml` (e.g. `8192` or `16384`) and rerun.

You should see a new timestamped folder like `data/results/pilot_YYYYMMDD_HHMMSS/` containing at least:

- `run_info.json` — resolved config + model id + pilot mode
- `pilot_sanity.json`, `pilot_test1_inference.json`, `pilot_test2_tle.json`, `pilot_test3_vc.json`, `pilot_test5_toh.json`, `pilot_feasibility.json`
- TextWorld + ToH episode JSONs (`ep_*.json`) plus optional `trace_*.jsonl`, `logprobs/`, `vc/`

## Multi-model batch (recommended for model selection)

```bash
cd /workspace/metacog-llm-compute
python scripts/run_pilot_models.py --config configs/pilot.yaml --pilot-mode cuda --real \
  --models-file configs/models_runpod.yaml --continue-on-fail
```

Afterwards, summarize the batch folder with:

```bash
python scripts/summarize_pilot_batch.py data/results/pilot_batch_YYYYMMDD_HHMMSS
```

## Prompt A/B testing (small, fast)

This repo includes a small A/B harness to compare prompt variants on a single model before running the full shortlist batch.

- **Variant configs**: `configs/prompt_variants/` (`v_base.yaml`, `v_nofewshot.yaml`, `v_think.yaml`)
- **Runner**:

```bash
cd /workspace/metacog-llm-compute
python scripts/run_prompt_ab.py --pilot-mode cuda --output-dir data/results/runpod_pilot
```

This writes `data/results/runpod_pilot/ab_YYYYMMDD_HHMMSS/ab_summary.json` plus per-variant pilot folders.

## Download results from the pod

From your **local machine** (repo root). RunPod shows **two** SSH options in the dashboard:

| Connection | SCP / SFTP |
|------------|------------|
| `user@ssh.runpod.io` | **Not supported** — no reliable file copy |
| **SSH over exposed TCP** (`root@IP -p PORT`) | **Supported** — use this for downloads |

**Recommended:** copy **IP, port, and user** from **Pod → Connect → “SSH over exposed TCP”**, then:

```bash
./scripts/download_runpod_results.sh --tcp root YOUR_IP YOUR_PORT ~/.ssh/id_ed25519
```

Or one-liner (note **`scp -P`** cap for port; `root` and path are typical for TCP SSH):

```bash
mkdir -p data/results/runpod_pilot
scp -O -i ~/.ssh/id_ed25519 -P YOUR_PORT -r \
  root@YOUR_IP:/workspace/metacog-llm-compute/data/results/ \
  ./data/results/runpod_pilot/
```

To download **only one run folder** (recommended when iterating):

```bash
mkdir -p data/results/runpod_pilot
scp -O -i ~/.ssh/id_ed25519 -P YOUR_PORT -r \
  root@YOUR_IP:/workspace/metacog-llm-compute/data/results/pilot_YYYYMMDD_HHMMSS \
  ./data/results/runpod_pilot/
```

The gateway URL (`ssh.runpod.io`) is fine for **interactive** `ssh`; for **scp/rsync** use **TCP** or workarounds (HTTP server on the pod, etc.).

After download, look for `run_info.json` and `pilot_*.json` inside the run folder under `data/results/runpod_pilot/`. The pod must be **running** for `scp` to succeed.

Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).
