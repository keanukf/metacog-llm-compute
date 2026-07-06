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

### Pod lifecycle: Stop vs Terminate

- **Stop** (pause): GPU billing stops; the pod and its **network volume** (`/workspace`) stay attached. You pay only a small storage fee for the volume — useful between sessions when you will return soon.
- **Terminate**: destroys the **container disk** (ephemeral root). Anything **not** on `/workspace` is lost (SSH keys in `~/.ssh/`, HF cache on container disk, etc.). Code and results under `/workspace/metacog-llm-compute` persist if you cloned there.

Prefer **Stop** when iterating; **Terminate** only when you are done with that pod for good.

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

## Step 5 — Get the repo on the pod

This repo is `keanukf/metacog-llm-compute`. On RunPod, the most reliable way is **GitHub SSH**.

Use `/workspace` so that code and results live on the network volume and persist across pod restarts.

**SSH key note:** Keys live in `~/.ssh/` on the **container disk** (ephemeral). On a **new pod**, regenerate the key (or restore it from `/workspace`) and add the public key in GitHub → Settings → SSH and GPG keys. The repo clone on `/workspace` persists; the SSH key usually does not.

### SSH setup (once per pod session)

```bash
# On the pod:
ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_ed25519_runpod
cat ~/.ssh/id_ed25519_runpod.pub
# Add that public key in GitHub → Settings → SSH and GPG keys

# Sanity check (must say "Hi keanukf!")
ssh -T -i ~/.ssh/id_ed25519_runpod git@github.com
```

**Recommended — make Git use this key automatically** (so plain `git pull` works):

```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_runpod
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Without `~/.ssh/config`, prefix Git commands with:

```bash
export GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_runpod -o IdentitiesOnly=yes'
```

### First time — clone

Skip this if `/workspace/metacog-llm-compute` already exists.

```bash
cd /workspace
git clone git@github.com:keanukf/metacog-llm-compute.git metacog-llm-compute
cd /workspace/metacog-llm-compute
```

(If you did not set up `~/.ssh/config`, use `GIT_SSH_COMMAND=... git clone ...` as above.)

### Returning session — pull (repo already on persistent storage)

If you cloned to `/workspace` before, **do not clone again** — update in place:

```bash
cd /workspace/metacog-llm-compute
git pull
```

If Git reports **divergent branches** and you only want the latest remote code (typical on a pod; local commits are not important):

```bash
cd /workspace/metacog-llm-compute
git fetch origin
git reset --hard origin/main
```

This resets **tracked** files only. Ignored paths such as `data/results/` and `.env` on the pod are **not** removed by `git reset --hard`.

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

## Step 6b — Generate TextWorld games (required before TextWorld pilot)

TextWorld story files (`.z8`) are **not** bundled in the repo. Generate them on the pod once per difficulty batch (or after wiping `data/tasks/textworld/`). Each instance needs **`textworld_{i}.z8`** and the matching **`textworld_{i}.json`** game dump from TextWorld — do not delete the `.json` sidecar.

From repo root on the pod (after `setup_cloud.sh`):

```bash
cd /workspace/metacog-llm-compute
python scripts/generate_textworld_games.py \
  --num-rooms 5 \
  --num-ingredients 2 \
  --cook \
  --seed 42 \
  --num-instances 5
```

This writes under `data/tasks/textworld/` (see [`docs/textworld.md`](textworld.md)). For a quick smoke test, `--num-instances 5` matches `pilot.instances` in `configs/pilot.yaml`. The final experiment uses 50 instances via [`scripts/build_textworld_manifest.py`](scripts/build_textworld_manifest.py) after difficulty calibration.

Optional sanity check:

```bash
python scripts/play_textworld.py data/tasks/textworld/textworld_0.z8
```

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

Run the pilot **with real inference** on the pod (Pilot 2). Use the persistent results path from Step 3:

```bash
cd /workspace/metacog-llm-compute
export RESULTS_DIR="/workspace/metacog-llm-compute/data/results"
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir "${RESULTS_DIR}" --pilot-mode cuda --real
```

Or use `--real` to auto-detect (on the pod this will select `cuda`).

**Inference contract (load-bearing for model selection):**

- **Thinking is stage-forced**, not controlled by a single global default. C0 and VC follow-up calls force `enable_thinking=false`; **C1 reason calls and C2 samples force `enable_thinking=true`** regardless of `inference.enable_thinking` in YAML. See [`docs/pilot.md`](pilot.md) § “Thinking mode” and “Output handling contract (C1)”.
- **VC sizing** for RunPod gates and Phase 1/2 is defined in [`configs/experiment_core.yaml`](../configs/experiment_core.yaml) (`vc.followup_max_tokens: 4`, `followup_temperature: 0.2`, prompt ending with `Confidence:`). Do **not** use the pilot-only `followup_max_tokens: 24` from `configs/pilot.yaml` when judging VC readiness.
- **C1 format compliance** is evaluated with thinking **ON**. Before trusting C1 on a new shortlist model, run the handoff gate (both domains if possible):

```bash
python scripts/run_c1_handoff_gate.py --config configs/experiment_core.yaml --pilot-mode cuda --real \
  --output-dir "${RESULTS_DIR}" --domain textworld --n-episodes 3 --max-steps 5
python scripts/run_c1_handoff_gate.py --config configs/experiment_core.yaml --pilot-mode cuda --real \
  --output-dir "${RESULTS_DIR}" --domain tower_of_hanoi --n-episodes 3 --max-steps 5
```

Inspect `c1_handoff_gate_*.md` and `debug_views/` for `parse_method: post_think` vs fallbacks/unparsed.

**Output layout on the pod:**

| Script | Typical output |
|--------|----------------|
| `run_pilot.py` | `${RESULTS_DIR}/pilot_YYYYMMDD_HHMMSS/` |
| `run_prompt_ab.py` | `data/results/runpod_pilot/ab_YYYYMMDD_HHMMSS/` (default `--output-dir`) |

Optional vLLM logprob probe before a long run:

```bash
python scripts/probe_vllm_logprobs.py --config configs/pilot.yaml --pilot-mode cuda --real
```

### Troubleshooting: weird model outputs (blank actions / prompt echo / VC always null / C1 parse failures)

- **Model outputs empty actions (0 tokens) / stops immediately**
  - Ensure `inference.chat_template: true` (required for instruct/chat models like Qwen3). Set explicitly in `configs/pilot.yaml` if missing; `experiment_core.yaml` defaults to `true` in the wrapper.
  - For TextWorld in the pilot config, avoid stopping on a single newline: `domain_prompts.textworld.action_stop: ["\n\n"]`.
- **Model echoes prompt fragments (e.g. “Do not use disk numbers.”)**
  - Strong signal that the chat template is not applied. Keep `inference.chat_template: true`.
- **VC is always `null`**
  - Align with the **final** VC contract in `configs/experiment_core.yaml`: `vc.followup_max_tokens: 4`, `followup_temperature: 0.2`, and the `Confidence:`-terminated `followup_instruction` (first line only is parsed; extra words → `null`).
  - If you run `run_pilot.py` with `configs/pilot.yaml`, temporarily mirror those `vc.*` keys from `experiment_core.yaml` for comparable VC rates — the pilot file’s **24**-token follow-up is a legacy convenience, not the thesis default.
- **Thinking text floods the action / C1 unparsed steps**
  - C1 is always evaluated with thinking **ON** on the reason call. Do not disable thinking globally to “fix” C0; use the C1 handoff gate above and inspect `debug_views/` (`reason` block, `parse_method` in traces).
  - For prompt A/B only, `configs/prompt_variants/v_think.yaml` toggles thinking for controlled comparisons — not the production stage contract.

**If vLLM fails during model init with a KV-cache / max-seq-len error (common on 24 GB GPUs):**

Some models advertise very large context lengths (e.g. 40960). On 24 GB GPUs, vLLM may fail with a message like “KV cache needed … larger than available …”. Set a smaller `inference.max_model_len` (e.g. `8192` or `16384`) in `configs/pilot.yaml` and `configs/experiment_core.yaml`, then rerun.

**vLLM dtype:** use `model.dtype: float16` in YAML. vLLM 0.19+ does not accept the alias `fp16` when constructing the engine (affects `verify_backend_parity.py`, Phase 1/2 runners, and pilot smoke tests).

**vLLM logprobs mode:** pin `logprobs_mode="raw_logprobs"` on the **engine** (`LLM(...)`), not on `SamplingParams`. V1 default is raw (pre-temperature). Parity gate: `python scripts/verify_backend_parity.py --backend vllm` (TLE tolerance + Same-T diagnostic; see `docs/pilot.md` §5.7).

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

The script copies remote `data/results/` into `data/results/runpod_pilot/` and **flattens** the common `runpod_pilot/results/pilot_*` nesting that plain `scp -r …/results/` creates.

To download **only one run folder** (recommended when iterating):

```bash
./scripts/download_runpod_results.sh --tcp root YOUR_IP YOUR_PORT ~/.ssh/id_ed25519 \
  --run pilot_YYYYMMDD_HHMMSS
```

If you already downloaded with an older one-liner and see `data/results/runpod_pilot/results/pilot_*`, repair locally:

```bash
python scripts/flatten_runpod_download.py data/results/runpod_pilot
```

**Expected local layout** after download or flatten:

```
data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS/run_info.json
data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS/pilot_sanity.json
```

The gateway URL (`ssh.runpod.io`) is fine for **interactive** `ssh`; for **scp/rsync** use **TCP** or workarounds (HTTP server on the pod, etc.).

After download, validate locally:

```bash
python scripts/validate_pilot_outputs.py data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS
python scripts/audit_pilot_signals.py data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS
```

The pod must be **running** (not stopped) for `scp` to succeed.

## Gate-1 readiness smoke (checklist)

Run this sequence when validating infrastructure before committing GPU budget to Gate 1 (20 episodes/domain at C0):

1. **Pod:** Start stopped pod (or create new one). Confirm `/workspace/metacog-llm-compute` exists; `git pull` on `feat/runpod-gate1-readiness` or `main` after merge.
2. **Env:** `bash scripts/setup_cloud.sh` (if deps changed).
3. **TextWorld:** Step 6b — generate games if `data/tasks/textworld/textworld_0.z8` is missing.
4. **Unit tests:** `python -m pytest tests/ -v`
5. **L0.1 probe:** `python scripts/probe_vllm_logprobs.py --config configs/pilot.yaml --pilot-mode cuda --real`
6. **C1 handoff gate (thinking ON):** `run_c1_handoff_gate.py` with `configs/experiment_core.yaml` on TextWorld and ToH (see Step 9 inference contract).
7. **Pilot:** `python scripts/run_pilot.py --config configs/pilot.yaml --output-dir "${RESULTS_DIR}" --pilot-mode cuda --real` (mirror `vc.*` from `experiment_core.yaml` if auditing VC against thesis defaults).
8. **Download** (local machine, pod running): `./scripts/download_runpod_results.sh --tcp … [--run pilot_…]`
9. **Audit** (local): `python scripts/audit_pilot_signals.py data/results/runpod_pilot/pilot_…`

**Pass criteria for this smoke:** `pilot_sanity.json` has `has_logprobs: true`; `audit_pilot_signals` shows C0 `tle_rate >= 0.95`; VC rate `>= 0.80` under the **`experiment_core.yaml` VC contract** (`followup_max_tokens: 4`); C1 handoff gate shows low unparsed rate with thinking ON; C2 traces use `self_consistency_majority_vote` when `compute_stages` includes C2. Gate 1 itself still requires a dedicated 20-episode C0 parseability run per domain (see `blueprints/thesis_dependency_map.html`).

Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).
