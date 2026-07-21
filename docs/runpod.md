# Running on RunPod Cloud GPU

The pilot with `--real` is designed for **~2 GPU-hours** on an **RTX 5090** (32 GB VRAM). Budget: about **$0.44** (see [`blueprints/infrastructureplan_pilot.md`](../blueprints/infrastructureplan_pilot.md) Section V).

**Related:** [`docs/pilot.md`](pilot.md) (pilot modes), [`docs/scripts.md`](scripts.md), [`configs/experiment_core.yaml`](../configs/experiment_core.yaml) (production model + inference config).

## Step 1 — Create & start the pod (dashboard settings)

- **GPU**: RTX 5090, **count = 1**
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

**RunPod template caveat:** The PyTorch template sets `HF_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, and related vars to `/workspace/.cache/*` in the container environment. That fills the **network volume** during `pip install` and model download. `setup_cloud.sh` overrides this via `scripts/cloud/shell/pod_runtime_env.sh` — do **not** re-export the template paths in your shell before setup.

If you want **results persistent** but **models ephemeral**, use the repo scripts (recommended):

```bash
cd /workspace/metacog-llm-compute
bash scripts/cloud/shell/setup_cloud.sh
source scripts/cloud/shell/activate_pod_env.sh
```

Manual overrides (only if not using `setup_cloud.sh`):

```bash
export RESULTS_DIR="/workspace/metacog-llm-compute/data/results"
export HF_HOME="/root/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="/root/.cache/pip"
```

## Step 4 — Hugging Face token (recommended)

Some checkpoints are **gated** on the Hub. On the pod:

```bash
export HF_TOKEN="..."   # read-only token is enough for downloads
```

This also improves rate limits for metadata scripts like `scripts/pilot_analysis/hf_model_card_gate.py`.

**Fast Hub downloads (`hf_transfer`):** Some RunPod images or `/workspace/secrets/env.sh` set `HF_HUB_ENABLE_HF_TRANSFER=1`. That requires the `hf_transfer` package — it is pinned in `requirements.txt` (Linux) and installed by `scripts/cloud/shell/setup_cloud.sh`. If you see `Fast download using 'hf_transfer' is enabled … but 'hf_transfer' package is not available`, re-run setup or `pip install hf_transfer`.

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

On a **new pod** (fresh container disk), one command from the repo root on `/workspace`:

```bash
cd /workspace/metacog-llm-compute   # skip clone if the volume already has the repo
bash scripts/cloud/shell/setup_cloud.sh
```

`setup_cloud.sh` prepares a **runnable pod** (secrets, SSH, Python env, model cache). It does **not** start vLLM or run Gate C / probes — you choose what to run afterward.

| Step | What |
|------|------|
| `/workspace/.cache` cleanup | removes stale template pip/HF/uv caches from the **network volume** |
| `source /workspace/secrets/env.sh` | HF_TOKEN, Langfuse, etc. (secrets only; cache paths re-forced afterward) |
| GitHub deploy key | copies `/workspace/secrets/runpod_github_ed25519` → `~/.ssh/` |
| Git sync | `git fetch origin`, then **ff-only pull on the branch already checked out** (no branch switch). Optional override: `GIT_BRANCH=main` |
| venv | `/root/venv-metacog` on **container disk** |
| Caches | `scripts/cloud/shell/pod_runtime_env.sh` → `HF_HOME=/root/.cache/huggingface`, `PIP_CACHE_DIR=/root/.cache/pip` (overrides RunPod `/workspace/.cache/*`) |
| deps + model | `requirements.txt` + optional `Qwen/Qwen3-8B` pre-download |

**Skip flags** (e.g. stopped pod with venv + model still on container disk):

```bash
SKIP_MODEL_DOWNLOAD=1 bash scripts/cloud/shell/setup_cloud.sh
# SKIP_GIT_SYNC=1   — no fetch/pull (deploy key still installed)
# SKIP_VENV=1       — reuse existing venv
```

**Switch branch before setup** (if you want a different branch than the one on disk):

```bash
cd /workspace/metacog-llm-compute
git checkout main   # or any branch you use
bash scripts/cloud/shell/setup_cloud.sh
# or one-shot: GIT_BRANCH=main bash scripts/cloud/shell/setup_cloud.sh
```

**New SSH session** (after setup):

```bash
cd /workspace/metacog-llm-compute
source scripts/cloud/shell/activate_pod_env.sh
```

Start **vLLM serve** separately when you need GPU inference — see `docs/runpod.md`.

This installs the **pinned** dependency set from `requirements.txt` (includes `vllm`, `transformers`, `textworld`, `numpy`, `pandas`, `scipy`, `pyyaml`, test deps, etc.).

**Optional — pre-download only** (if you already ran setup and only need weights):

```bash
export MODEL_NAME="Qwen/Qwen3-8B"
export SKIP_MODEL_DOWNLOAD=0
bash scripts/cloud/shell/setup_cloud.sh
```

Store models on the **container disk** (`HF_HOME=/root/.cache/huggingface`) so the 10 GB network volume is not filled; results stay under `/workspace/metacog-llm-compute/data/results`.

## Step 6b — Generate TextWorld games (required before TextWorld pilot)

TextWorld story files (`.z8`) are **not** bundled in the repo. Generate them on the pod once per difficulty batch (or after wiping `data/tasks/textworld/`). Each instance needs **`textworld_{i}.z8`** and the matching **`textworld_{i}.json`** game dump from TextWorld — do not delete the `.json` sidecar.

From repo root on the pod (after `setup_cloud.sh`):

```bash
cd /workspace/metacog-llm-compute
python scripts/datasets/generate_textworld_games.py \
  --num-rooms 5 \
  --num-ingredients 2 \
  --cook \
  --seed 42 \
  --num-instances 5
```

This writes under `data/tasks/textworld/` (see [`docs/textworld.md`](textworld.md)). For a quick smoke test, `--num-instances 5` matches `pilot.instances` in `configs/pilot.yaml`. The final experiment uses 50 instances via [`scripts/datasets/build_textworld_manifest.py`](scripts/datasets/build_textworld_manifest.py) after difficulty calibration.

Optional sanity check:

```bash
python scripts/datasets/play_textworld.py data/tasks/textworld/textworld_0.z8
```

## Step 7 — Hugging Face model-card gate (before burning GPU time)

Quick, read-only Hub scan for obvious exclusion flags (MoE-ish language, multimodal/VL language, “thinking mode” keywords):

```bash
cd /workspace/metacog-llm-compute
python scripts/pilot_analysis/hf_model_card_gate.py --repo-id Qwen/Qwen3-8B
```

(`--models-file` also accepts an ad-hoc YAML list of repo ids if you want to scan several at once;
the former shipped shortlist files were removed in the 2026-07-21 refactor.)

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
python scripts/experiment/run_pilot.py --config configs/pilot.yaml --output-dir "${RESULTS_DIR}" --pilot-mode cuda --real
```

Or use `--real` to auto-detect (on the pod this will select `cuda`).

**Inference contract (load-bearing for model selection):**

- **Thinking is stage-forced**, not controlled by a single global default. C0 and VC follow-up calls force `enable_thinking=false`; **C1 reason calls and C2 samples force `enable_thinking=true`** regardless of `inference.enable_thinking` in YAML. See [`docs/pilot.md`](pilot.md) § “Thinking mode” and “Output handling contract (C1)”.
- **VC sizing** for RunPod gates and Phase 1/2 is defined in [`configs/experiment_core.yaml`](../configs/experiment_core.yaml) (`vc.followup_max_tokens: 4`, `followup_temperature: 0.2`, prompt ending with `Confidence:`). Do **not** use the pilot-only `followup_max_tokens: 24` from `configs/pilot.yaml` when judging VC readiness.
- **C1 format compliance** is evaluated with thinking **ON**. Before trusting C1 on a new shortlist model, run the handoff gate (both domains if possible):

```bash
python scripts/pilot_analysis/run_c1_handoff_gate.py --config configs/experiment_core.yaml --pilot-mode cuda --real \
  --output-dir "${RESULTS_DIR}" --domain textworld --n-episodes 3 --max-steps 5
python scripts/pilot_analysis/run_c1_handoff_gate.py --config configs/experiment_core.yaml --pilot-mode cuda --real \
  --output-dir "${RESULTS_DIR}" --domain tower_of_hanoi --n-episodes 3 --max-steps 5
```

Inspect `c1_handoff_gate_*.md` and `debug_views/` for `parse_method: post_think` vs fallbacks/unparsed.

**Output layout on the pod:**

| Script | Typical output |
|--------|----------------|
| `run_pilot.py` | `${RESULTS_DIR}/pilot_YYYYMMDD_HHMMSS/` |

Optional vLLM logprob probe before a long run:

```bash
python scripts/instrument_validation/probe_vllm_logprobs.py --config configs/pilot.yaml --pilot-mode cuda --real
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

**If vLLM fails during model init with a KV-cache / max-seq-len error (common on 24 GB GPUs):**

Some models advertise very large context lengths (e.g. 40960). On 24 GB GPUs, vLLM may fail with a message like “KV cache needed … larger than available …”. Set a smaller `inference.max_model_len` (e.g. `8192` or `16384`) in `configs/pilot.yaml` and `configs/experiment_core.yaml`, then rerun.

**vLLM dtype:** use `model.dtype: float16` in YAML. vLLM 0.19+ does not accept the alias `fp16` when constructing the engine (affects `verify_backend_parity.py`, Phase 1/2 runners, and pilot smoke tests).

**vLLM logprobs mode:** pin `logprobs_mode="raw_logprobs"` on the **engine** (`LLM(...)`), not on `SamplingParams`. V1 default is raw (pre-temperature). Parity gate: `python scripts/instrument_validation/verify_backend_parity.py --backend vllm` (TLE tolerance + Same-T diagnostic; see `docs/pilot.md` §5.7).

You should see a new timestamped folder like `data/results/pilot_YYYYMMDD_HHMMSS/` containing at least:

- `run_info.json` — resolved config + model id + pilot mode
- `pilot_sanity.json`, `pilot_test1_inference.json`, `pilot_test2_tle.json`, `pilot_test3_vc.json`, `pilot_test5_toh.json`, `pilot_feasibility.json`
- TextWorld + ToH episode JSONs (`ep_*.json`) plus optional `trace_*.jsonl`, `logprobs/`, `vc/`

## Phase 1 / Phase 2 autostop

Wrap long runs with [`scripts/cloud/shell/run_with_autostop.sh`](../scripts/cloud/shell/run_with_autostop.sh) (best-effort `runpodctl stop` when `RUNPOD_POD_ID` is set; local no-op):

```bash
cd /workspace/metacog-llm-compute
export RESULTS_DIR="/workspace/metacog-llm-compute/data/results"
STOP_POD=1 ./scripts/cloud/shell/run_with_autostop.sh scripts/experiment/run_phase1.py \
  --config configs/experiment_core.yaml --real --resume \
  --checkpoint-dir "${RESULTS_DIR}/phase1/phase1_YYYYMMDD_HHMMSS_UTC"
```

**Resume directory trap:** `--checkpoint-dir` must be the **exact timestamped run folder** that contains `ep_*.json` (and optionally `quarantine.jsonl`), not the parent `phase1/` base.

Phase runners write `env_assertion` / `label_error` failures to `quarantine.jsonl` and skip those episodes on resume; all other episode failures still append to `errors.jsonl` unchanged.

## Download results from the pod

From your **local machine** (repo root). RunPod shows **two** SSH options in the dashboard:

| Connection | SCP / SFTP |
|------------|------------|
| `user@ssh.runpod.io` | **Not supported** — no reliable file copy |
| **SSH over exposed TCP** (`root@IP -p PORT`) | **Supported** — use this for downloads |

**Recommended:** copy **IP, port, and user** from **Pod → Connect → “SSH over exposed TCP”**, then:

```bash
./scripts/cloud/shell/download_runpod_results.sh --tcp root YOUR_IP YOUR_PORT ~/.ssh/id_ed25519
```

The script copies remote `data/results/` into `data/results/runpod_pilot/` and **flattens** the common `runpod_pilot/results/pilot_*` nesting that plain `scp -r …/results/` creates.

To download **only one run folder** (recommended when iterating):

```bash
./scripts/cloud/shell/download_runpod_results.sh --tcp root YOUR_IP YOUR_PORT ~/.ssh/id_ed25519 \
  --run pilot_YYYYMMDD_HHMMSS
```

If you already downloaded with an older one-liner and see `data/results/runpod_pilot/results/pilot_*`, repair locally:

```bash
python scripts/cloud/python/flatten_runpod_download.py data/results/runpod_pilot
```

**Expected local layout** after download or flatten:

```
data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS/run_info.json
data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS/pilot_sanity.json
```

The gateway URL (`ssh.runpod.io`) is fine for **interactive** `ssh`; for **scp/rsync** use **TCP** or workarounds (HTTP server on the pod, etc.).

After download, validate locally:

```bash
python scripts/pilot_analysis/validate_pilot_outputs.py data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS
python scripts/pilot_analysis/audit_pilot_signals.py data/results/runpod_pilot/pilot_YYYYMMDD_HHMMSS
```

The pod must be **running** (not stopped) for `scp` to succeed.

## Parallel execution (`src/execution/`) — vLLM server mode

Phase 1/2 with `--real` use **`ServerBackend`** (sync HTTP) against a shared OpenAI-compatible server — not in-process `VLLMWrapper`.

**Start server (5090 pod, load-bearing flags):**

```bash
vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype float16 \
  --max-model-len 16384 \
  --logprobs-mode raw_logprobs \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --port 8000
```

Tune `--max-num-seqs` to at least your chosen `execution.max_concurrent_episodes` (after throughput sweep). Raise `--gpu-memory-utilization` toward `0.95` only if the pod stays stable (watch for KV preemption/OOM). **Freeze** attention backend and prefix-caching settings before `verify_backend_parity.py` — changing them later requires re-running parity.

**5090 / Blackwell workaround:** On some RunPod images the default `FLASH_ATTN` (FlashAttention-2) path fails at engine init with `cudaErrorUnsupportedPtxVersion` (`the provided PTX was compiled with an unsupported toolchain`). The model loads fine; the crash is in the FA2 kernels. Use `--attention-backend TRITON_ATTN` (or `FLASHINFER`) instead. Re-verify logprobs after switching backends.

Set in YAML (`execution.server_url: "http://127.0.0.1:8000/v1"`) or export before runners. The Python client (`ServerBackend`) must **not** serialize HTTP POSTs — parallel episode threads rely on vLLM continuous batching. C2 uses one request with `n=3` samples when the server supports it.

Optional: `execution.server_timeout_s` in YAML (default 600s) for long C2 thinking generations under load.

### Validation regime: plumbing smoke vs TLE invariance under load

| Level | Script | N | GO/NO-GO |
|-------|--------|---|----------|
| **Plumbing smoke** | `scripts/instrument_validation/smoke_parallel.py` | Small N in `configs/dev/smoke.yaml` (e.g. 3) | `SMOKE_PARALLEL: GO/NO-GO` — scheduler, checkpoints, concurrency |
| **TLE invariance validation** | `scripts/instrument_validation/verify_backend_parity.py --backend server` | Production `max_concurrent_episodes` from `experiment_core.yaml` | Temperature + batch invariance at committed-action TLE window |

**Green plumbing smoke ≠ TLE invariance evidence.** Smoke does not replace batch-invariance or eps derivation under load.

```bash
# Plumbing (mock locally, or --real on pod with server running):
python scripts/instrument_validation/smoke_parallel.py --config configs/dev/smoke.yaml --output-dir data/results/smoke_parallel
python scripts/instrument_validation/smoke_parallel.py --config configs/dev/smoke.yaml --real --output-dir data/results/smoke_parallel

# TLE invariance validation (production N, saturated pool):
python scripts/instrument_validation/verify_backend_parity.py --backend server \
  --config configs/experiment_core.yaml \
  --output-dir data/results/instrument_validation
```

### Instrument validation session (5090 pod)

After connecting Cursor via SSH Remote to `/workspace/metacog-llm-compute`:

**Order matters:** pick production `max_concurrent_episodes` from measured throughput *before* backend parity (batch invariance must use the N you will run in Phase 1).

```bash
bash scripts/cloud/shell/instrument_validation_preflight.sh
# Terminal A: vllm serve … (see above; pin --revision to match experiment_core.yaml)
# Terminal B — results under data/results/instrument_validation/

# 1) Plumbing + thinking check
python scripts/instrument_validation/smoke_parallel.py --config configs/dev/smoke.yaml --real \
  --output-dir data/results/instrument_validation/smoke_parallel

# 2) Throughput sweep — choose N (try 8,16,24,32 after server concurrency fix; extend if headroom)
python scripts/instrument_validation/measure_concurrent_throughput.py --real \
  --candidates 8,16,24,32 \
  --output data/results/instrument_validation/throughput_sweep.json
# Set execution.max_concurrent_episodes in experiment_core.yaml + dev configs to chosen N

# 3) Backend parity at production N (hard stop if FAIL)
python scripts/instrument_validation/verify_backend_parity.py --backend server \
  --config configs/experiment_core.yaml \
  --output-dir data/results/instrument_validation

# 4) Format/VC probe — check vc_rate + traces; preanalysis AUROC is smoke-only here
python scripts/experiment/run_phase1.py --config configs/dev/format_vc_probe.yaml --real \
  --checkpoint-dir data/results/instrument_validation
PROBE=$(ls -td data/results/instrument_validation/phase1_* | head -1)
python scripts/pilot_analysis/audit_pilot_signals.py "$PROBE" --json
python -m src.analysis.preanalysis_screen "$PROBE"   # AUROC marked (smoke) if n<50

# 5) ToH parse probe
python scripts/experiment/run_phase1.py --config configs/dev/toh_parse_probe.yaml --real \
  --checkpoint-dir data/results/instrument_validation

# 6) Signal smoke — extrapolate wall time from probe ep/h before starting
python scripts/experiment/run_phase1.py --config configs/dev/signal_smoke.yaml --real \
  --checkpoint-dir data/results/instrument_validation
RUN=$(ls -td data/results/instrument_validation/phase1_* | head -1)
python -m src.analysis.preanalysis_screen "$RUN"
python scripts/instrument_validation/sweep_topk_sensitivity.py "$RUN" --output "$RUN/topk_sensitivity.json"
```

Dev configs: `format_vc_probe.yaml`, `toh_parse_probe.yaml`, `signal_smoke.yaml` (72 episodes, 1 run/condition).
Track progress in `docs/instrument_validation_session.md` (not committed).
Gate checklist: `blueprints/gate_p1_readiness.md` (Gate C), `docs/consistency_log.md`.

**One-shot after `perf/vllm-server-concurrency` merge on pod:**

```bash
bash scripts/cloud/shell/run_instrument_validation_after_perf.sh
```

This runs re-sweep → `apply_production_n.py` → parity → format_vc_probe → toh_parse_probe → (optional) signal_smoke.

**Production N:** Do not freeze `max_concurrent_episodes` at a guess (e.g. 3). Sweep first; if Phase 1 later uses a higher N, re-run parity. Running parity at lower N than production limits generalizability — document as limitation if unavoidable.

**Attention backend:** If you switch `TRITON_ATTN` → `FLASHINFER` (or back) after a passed parity run, **re-run** `verify_backend_parity.py` — backends can change numerics.

**Budget / time:** Rough ep/h from the sweep × planned episodes is enough for planning; no need to optimize to the last euro. After `format_vc_probe`, extrapolate C-5 duration before launching `signal_smoke`.

Block before Smoke `--real` GO: `enable_thinking` must produce thinking blocks on the server (C1/C2 degrade silently otherwise).

### §5.7.5 — eps under load (5090)

Der `verify_backend_parity`-Kontrolllauf zur eps-Ableitung läuft **unter Last** (gesättigter Pool bei Produktions-N), nicht solo. Begründung: der Same-T-Kontrollterm in `resolve_tle_invariance_eps()` muss Batch-Rauschen des Erhebungsregimes enthalten; solo-abgeleitetes eps wäre zu eng.

Nach erfolgreicher TLE-Invarianz-Validierung: `(N, eps)` in `run_metadata.frozen_execution_params` einfrieren (`execution.frozen_*` in YAML oder `--freeze-metadata-dir` on `verify_backend_parity.py --backend server`).

### Fallback escalation (document only — not automated)

If `max |dTLE|` exceeds eps: (1) enable vLLM batch-invariance kernels (FlashInfer/Blackwell path on 5090), (2) re-derive eps under load at the same production N and re-freeze, (3) accept ~55–60% throughput loss. Default remains without batch-invariant kernels.

### §5.9 limitation note

Batch jitter within eps is bounded measurement noise, not a confound (content-agnostic scheduling; Phase 1 uses stage-homogeneous passes). Trajectory divergence solo vs parallel is **descriptive only** (`execution_metrics.trajectory_divergence_rate`), never GO/NO-GO.

## Gate-1 readiness smoke (checklist)

Run this sequence when validating infrastructure before committing GPU budget to Gate 1 (20 episodes/domain at C0):

1. **Pod:** Start stopped pod (or create new one). Confirm `/workspace/metacog-llm-compute` exists; `git pull` on `feat/runpod-gate1-readiness` or `main` after merge.
2. **Env:** `bash scripts/cloud/shell/setup_cloud.sh` (if deps changed).
3. **TextWorld:** Step 6b — generate games if `data/tasks/textworld/textworld_0.z8` is missing.
4. **Unit tests:** `python -m pytest tests/ -v`
5. **L0.1 probe:** `python scripts/instrument_validation/probe_vllm_logprobs.py --config configs/pilot.yaml --pilot-mode cuda --real`
6. **C1 handoff gate (thinking ON):** `run_c1_handoff_gate.py` with `configs/experiment_core.yaml` on TextWorld and ToH (see Step 9 inference contract).
7. **Pilot:** `python scripts/experiment/run_pilot.py --config configs/pilot.yaml --output-dir "${RESULTS_DIR}" --pilot-mode cuda --real` (mirror `vc.*` from `experiment_core.yaml` if auditing VC against thesis defaults).
8. **Download** (local machine, pod running): `./scripts/cloud/shell/download_runpod_results.sh --tcp … [--run pilot_…]`
9. **Audit** (local): `python scripts/pilot_analysis/audit_pilot_signals.py data/results/runpod_pilot/pilot_…`

**Pass criteria for this smoke:** `pilot_sanity.json` has `has_logprobs: true`; `audit_pilot_signals` shows C0 `tle_rate >= 0.95`; VC rate `>= 0.80` under the **`experiment_core.yaml` VC contract** (`followup_max_tokens: 4`); C1 handoff gate shows low unparsed rate with thinking ON; C2 traces use `self_consistency_majority_vote` when `compute_stages` includes C2. Gate 1 itself still requires a dedicated 20-episode C0 parseability run per domain (see `blueprints/thesis_dependency_map.html`).

Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).
