# Metacognitive Effort Allocation in Sequential LM Agents

## General information

**Repository goal.** This is the thesis codebase for **metacognitive effort allocation in sequential language-model agents**. The core idea is to use lightweight proxy signals—**token-level entropy (TLE)** and **verbalized confidence (VC)**—to decide how much compute to spend per step (direct answer vs chain-of-thought with verification vs best-of-N). The pipeline includes calibration, an agent loop with fixed compute stages (**C0 / C1 / C2**), environments such as TextWorld and delayed-cue recall, and runners for local pilots and cloud experiments (e.g. RunPod). Design choices, sample sizes, and infrastructure are documented under `blueprints/`.

**Structure (high level).**

| Location | Role |
|----------|------|
| `src/` | Main Python package: `agent/` (loop, stages, allocator), `signals/` (TLE, VC, etc.), `environments/` (tasks), `analysis/` (calibration, comparison, plots), `utils/` (model wrapper, logging, checkpoints). |
| `configs/` | YAML experiment and pilot configs (no secrets; use env vars for API keys). |
| `scripts/` | Entry points such as `run_pilot.py`, `run_phase1.py`, `run_phase2.py`, and helper utilities. |
| `tests/` | `pytest` suite with mocks (no GPU required). |
| `data/` | Task assets and run outputs (e.g. `data/tasks/`, `data/results/`); paths are configurable. |
| `blueprints/` | Thesis design, pilot infrastructure, and related planning notes. |

Run scripts from the **repository root** so imports resolve (see `run_pilot.py` pattern).

---

## Unit tests vs pilot: what each is for

| | **Unit tests (pytest)** | **Pilot (run_pilot.py)** |
|---|-------------------------|---------------------------|
| **Purpose** | Check that **code and interfaces** are correct: signals, agent loop, logging, calibration logic. | Check **setup and hardware** in a small end-to-end run: real model, throughput, full pipeline. |
| **Runs** | Pytest suite with **mocks** (no model, no GPU). | One script that runs Tests 1–6 in sequence; **pilot mode** chooses mock, hf, CUDA, or lmstudio. |
| **Where** | Local or on pod; **no GPU required**. | **Pilot 0** (mock): anywhere. **Pilot 1** (hf): Mac with Apple Silicon (HF+MPS). **Pilot 2** (CUDA): e.g. RunPod. **Pilot 3** (lmstudio): LM Studio server on LAN or localhost. |
| **When** | After every code change; in CI. | Pilot 0: quick local sanity. Pilot 1: test HF+MPS on Mac before buying GPU. Pilot 2: confirm GPU setup. Pilot 3: LM Studio (often faster than raw HF on the same Mac). |
| **Output** | Pass/fail per test. | `pilot_benchmark.json`, `pilot_calibration.json`, and optionally `pilot_cost_validation.md`, `pilot_feasibility_report.md`. |

**Summary:** Unit tests validate *logic*; the pilot validates *environment and hardware* in a small run. Pilot levels: **mock** (no real model), **hf** (HuggingFace + MPS on Apple Silicon; CLI still accepts deprecated **m1**), **cuda** (vLLM on GPU), **lmstudio** (LM Studio local OpenAI API, e.g. `http://host:1234/v1`).

---

## Running unit tests (no GPU)

All tests use mocks and run without GPU or cloud. From the repo root:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Optional: run a single test file or add markers:

```bash
python -m pytest tests/test_02_token_entropy.py -v
python -m pytest tests/ -v -m "not slow"
```

These tests assert structure (e.g. result has `tokens_per_sec`, `latency_mean`), signal behaviour (TLE, VC parsing), and that the mini e2e pipeline produces the expected episode keys. They do **not** run a real model.

---

## Running the pilot (setup and hardware check)

The pilot runs Tests 1–6 in sequence and writes benchmark and calibration outputs. Use `--pilot-mode` to choose the level:

| Mode | Flag | Hardware | Backend | Use case |
|------|------|----------|---------|----------|
| **Pilot 0** | `--pilot-mode mock` (default) | None | Mock | Quick local sanity; CI; no model download. |
| **Pilot 1** | `--pilot-mode hf` | Mac M1/M2 (Apple Silicon) | HuggingFace + MPS | Test full pipeline locally before buying GPU. |
| **Pilot 2** | `--pilot-mode cuda` | NVIDIA GPU (e.g. RunPod) | vLLM (or HF) | Validate GPU setup and throughput before Phase 1/2. |
| **Pilot 3** | `--pilot-mode lmstudio` | LM Studio host (LAN or localhost) | OpenAI-compatible HTTP | Local or LAN LM Studio (`LM_STUDIO_BASE_URL`, default `http://localhost:1234/v1`). |

### Pilot 0 — Mock (default)

No real model, stub environments. Confirms the script and output format:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results
```

You get per-step JSON under `--output-dir` (e.g. `pilot_test1_inference.json`, `pilot_test2_tle.json`, …) plus `ep_textworld_*.json` / `ep_tower_of_hanoi_*.json` for episodes and `pilot_feasibility.json`. In **mock** mode, metrics are synthetic (e.g. unrealistic `tokens_per_sec`).

Run individual steps without the full pipeline:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --only test2
```

`--only` accepts one or more of: `sanity`, `test1`, `test2`, `test3`, `test4`, `test5`, `feasibility` (executed in that order). For `feasibility`, missing inputs are filled from JSON already present in `output_dir` when available.

### Pilot 1 — HuggingFace on Apple Silicon (hf)

On a Mac with M1/M2/M3, run the same pipeline with the real model via HuggingFace on Metal (MPS). No CUDA or vLLM required:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode hf
```

`--pilot-mode m1` is accepted as a deprecated alias for `hf`. For the same machine, **lmstudio** is often faster because inference runs inside LM Studio instead of raw Transformers.

Requires `transformers`, `torch` with MPS support, and enough RAM/Unified Memory for the model (e.g. Qwen2.5-3B). Slower than Pilot 2 but catches setup and integration errors before spending on cloud GPU.

### Pilot 2 — Real CUDA GPU

On a machine with CUDA (e.g. RunPod RTX 3090), use vLLM for real inference and measured throughput:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode cuda
```

Or use `--real` to auto-detect: if CUDA is available → cuda; else if MPS (Mac) → hf; else mock.

### Pilot 3 — LM Studio

LM Studio exposes an OpenAI-compatible API (often `http://localhost:1234/v1` or a LAN address). Set `model.name` to the **exact model identifier** shown in LM Studio for API requests:

```bash
export LM_STUDIO_BASE_URL="http://192.168.178.173:1234/v1"
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode lmstudio
```

Or use `inference.lmstudio_base_url` / `inference.lmstudio_api_key` in YAML; default API key is `lm-studio` if unset (`LM_STUDIO_API_KEY`). Requires the `openai` package.

- **Test 1:** Real prompts → measured tokens/s, latency, VRAM (CUDA only).
- **Tests 2–3:** With a real model, TLE and VC are evaluated on **real generations**; in mock mode they use synthetic/static checks.
- **Test 4:** TextWorld e2e episodes (stub env unless a real game file is wired in).
- **Test 5:** Tower of Hanoi parseability (C0; count from `configs/pilot.yaml`).
- **Feasibility:** Go/No-Go checklist and ECE (from TextWorld episodes), written to `pilot_feasibility.json`.

After a non-mock pilot, `pilot_test1_inference.json` includes realistic `tokens_per_sec` for that device/endpoint.

---

## Running on RunPod Cloud GPU

The pilot with `--real` is designed for **~2 GPU-hours** on an **RTX 3090** (24 GB VRAM). Budget: about **$0.44** (see `blueprints/infrastructureplan_pilot.md` Section V).

### 1. Create and start a RunPod pod

- **GPU:** RTX 3090 (24 GB VRAM)
- **Template:** RunPod PyTorch 2.x
- **Volume:** 30 GB Network Volume (persistent for model and results)
- **Pricing:** Spot (~$0.22/hr) is sufficient; use On-Demand if you need no preemption.

Connect via SSH using the credentials shown in the RunPod dashboard.

### 2. Clone or upload the repo on the pod

From your **local machine**, upload the repo (e.g. rsync or git clone if the pod has network access):

```bash
# From your laptop: sync repo to the pod (replace POD_IP and KEY with your values)
rsync -avz -e "ssh -i /path/to/key" \
  --exclude '.git' --exclude '__pycache__' --exclude 'data/results/*.json' \
  ./ user@POD_IP:/workspace/metacog-llm-compute/
```

Or on the **pod** (if git is available):

```bash
cd /workspace
git clone <your-repo-url> metacog-llm-compute
cd metacog-llm-compute
```

Use `/workspace` so that code and results live on the network volume and persist across pod restarts.

### 3. Set up the environment on the pod

On the pod, from the repo root:

```bash
cd /workspace/metacog-llm-compute
bash scripts/setup_cloud.sh
```

This installs: `vllm`, `transformers`, `textworld`, `numpy`, `pandas`, `scipy`, `pyyaml`.

**Optional — pre-download the model** (saves time during the pilot; `scripts/setup_cloud.sh` already does this using `MODEL_NAME`, default `Qwen/Qwen3.5-4B` to match `configs/pilot.yaml`; override if your pilot uses another id):

```bash
export MODEL_NAME="Qwen/Qwen3.5-4B"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$MODEL_NAME')"
```

Store the model on the network volume (e.g. under `/workspace`) so it persists.

### 4. Run unit tests on the pod (no GPU needed)

Same as locally; all tests use mocks:

```bash
cd /workspace/metacog-llm-compute
pip install pytest pytest-cov  # if not already installed by setup_cloud.sh
python -m pytest tests/ -v
```

All tests should pass (run `pytest` locally to see the current count). This confirms the codebase and interfaces before you run the real pilot.

### 5. Run the pilot with real model (GPU workload)

Run the pilot **with real inference** on the pod (Pilot 2):

```bash
cd /workspace/metacog-llm-compute
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir /workspace/metacog-llm-compute/data/results --pilot-mode cuda
```

Or use `--real` to auto-detect (on the pod this will select `cuda`).

You should see:

- `data/results/pilot_benchmark.json` — real inference speed, VRAM, latency
- `data/results/pilot_calibration.json` — episode data with TLE, VC, compute stage
- `data/results/pilot_cost_validation.md` — measured vs expected throughput and budget note
- `data/results/pilot_feasibility_report.md` — Go/No-Go checklist

### 6. Download results from the pod

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

The gateway URL (`ssh.runpod.io`) is fine for **interactive** `ssh`; for **scp/rsync** use **TCP** or workarounds (HTTP server on the pod, etc.).

After download, look for `pilot_benchmark.json` directly under `data/results/runpod_pilot/` (not `…/runpod_pilot/results/`). The pod must be **running** for `scp` to succeed.

Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).

---

## TextWorld Cooking dataset (thesis exploration domain)

Core experiments use **TextWorld Cooking** (partial observability / exploration) and **Tower of Hanoi** (full observability / planning). TextWorld games are **not** bundled inside the library: you **generate** compiled story files (Inform 7 **`.z8`**, or legacy **`.ulx`**) and load them by path. This repo builds a reproducible dataset under `data/tasks/textworld/`.

### Generation (`scripts/generate_textworld_games.py`)

Generates TextWorld Cooking games via `python -m textworld.challenges.tw_cooking` with:

- **CLI:** `--num-rooms`, `--num-ingredients`, `--cut`, `--cook`, `--open`, `--num-instances`, `--seed` (master seed; each instance gets a deterministic sub-seed).
- **Output:** `textworld_{i}.z8`, TextWorld’s **serialized game** `textworld_{i}.json` (required at play time — do not overwrite), and experiment metadata **`textworld_{i}.meta.json`** in `--output-dir` (default `data/tasks/textworld/`).
- **`.meta.json`** includes generation parameters, seeds, walkthrough from game metadata (reference/debugging), entity list, `max_score`, and expected step count (walkthrough length).
- **Generation-time** `max_steps` is **50** (generous engine limit). The **agent** caps episodes via `episode.max_steps_per_episode` in config (e.g. 20–25); do not rely on the game file alone for the experiment cap.

Optional: `--write-manifest` writes `difficulty_manifest.json` in the output directory (useful after the final 50-instance run).

Example (from repo root; requires `textworld` installed):

```bash
python scripts/generate_textworld_games.py \
  --num-rooms 5 \
  --num-ingredients 2 \
  --cook \
  --seed 42 \
  --num-instances 3
```

This writes `data/tasks/textworld/textworld_0.z8` … `textworld_2.z8`, matching `textworld_{i}.json` (game dump) and `textworld_{i}.meta.json`. Add `--cut` for cutting, or omit `--cook` for a take-only style run, depending on the sweep cell you are testing.

### Difficulty sweep (`scripts/sweep_textworld_difficulty.py`)

Runs a grid over rooms `{3,5,7}` × ingredients `{1,2,3}` × operations `{take-only, take+cook, take+cut+cook}`, generates a small batch per cell, runs **C0** episodes via the existing agent loop, and writes **`sweep_results.json`** plus a ranked console summary. Goal: land near **30–50% C0 task success** and **8–15 steps** mean episode length before freezing parameters for the final 50 instances. Use `--real` with `configs/experiment_core.yaml` (or your sweep config) so the model matches the thesis run (e.g. Qwen2.5-3B-Instruct on GPU).

### Manual play (`scripts/play_textworld.py`)

Interactive terminal play for a single compiled story (`.z8` or `.ulx`): observations, score vs max score, and termination as `YOU WON` / `YOU LOST` / `MAX STEPS REACHED`. Use this to sanity-check generated games before long sweeps. Requires `textworld` (same stack as generation).

Example (from repo root; path matches instances produced by `generate_textworld_games.py`):

```bash
python scripts/play_textworld.py data/tasks/textworld/textworld_0.z8
```

Use `--max-steps N` to change the interactive cap for the session (default 50).

### Final manifest (`scripts/build_textworld_manifest.py`)

After generating the **50** immutable instances with chosen parameters, build **`data/tasks/textworld/difficulty_manifest.json`**: metadata per instance, difficulty tier, and **`holdout: true/false`** for **5 / 45** split (Phase 1 threshold tuning vs Phase 2). Policies: `--holdout-policy first-n` or `mod-10`.

### `TextWorldEnv` and step correctness

`src/environments/textworld_env.py` loads a `.z8`/`.ulx` and, when present, the **`textworld_{i}.meta.json` sidecar** (walkthrough kept for reference only; legacy `textworld_{i}.json` experiment-only files are still detected if they contain `generation_parameters`). Step labels use the **game engine**, not walkthrough matching (Cooking allows multiple valid orderings):

- **`optimal`:** score increased after the step.
- **`legal`:** admissible / no parser error, score unchanged.
- **`illegal`:** unrecognized command or error feedback.

Phase runners resolve games from `paths.tasks_dir` in `experiment_core.yaml`: either `data/tasks/textworld_{i}.z8`/`.ulx` or **`data/tasks/textworld/textworld_{i}.z8`/`.ulx`** (see `src/utils/experiment_env.py`).

---

## Tower of Hanoi (manual play)

The planning domain is implemented in `src/environments/tower_of_hanoi.py` (same `reset()` / `step(action)` API as other envs). To verify parsing, legality, and goal detection **without a model**, use the interactive CLI:

```bash
python scripts/play_tower_of_hanoi.py
```

Defaults: **3 disks**, classic start (all disks on peg A), seed **42**, step cap from the generated task. Moves accept the same text as the agent (e.g. `A->C`, `A → C`, or “move disk from A to C”). Type `quit`, `exit`, or `:q` to leave.

Useful flags:

| Flag | Meaning |
|------|---------|
| `--num-disks N` | Disk count (default 3). |
| `--seed SEED` | Instance generation seed (default 42). |
| `--partial-moves K` | Start after **K** optimal moves from the classic tower-on-A state (simulates mid-game; default 0). |
| `--max-steps N` | Override the episode step cap (default: from generated task). |

Example with a scrambled start:

```bash
python scripts/play_tower_of_hanoi.py --num-disks 3 --partial-moves 5 --seed 0
```

---

## Test suite layout (what each test file does)

| Test file | What it does (unit test) | Pilot counterpart |
|-----------|--------------------------|--------------------|
| `tests/test_01_inference_speed.py` | Mock benchmark: 50 “prompts”, assert result has `tokens_per_sec`, `latency_mean`, positive tok/s. | With `--real`: real 50 prompts, real tok/s and VRAM. |
| `tests/test_02_token_entropy.py` | Unit tests for `token_entropy`: synthetic logprobs, TLE higher for “hard” than “easy”. | Pilot uses same signal code on synthetic data. |
| `tests/test_03_verbalized_confidence.py` | Unit tests for `parse_confidence()`: “Confidence: 85” → 85, unparseable → `None`. | Pilot uses same parser on sample strings. |
| `tests/test_04_textworld_env.py` | `TextWorldEnv` interface: `reset()`, `step()`, `.observation`, `.done` (stub). | Pilot runs reset/step; env can be stub or real game if file provided. |
| `tests/test_05_e2e_mini_experiment.py` | 6 episodes with mock env and mock model; assert episode dict keys and JSON output. | With `--real`: same loop with real model (and optional real env). |
| `tests/test_06_logging_and_analysis.py` | Episode round-trip and ECE on synthetic data. | Pilot writes calibration JSON and computes ECE on episode data. |

Shared fixtures (mock model, mock env, sample episode data, temp dir) are in `tests/conftest.py`.

---

## Current implementation (high level)

- **Model:** `src/utils/model_wrapper.py` provides `VLLMWrapper`, `HFWrapper`, and `LMStudioWrapper` (OpenAI HTTP for LM Studio); `create_wrapper(backend, model_name, dtype)` returns the right wrapper. Pilot uses it when a real backend is requested; CUDA mode fails fast if the wrapper cannot load.
- **Pilot script:** `run_pilot.py` runs inference benchmarks, TLE/VC checks, TextWorld e2e, Tower of Hanoi parseability, and feasibility JSON; optional `--only` runs a subset. See `configs/pilot.yaml`.
- **TextWorld:** `TextWorldEnv` loads real `.z8`/`.ulx` games via `textworld.gym` when the file exists; loads optional `.meta.json` sidecars; score-based step correctness for real games. Generate Cooking datasets with `scripts/generate_textworld_games.py` into `data/tasks/textworld/`. See **TextWorld Cooking dataset** above.
- **Tower of Hanoi:** `TowerOfHanoiEnv` in `src/environments/tower_of_hanoi.py`. Interactive sanity check: `scripts/play_tower_of_hanoi.py` (no model; same move parsing as experiments).
- **Reports:** Pilot writes per-step JSON and `pilot_feasibility.json` under the results directory.
- **Phase 1 / 2:** `run_phase1.py` and `run_phase2.py` support checkpointing and `--resume`; use `--real` for GPU/vLLM runs.

---

## Project structure

```
configs/          # pilot.yaml, experiment_core.yaml, experiment_ext.yaml
src/
  agent/          # base_agent, compute_stages, allocator
  signals/        # token_entropy, verbalized_confidence, semantic_consistency
  environments/   # textworld_env, tower_of_hanoi, delayed_cue (legacy), logical_reasoning
  analysis/       # calibration, comparison, visualization
  utils/          # logging_utils, model_wrapper, checkpointing
scripts/          # run_pilot/phase1/phase2, setup_cloud.sh, generate_textworld_games,
                  # sweep_textworld_difficulty, play_textworld, play_tower_of_hanoi, build_textworld_manifest
data/tasks/       # e.g. data/tasks/textworld/ — TextWorld Cooking .z8 + Game .json + .meta.json, difficulty_manifest.json
data/results/     # pilot_test*.json, pilot_feasibility.json, ep_*.json, phase1/2 checkpoints
tests/            # conftest.py, test_01_* … test_06_*
```

---

## References

- **Infrastructure and pilot:** `blueprints/infrastructureplan_pilot.md` (Section V: Pilot, Section VI: Code structure)
- **Thesis design:** `blueprints/thesis_design.md`
