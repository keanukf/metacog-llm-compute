# Metacognitive Effort Allocation in Sequential LM Agents

Thesis codebase: pilot tests, agent loop, signal extraction (TLE, VC), and experiment runners for RunPod Cloud GPU. See `blueprints/` for design and infrastructure.

---

## Unit tests vs pilot: what each is for

| | **Unit tests (pytest)** | **Pilot (run_pilot.py)** |
|---|-------------------------|---------------------------|
| **Purpose** | Check that **code and interfaces** are correct: signals, agent loop, logging, calibration logic. | Check **setup and hardware** in a small end-to-end run: real model, VRAM, throughput, full pipeline on GPU. |
| **Runs** | 21 functional tests with **mocks** (no model, no GPU). | One script that runs Tests 1–6 in sequence; optionally **real model** when you pass `--real`. |
| **Where** | Local or on pod; **no GPU required**. | Without `--real`: mock (local/pod, no GPU). With `--real`: **needs GPU** (e.g. RunPod). |
| **When** | After every code change; in CI. | Before spending GPU budget: confirm vLLM, tok/s, VRAM, and that episodes/logs are produced. |
| **Output** | Pass/fail per test. | `pilot_benchmark.json`, `pilot_calibration.json`, and optionally `pilot_cost_validation.md`, `pilot_feasibility_report.md`. |

**Summary:** Unit tests validate *logic*; the pilot validates *environment and hardware* in a small run so you can detect errors and confirm throughput before starting full Phase 1/2 experiments.

---

## Running unit tests (no GPU)

All 21 tests use mocks and run without GPU or cloud. From the repo root:

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

The pilot runs Tests 1–6 in sequence and writes benchmark and calibration outputs. It exists in two modes:

### Pilot with mocks (no GPU)

Default: no real model, stub environments. Use this to confirm the script and outputs locally:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results
```

You get `pilot_benchmark.json` and `pilot_calibration.json` with **mock** numbers (e.g. very high tokens_per_sec). Reports (`pilot_cost_validation.md`, `pilot_feasibility_report.md`) are written from config paths.

### Pilot with real model (GPU required)

On a machine with CUDA (e.g. RunPod RTX 3090), use `--real` to load the configured model (vLLM or HuggingFace) and run real inference:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --real
```

Or set `USE_REAL_MODEL=1`. With `--real`:

- **Test 1:** 50 real prompts → measured tokens/s, latency, VRAM.
- **Tests 2–3:** Unchanged (synthetic/sample data).
- **Test 4:** TextWorld env (stub or real game if a game file is provided).
- **Test 5:** Same model runs the e2e episode loop (stub or real env).
- **Test 6:** ECE and logging on the episode data.

After a real pilot run you should see realistic `tokens_per_sec` in `pilot_benchmark.json` and full episode data in `pilot_calibration.json`. Use the generated markdown reports for go/no-go and cost validation.

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

**Optional — pre-download the model** (saves time during the pilot; uncomment in `scripts/setup_cloud.sh` or run):

```bash
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
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

All 21 tests should pass. This confirms the codebase and interfaces before you run the real pilot.

### 5. Run the pilot with real model (GPU workload)

Run the pilot **with real inference** to check setup and hardware:

```bash
cd /workspace/metacog-llm-compute
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir /workspace/metacog-llm-compute/data/results --real
```

You should see:

- `data/results/pilot_benchmark.json` — real inference speed, VRAM, latency
- `data/results/pilot_calibration.json` — episode data with TLE, VC, compute stage
- `data/results/pilot_cost_validation.md` — measured vs expected throughput and budget note
- `data/results/pilot_feasibility_report.md` — Go/No-Go checklist

### 6. Download results from the pod

From your **local machine**:

```bash
rsync -avz -e "ssh -i /path/to/key" \
  user@POD_IP:/workspace/metacog-llm-compute/data/results/ ./data/results/
```

Or use `scp`. Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).

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

## Current implementation (no further “implement the pilot” steps)

- **Model:** `src/utils/model_wrapper.py` provides `VLLMWrapper` and `HFWrapper`; `create_wrapper(backend, model_name, dtype)` returns the right wrapper. Pilot uses it when `--real` is set.
- **Pilot script:** `run_pilot.py` supports `--real` and `USE_REAL_MODEL=1`; Test 1 runs real inference and measures tok/s/VRAM when a real wrapper is used; Test 5 uses the same wrapper for e2e episodes.
- **TextWorld:** `TextWorldEnv` loads real games when `game_file` is set and the file exists (via `textworld.gym`); otherwise uses the stub. `scripts/generate_textworld_games.py` can generate small games into `data/tasks/`.
- **Reports:** Pilot writes `pilot_cost_validation.md` and `pilot_feasibility_report.md` when paths are set in config.
- **Phase 1:** `scripts/run_phase1.py` runs the full episode loop with checkpointing and `--resume`; use `--real` for real model on GPU.

---

## Project structure

```
configs/          # pilot.yaml, experiment_core.yaml, experiment_ext.yaml
src/
  agent/          # base_agent, compute_stages, allocator
  signals/        # token_entropy, verbalized_confidence, semantic_consistency
  environments/   # textworld_env, delayed_cue, logical_reasoning
  analysis/       # calibration, comparison, visualization
  utils/          # logging_utils, model_wrapper, checkpointing
scripts/          # run_pilot.py, run_phase1.py, run_phase2.py, setup_cloud.sh, generate_textworld_games.py
data/tasks/       # generated task instances (e.g. TextWorld games)
data/results/     # pilot_benchmark.json, pilot_calibration.json, pilot_*.md, phase1/2 episode JSONs
tests/            # conftest.py, test_01_* … test_06_*
```

---

## References

- **Infrastructure and pilot:** `blueprints/infrastructureplan_pilot.md` (Section V: Pilot, Section VI: Code structure)
- **Thesis design:** `blueprints/thesis_design.md`
