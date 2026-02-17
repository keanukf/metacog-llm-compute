# Metacognitive Effort Allocation in Sequential LM Agents

Thesis codebase: pilot tests, agent loop, signal extraction (TLE, VC), and experiment runners for RunPod Cloud GPU. See `blueprints/` for design and infrastructure.

---

## Running tests locally (no GPU)

All pilot tests run without GPU or cloud by using mocks. From the repo root:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Optional: run a single test file or add markers:

```bash
python -m pytest tests/test_02_token_entropy.py -v
python -m pytest tests/ -v -m "not slow"
```

---

## Running tests on RunPod Cloud GPU

The pilot is designed for **~2 GPU-hours** on an **RTX 3090** (24 GB VRAM) to validate the full pipeline and compute estimates. Budget: about **$0.44** (see `blueprints/infrastructureplan_pilot.md` Section V).

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

### 4. Run the unit/integration test suite on the pod

Same as locally; no GPU required for the current tests (they use mocks):

```bash
cd /workspace/metacog-llm-compute
pip install pytest pytest-cov  # if not already installed by setup_cloud.sh
python -m pytest tests/ -v
```

All 21 tests should pass. This checks that the codebase and interfaces are correct before running the real pilot.

### 5. Run the pilot script (real GPU workload)

The pilot runs Tests 1–6 **with a real model** (when implemented) and writes benchmark and calibration outputs:

```bash
cd /workspace/metacog-llm-compute
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir /workspace/metacog-llm-compute/data/results
```

**Current state:** `run_pilot.py` uses a **mock model** and stub environments, so it runs without GPU and produces placeholder JSON. To turn it into the real pilot you must:

1. **Implement a vLLM (or HF) wrapper** in `src/utils/model_wrapper.py` that loads Qwen2.5-3B and exposes `generate(prompt, logprobs=True)`.
2. **Wire that wrapper** into `run_pilot.py` instead of the mock (e.g. in `run_test1_inference_speed`, `run_test5_e2e`).
3. **Use real TextWorld games** in Test 4 and Test 5 (generate or copy game files and use the TextWorld API in `src/environments/textworld_env.py`).

After the real pilot run you should see:

- `data/results/pilot_benchmark.json` — inference speed, VRAM, latency
- `data/results/pilot_calibration.json` — 15 episodes with TLE, VC, correctness, compute stage
- Optionally: `pilot_cost_validation.md`, `pilot_feasibility_report.md` (if you add those writers in the script)

### 6. Download results from the pod

From your **local machine**:

```bash
# Download only results (adjust paths and SSH key)
rsync -avz -e "ssh -i /path/to/key" \
  user@POD_IP:/workspace/metacog-llm-compute/data/results/ ./data/results/
```

Or use `scp`:

```bash
scp -i /path/to/key -r user@POD_IP:/workspace/metacog-llm-compute/data/results/*.json ./data/results/
```

Then run analysis locally (e.g. ECE on `pilot_calibration.json` via `src/analysis/calibration.py`).

---

## Test suite layout (pilot Tests 1–6)

| Test file | Pilot test | What it does |
|-----------|------------|--------------|
| `tests/test_01_inference_speed.py` | Test 1 — Inferenzgeschwindigkeit | With a mock: run 50 “prompts”, assert result has `tokens_per_sec`, `latency_mean`. On GPU: replace with real vLLM benchmark. |
| `tests/test_02_token_entropy.py` | Test 2 — Token-Entropie-Extraktion | Unit tests for `src/signals/token_entropy.py`: synthetic logprobs, TLE higher for “hard” than “easy”. No GPU. |
| `tests/test_03_verbalized_confidence.py` | Test 3 — Verbalisierte Konfidenz | Unit tests for `parse_confidence()`: strings like “Confidence: 85” or “0-100: 70” → numeric 0–100; unparseable → `None`. No GPU. |
| `tests/test_04_textworld_env.py` | Test 4 — TextWorld Mini-Environment | `TextWorldEnv` interface: `reset()`, `step(action)`, `.observation`, `.done`. Uses stub env (no TextWorld install required). |
| `tests/test_05_e2e_mini_experiment.py` | Test 5 — End-to-End Mini-Experiment | 2×3×1 = 6 episodes with mock env and mock model; assert 6 JSON files with `steps`, `lm_calls`, `tokens`, `task_success`, TLE/VC. On GPU: use real env and model. |
| `tests/test_06_logging_and_analysis.py` | Test 6 — Logging & Download | Episode dict round-trip via `logging_utils`; ECE from `calibration.compute_ece()` on 15 synthetic points. No GPU. |

Shared fixtures (mock model, mock env, sample episode data, temp dir) live in `tests/conftest.py`.

---

## Implementing the real pilot on RunPod

1. **Model wrapper**  
   In `src/utils/model_wrapper.py`, implement a class that:
   - Loads Qwen2.5-3B-Instruct (vLLM or HuggingFace).
   - Exposes `generate(prompt, logprobs=False, max_tokens=256, temperature=0.3)` returning `(text, logprobs_or_none)`.
   - For vLLM: use the `logprobs` parameter. For HuggingFace: use `output_scores=True` (or equivalent).

2. **Pilot script**  
   In `scripts/run_pilot.py`:
   - Instantiate the real wrapper (e.g. from config or env).
   - In `run_test1_inference_speed`: run 50 real prompts, measure wall time and token count, compute tok/s and VRAM.
   - In `run_test5_e2e`: use the real wrapper and real TextWorld env instead of `MockModel` and stub env.

3. **TextWorld**  
   In `src/environments/textworld_env.py`: use the `textworld` package to load/generate small games (3–5 rooms), and implement `reset()` / `step(action)` against the TextWorld API so the agent loop in Test 4 and Test 5 runs on real games.

4. **Checklist**  
   After the run, fill the Go/No-Go checklist (see blueprint Section V) and optionally write `pilot_feasibility_report.md` and `pilot_cost_validation.md` from the script.

---

## Project structure (short)

```
configs/          # pilot.yaml, experiment_core.yaml, experiment_ext.yaml
src/
  agent/          # base_agent, compute_stages, allocator
  signals/        # token_entropy, verbalized_confidence, semantic_consistency
  environments/   # textworld_env, delayed_cue, logical_reasoning
  analysis/       # calibration, comparison, visualization
  utils/          # logging_utils, model_wrapper, checkpointing
scripts/          # run_pilot.py, run_phase1.py, run_phase2.py, setup_cloud.sh
data/tasks/       # generated task instances
data/results/     # pilot_benchmark.json, pilot_calibration.json, phase1/2 episode JSONs
tests/            # conftest.py, test_01_* … test_06_*
```

---

## References

- **Infrastructure and pilot:** `blueprints/infrastructureplan_pilot.md` (Section V: Pilot, Section VI: Code structure)
- **Thesis design:** `blueprints/thesis_design.md`
