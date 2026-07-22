# Metacognitive Effort Allocation in Sequential LM Agents

## Documentation map

| Doc | Content |
|-----|---------|
| [`docs/architecture.md`](docs/architecture.md) | Runtime flow, layer boundaries |
| [`docs/runbook.md`](docs/runbook.md) | Quality loop, mock pilot checklist, `data/` layout |
| [`docs/pilot.md`](docs/pilot.md) | Pilot modes (mock / hf / cuda / lmstudio), flags, outputs |
| [`docs/runpod.md`](docs/runpod.md) | RunPod setup, CUDA pilot, download results |
| [`docs/textworld.md`](docs/textworld.md) | TextWorld dataset generation and manifest |
| [`docs/scripts.md`](docs/scripts.md) | All `scripts/` entry points with status tags |
| [`configs/README.md`](configs/README.md) | YAML files and `pilot.yaml` key reference |
| [`docs/adrs.md`](docs/adrs.md) | Harness decision log |
| [`docs/artifact_schema.md`](docs/artifact_schema.md) | Episode JSON contract |
| [`blueprints/`](blueprints/) | Thesis design, sample sizes, infrastructure |

## Repository goal

Thesis codebase for **metacognitive effort allocation in sequential language-model agents**. Proxy signals **token-level entropy (TLE)** and **verbalized confidence (VC)** drive per-step compute (**C0 / C1 / C2**). Environments include TextWorld and Tower of Hanoi; runners support local pilots and cloud experiments (RunPod).

| Location | Role |
|----------|------|
| `src/` | Package: `agent/`, `signals/`, `environments/`, `analysis/`, `utils/`, `pilot/` |
| `configs/` | YAML experiment and pilot configs |
| `scripts/` | CLI entry points, grouped into purpose subfolders (see [`docs/scripts.md`](docs/scripts.md)) |
| `tests/` | `pytest` with mocks (no GPU) |
| `data/` | Task assets (`data/tasks/`) and run outputs (`data/results/`) |
| `blueprints/` | Thesis design and infrastructure planning |

`scripts/` subfolders (full catalog in [`docs/scripts.md`](docs/scripts.md)):

| Subfolder | Contents |
|-----------|----------|
| `scripts/experiment/` | Production entry points: `run_pilot.py`, `run_phase1.py`, `run_phase2.py` |
| `scripts/datasets/` | Task-instance generation, manifests, interactive play |
| `scripts/difficulty_calibration/` | Difficulty tuning/sweeps/freezing probes |
| `scripts/instrument_validation/` | Backend parity, throughput, logprob-invariance checks |
| `scripts/analysis_rehearsal/` | Full-pipeline dry run + H3 power simulation |
| `scripts/run_readiness/` | Budget, run-hygiene, resume, output-QC, progress watcher |
| `scripts/pilot_analysis/` | Post-run pilot analysis, validation, summaries |
| `scripts/cloud/shell/`, `scripts/cloud/python/` | Pod setup/transfer (shell) + one download-repair helper (python) |

Run scripts from the **repository root**.

## Installation

Requires **Python 3.11+** (`pyproject.toml`; CI uses 3.11).

**Local development / CI (macOS or Linux, no vLLM):**

```bash
pip install -e ".[dev]"
```

**Local Mac + LM Studio pilot** (optional; adds inference client deps):

```bash
pip install -r requirements-local.txt
# or: pip install lmstudio httpx torch transformers accelerate
```

Do **not** expect `vllm` to install on macOS — use `--pilot-mode lmstudio` or `mock` locally; vLLM is for RunPod (`--pilot-mode cuda`).

**RunPod / cloud GPU (pinned, includes vLLM on Linux):**

```bash
pip install -r requirements.txt
```

`pyproject.toml` holds package metadata and tooling; `requirements.txt` is what [`scripts/cloud/shell/setup_cloud.sh`](scripts/cloud/shell/setup_cloud.sh) installs on pods.

## Quickstart

**1. Unit tests (no GPU):**

```bash
python -m pytest tests/ -v
```

**2. Mock pilot (sanity, no model):**

```bash
python scripts/experiment/run_pilot.py --config configs/pilot.yaml --pilot-mode mock --output-dir data/results
```

**3. Next steps:**

- Real hardware and modes → [`docs/pilot.md`](docs/pilot.md)
- RunPod CUDA path → [`docs/runpod.md`](docs/runpod.md)
- Quality gates → [`docs/runbook.md`](docs/runbook.md)

## Unit tests vs pilot

| | **Unit tests (pytest)** | **Pilot (`scripts/experiment/run_pilot.py`)** |
|---|-------------------------|----------------------------|
| **Purpose** | Code and interfaces (signals, agent loop, logging) | Setup and hardware in a small end-to-end run |
| **Runs** | Mocks, no GPU | Tests 1–6; `--pilot-mode` mock / cuda / lmstudio |
| **When** | After every change; in CI | Mock: anywhere; lmstudio: Mac/local API; cuda: RunPod vLLM |
| **Output** | Pass/fail | `pilot_test*.json`, `pilot_feasibility.json`, episode JSONs |

## Test suite (summary)

| Test file | Focus |
|-----------|--------|
| `test_inference_speed.py` | Benchmark structure / tok/s |
| `test_token_entropy.py` | TLE from logprobs |
| `test_verbalized_confidence.py` | VC parsing |
| `test_textworld_env.py` | TextWorld env API |
| `test_e2e_mini_experiment.py` | Mini episode loop + JSON keys |
| `test_logging_and_analysis.py` | Episode round-trip, ECE |
| `test_tower_of_hanoi_env.py` | Tower of Hanoi env API |
| `test_analysis_utils.py` | Analysis helper utilities |
| `test_analysis_pipeline.py` | End-to-end analysis pipeline |

The former `test_01`…`test_09` numeric prefixes were dropped in the 2026-07-21 refactor;
gate-jargon test names were renamed too (`test_gate_d_metrics.py` → `test_difficulty_metrics.py`,
`test_gate_d_manifest_smoke.py` → `test_manifest_success_smoke.py`,
`test_gate_d_abort_actions.py` → `test_inspect_abort_last_actions.py`).

Fixtures: `tests/conftest.py`. Pilot counterparts: [`docs/pilot.md`](docs/pilot.md).

## Implementation snapshot

- **Model:** `src/utils/inference/` — `VLLMWrapper`, `LMStudioWrapper` (responses-only); `model_wrapper.py` re-exports
- **Pilot:** `scripts/experiment/run_pilot.py` → orchestration in `src/pilot/`; config in `configs/pilot.yaml`
- **Agent:** `src/agent/base_agent.py` facade; compute stages in `src/agent/stages/` (`c0`, `c1`, `c2`; `compute_stages.py` facade)
- **Phase 1/2:** `scripts/experiment/run_phase1.py`, `scripts/experiment/run_phase2.py` (checkpointing; use after pilot)

## References

- [`blueprints/infrastructureplan_pilot.md`](blueprints/infrastructureplan_pilot.md) — pilot budget and code structure
- [`blueprints/thesis_design.md`](blueprints/thesis_design.md) — experiment design
