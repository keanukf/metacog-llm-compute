# Architecture and Dataflow

## Documentation map

- [`docs/runbook.md`](runbook.md) — quality loop, mock pilot, `data/` layout
- [`docs/pilot.md`](pilot.md) — pilot modes and CLI
- [`docs/runpod.md`](runpod.md) — cloud GPU workflow
- [`docs/textworld.md`](textworld.md) — TextWorld dataset
- [`docs/scripts.md`](scripts.md) — script catalog
- [`docs/artifact_schema.md`](artifact_schema.md) — episode JSON contract
- [`docs/adrs.md`](adrs.md) — decision log
- [`configs/README.md`](../configs/README.md) — YAML reference

## Runtime Flow

1. A runner script (`scripts/experiment/run_pilot.py`, `scripts/experiment/run_phase1.py`, `scripts/experiment/run_phase2.py`) loads YAML config.
2. Environment instances are created via `src/utils/experiment_env.py`.
3. Agent loop (`src/agent/base_agent.py`) executes step-by-step with compute stage selection.
4. Stage execution (`src/agent/stages/` — `c0`, `c1`, `c2`; facade `compute_stages.py`) produces action + optional TLE/VC signals.
5. Episode artifacts are written through `src/utils/logging_utils.py`.
6. Analysis loaders (`src/analysis/datasets.py`) flatten artifacts into episode/step tables.

## Agent Layer Boundaries

- `src/agent/base_agent.py`: stable episode APIs (`run_episode`, `run_adaptive_episode`)
- `src/agent/allocator.py`: strategy-based stage assignment
- `src/agent/stages/`: C0/C1/C2 implementations; `compute_stages.py` re-exports `get_step_fn` and constants

## Signal and Analysis Layer

- `src/signals/token_entropy.py`: TLE extraction and aggregation
- `src/signals/verbalized_confidence.py`: VC parsing and structured output
- `src/analysis/*`: calibration/comparison/visualization and run dataset flattening

## Artifact Boundary

- Episode JSON files are the main contract between runtime and analysis.
- Sidecar artifacts (`logprobs/`, `vc/`, `trace_*.jsonl`) carry optional high-volume detail.
