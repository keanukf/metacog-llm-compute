# Quality and Reproduction Runbook

## Local quality loop

1. Install dev setup:
   - `pip install -e ".[dev]"`
2. Run checks:
   - `ruff check src tests scripts`
   - `ruff format --check src tests scripts`
   - `mypy src`
   - `python -m pytest tests/ -v`

## Mock pilot reproduction

- `python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode mock --output-dir data/results`

Expected outputs include `pilot_test*.json`, `pilot_feasibility.json`, and episode JSON files.

## RunPod path (pinned)

1. Use pinned environment setup:
   - `bash scripts/setup_cloud.sh`
2. Run real CUDA pilot:
   - `python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode cuda --real`

Use `requirements.txt` as the pinned cloud install source.
