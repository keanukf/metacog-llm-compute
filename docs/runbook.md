# Quality and Reproduction Runbook

Requires **Python 3.11+** (see `pyproject.toml`; CI uses 3.11).

**See also:** [`docs/pilot.md`](pilot.md), [`docs/scripts.md`](scripts.md), [`configs/README.md`](../configs/README.md).

## Local quality loop

1. Install dev setup:
   - `pip install -e ".[dev]"`
2. Run checks:
   - `ruff check src tests scripts`
   - `ruff format --check src tests scripts`
   - `mypy src`
   - `python -m pytest tests/ -v`

## Mock pilot checklist

After merging doc or config changes, confirm the pilot path still runs:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode mock --output-dir data/results
```

Expected: a timestamped folder `data/results/pilot_YYYYMMDD_HHMMSS/` with `pilot_test*.json`, `pilot_feasibility.json`, and episode JSON files (`ep_*.json`).

Optional integrity check:

```bash
python scripts/validate_pilot_outputs.py --pilot-dir data/results/pilot_YYYYMMDD_HHMMSS
```

## RunPod path (pinned)

1. Use pinned environment setup:
   - `bash scripts/setup_cloud.sh`
2. Run real CUDA pilot:
   - `python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode cuda --real`

Use `requirements.txt` as the pinned cloud install source. Full steps: [`docs/runpod.md`](runpod.md).

## `data/` layout

| Path | Contents | In git? |
|------|----------|---------|
| `data/tasks/` | TextWorld `.z8`, `.meta.json`, manifests | Task assets as generated (large files often local only) |
| `data/results/` | Pilot and experiment outputs (`pilot_*`, `ep_*.json`, checkpoints) | Typically gitignored run artifacts |

Pilot runs default to timestamped subfolders under `--output-dir` (e.g. `data/results/pilot_20250520_143022/`). See [`docs/artifact_schema.md`](artifact_schema.md) for JSON field contracts.

## Local notes (Cursor plans)

`.cursor/` is **gitignored** (IDE/agent plans stay local). If historical plan files were removed from your working tree after a merge, restore them from git history:

```bash
bash scripts/restore_cursor_plans.sh
```

Plans are not pushed to remote; `git pull` will not delete restored files while they remain untracked and ignored.
