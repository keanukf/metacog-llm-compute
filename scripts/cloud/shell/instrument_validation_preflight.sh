#!/usr/bin/env bash
# Preflight for instrument-validation runs on RunPod (5090 + vLLM server).
# Usage: bash scripts/instrument_validation_preflight.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/data/results}"
export REPO_ROOT="$REPO_ROOT"

mkdir -p "$RESULTS_DIR/instrument_validation"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "WARN: nvidia-smi failed (not on GPU pod?)"
}

echo "== Repo =="
git fetch origin 2>/dev/null || true
git pull --ff-only 2>/dev/null || echo "WARN: git pull skipped or failed"
git rev-parse HEAD
python -V
pip show vllm 2>/dev/null | head -2 || echo "WARN: vllm not installed"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "== Loaded .env =="
fi

if [[ "${RUN_SETUP_CLOUD:-0}" == "1" ]]; then
  bash scripts/cloud/shell/setup_cloud.sh
fi

echo "== vLLM health =="
if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "vLLM server: OK"
else
  echo "vLLM server: not reachable on :8000 (start vllm serve in background terminal)"
fi

echo "== Done =="
echo "Results root: $RESULTS_DIR/instrument_validation"
