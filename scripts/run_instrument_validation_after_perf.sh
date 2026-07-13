#!/usr/bin/env bash
# Gate C sequence after perf/vllm-server-concurrency (pod).
# Prereq: vLLM server running with flags from docs/runpod.md (Terminal A).
# Usage: bash scripts/run_instrument_validation_after_perf.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/data/results}"
export PATH="${VENV_BIN:-/root/venv-metacog/bin}:$PATH"
PYTHON="${PYTHON:-python}"

IV="$RESULTS_DIR/instrument_validation"
mkdir -p "$IV" /root/logs

if [[ -f /workspace/secrets/env.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/secrets/env.sh
fi

echo "== Preflight =="
bash scripts/instrument_validation_preflight.sh

echo "== Git: perf branch =="
git fetch origin
git checkout perf/vllm-server-concurrency
git pull --ff-only origin perf/vllm-server-concurrency
git rev-parse HEAD

if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "ERROR: start vLLM in another terminal (see docs/runpod.md serve command)"
  exit 1
fi

echo "== Throughput re-sweep =="
$PYTHON scripts/measure_concurrent_throughput.py --real \
  --candidates "${SWEEP_CANDIDATES:-8,16,24,32}" \
  --output "$IV/throughput_sweep_post_perf.json" \
  --output-dir "$IV/throughput_sweep_post_perf" \
  2>&1 | tee /root/logs/throughput_sweep_post_perf.log

$PYTHON scripts/apply_production_n.py "$IV/throughput_sweep_post_perf.json"

echo "== C-1 Backend parity (hard stop) =="
$PYTHON scripts/verify_backend_parity.py --backend server \
  --config configs/experiment_core.yaml \
  --output-dir "$IV" \
  --freeze-metadata-dir "$IV" \
  2>&1 | tee /root/logs/backend_parity_post_perf.log
if ! grep -q "Overall: PASS" /root/logs/backend_parity_post_perf.log 2>/dev/null; then
  echo "STOP: backend parity failed"
  exit 1
fi

echo "== C-2/C-4 format_vc_probe =="
$PYTHON scripts/run_phase1.py --config configs/dev/format_vc_probe.yaml --real \
  --checkpoint-dir "$IV" \
  2>&1 | tee /root/logs/format_vc_probe_post_perf.log
PROBE=$(ls -td "$IV"/phase1_* 2>/dev/null | head -1)
$PYTHON scripts/audit_pilot_signals.py "$PROBE" --json | tee "$PROBE/audit_signals.json"
$PYTHON -m src.analysis.preanalysis_screen "$PROBE"

echo "== C-3 toh_parse_probe =="
$PYTHON scripts/run_phase1.py --config configs/dev/toh_parse_probe.yaml --real \
  --checkpoint-dir "$IV" \
  2>&1 | tee /root/logs/toh_parse_probe_post_perf.log

echo "== C-5 signal_smoke (only if C-2/C-4 acceptable) =="
read -r -p "Continue to signal_smoke (~72 ep)? [y/N] " ans
if [[ "${ans:-}" == "y" || "${ans:-}" == "Y" ]]; then
  $PYTHON scripts/run_phase1.py --config configs/dev/signal_smoke.yaml --real \
    --checkpoint-dir "$IV" \
    2>&1 | tee /root/logs/signal_smoke_post_perf.log
  RUN=$(ls -td "$IV"/phase1_* 2>/dev/null | head -1)
  $PYTHON -m src.analysis.preanalysis_screen "$RUN"
  echo "== C-6 topk sweep =="
  $PYTHON scripts/sweep_topk_sensitivity.py "$RUN" --output "$RUN/topk_sensitivity.json"
fi

echo "== Done. Update docs/instrument_validation_session.md and consistency_log.md =="
