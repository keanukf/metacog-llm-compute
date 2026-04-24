#!/usr/bin/env bash
# RunPod Pod setup: install pinned Python deps from requirements.txt (reproducible),
# optionally pre-download a model checkpoint to the HF cache on a persistent volume.
#
# Run from repo root on the pod after clone/upload.
# Usage: bash scripts/setup_cloud.sh

set -euo pipefail

echo "Upgrading pip (recommended on fresh pods)..."
python -m pip install --upgrade pip

echo "Installing pinned Python dependencies from requirements.txt..."
python -m pip install -r requirements.txt

# Optional: pre-download weights/tokenizer into the HF cache (persistent volume recommended).
# - Set SKIP_MODEL_DOWNLOAD=1 to skip.
# - Set MODEL_NAME to the HF repo id you will actually run first.
#   If unset, we default to the first entry in configs/models_runpod.yaml (single source of truth).
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
MODEL_NAME="${MODEL_NAME:-}"

if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME="$(python - <<'PY'
try:
    import yaml
except Exception:
    print("Qwen/Qwen3-8B")
    raise SystemExit(0)

with open("configs/models_runpod.yaml", "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}

models = None
if isinstance(raw, dict):
    models = raw.get("models")
elif isinstance(raw, list):
    models = raw

if isinstance(models, list) and models and isinstance(models[0], str) and models[0].strip():
    print(models[0].strip())
else:
    print("Qwen/Qwen3-8B")
PY
)"
fi

if [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]]; then
  echo "Skipping model pre-download (SKIP_MODEL_DOWNLOAD=1)."
else
  echo "Pre-downloading tokenizer+weights for MODEL_NAME=${MODEL_NAME} (this can take a while)..."
  python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "${MODEL_NAME}"
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Pre-download done:", model_name)
PY
fi

echo "Setup done."
echo "Next:"
echo "  python scripts/hf_model_card_gate.py --models-file configs/models_runpod.yaml"
echo "  python scripts/run_pilot.py --config configs/pilot.yaml --pilot-mode cuda --real --model-name \"${MODEL_NAME}\""
