#!/usr/bin/env bash
# RunPod Pod setup: install deps and optionally download Qwen2.5-3B-Instruct.
# Run from repo root on the pod after clone/upload.
# Usage: bash scripts/setup_cloud.sh

set -e
echo "Installing Python dependencies..."
pip install vllm transformers textworld numpy pandas scipy pyyaml

# Optional: pre-download model to network volume (persistent)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$MODEL_NAME')"

echo "Setup done. Use: python scripts/run_pilot.py --config configs/pilot.yaml"
