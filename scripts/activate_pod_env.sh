#!/usr/bin/env bash
# Source on RunPod after setup_cloud.sh (new shell / new SSH session).
# Usage: source scripts/activate_pod_env.sh

export VENV_DIR="${VENV_DIR:-/root/venv-metacog}"
export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"
export RESULTS_DIR="${RESULTS_DIR:-/workspace/metacog-llm-compute/data/results}"
