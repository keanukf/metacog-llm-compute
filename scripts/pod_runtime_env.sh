#!/usr/bin/env bash
# RunPod template env (PID1) points HF/pip/uv caches at /workspace/.cache — fills the
# network volume. Source this from setup_cloud.sh and activate_pod_env.sh to force
# ephemeral caches onto the container disk instead.

export VENV_DIR="/root/venv-metacog"
export HF_HOME="/root/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="/root/.cache/pip"
export UV_CACHE_DIR="/root/.cache/uv"
export VIRTUALENV_OVERRIDE_APP_DATA="/root/.cache/virtualenv"

# RunPod may set XDG_CACHE_HOME=/workspace/.cache; drop it so tools fall back to $HOME.
unset XDG_CACHE_HOME

export RESULTS_DIR="${RESULTS_DIR:-/workspace/metacog-llm-compute/data/results}"
