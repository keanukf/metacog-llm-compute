#!/usr/bin/env bash
# RunPod pod setup: GitHub deploy key, venv on container disk, pinned deps, optional model pre-download.
#
# Run from repo root:
#   cd /workspace/metacog-llm-compute
#   bash scripts/setup_cloud.sh
#
# Optional env:
#   ENV_FILE=/workspace/secrets/env.sh     — HF_TOKEN, Langfuse, etc. (default)
#   DEPLOY_KEY=/workspace/secrets/runpod_github_ed25519
#   GIT_BRANCH=<name>                      — optional: checkout this branch before pull
#   SKIP_GIT_SYNC=1                        — skip git fetch/pull (deploy key still installed)
#   SKIP_VENV=1                            — skip venv create (use existing PATH python)
#   SKIP_DEPLOY_KEY=1                      — skip SSH key install
#   SKIP_MODEL_DOWNLOAD=1                  — skip HF weight pre-download
#   SKIP_WORKSPACE_CACHE_CLEAN=1           — do not remove stale /workspace/.cache/*
#   MODEL_NAME=Qwen/Qwen3-8B               — override model id for pre-download
#
# RunPod's container template pre-sets HF_HOME/PIP_CACHE_DIR under /workspace/.cache.
# scripts/pod_runtime_env.sh forces ephemeral caches onto the container disk.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/pod_runtime_env.sh"

ENV_FILE="${ENV_FILE:-/workspace/secrets/env.sh}"
DEPLOY_KEY="${DEPLOY_KEY:-/workspace/secrets/runpod_github_ed25519}"

_clean_workspace_caches() {
  if [[ "${SKIP_WORKSPACE_CACHE_CLEAN:-0}" == "1" ]]; then
    echo "Skipping /workspace/.cache cleanup (SKIP_WORKSPACE_CACHE_CLEAN=1)."
    return 0
  fi
  if [[ ! -d /workspace/.cache ]]; then
    return 0
  fi
  local size_mb
  size_mb="$(du -sm /workspace/.cache 2>/dev/null | awk '{print $1}')"
  if [[ "${size_mb:-0}" -lt 1 ]]; then
    return 0
  fi
  echo "Removing stale RunPod template caches from network volume: /workspace/.cache (${size_mb} MB)..."
  rm -rf /workspace/.cache/pip /workspace/.cache/huggingface /workspace/.cache/uv /workspace/.cache/virtualenv
  rmdir /workspace/.cache 2>/dev/null || true
}

_clean_repo_tool_caches() {
  local removed=0
  for dir in .mypy_cache .pytest_cache .ruff_cache; do
    if [[ -d "${REPO_ROOT}/${dir}" ]]; then
      rm -rf "${REPO_ROOT}/${dir}"
      removed=1
    fi
  done
  if [[ "${removed}" == "1" ]]; then
    echo "Removed local tool caches (.mypy_cache, .pytest_cache, .ruff_cache) from repo on /workspace."
  fi
}

_print_disk_layout() {
  echo ""
  echo "Disk layout (network volume should stay small — code + results only):"
  df -h / /workspace 2>/dev/null || true
  echo "  /workspace usage:"
  du -sh /workspace/* /workspace/.??* 2>/dev/null | sort -hr | head -10 || true
  echo "  container caches:"
  du -sh "${VENV_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" 2>/dev/null || true
}

_reassert_runtime_env() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/pod_runtime_env.sh"
}

_install_deploy_key() {
  if [[ "${SKIP_DEPLOY_KEY:-0}" == "1" ]]; then
    echo "Skipping deploy key (SKIP_DEPLOY_KEY=1)."
    return 0
  fi
  if [[ ! -f "${DEPLOY_KEY}" ]]; then
    echo "No deploy key at ${DEPLOY_KEY}; skipping GitHub SSH setup."
    return 0
  fi
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  install -m 600 "${DEPLOY_KEY}" ~/.ssh/id_ed25519_runpod
  if ! grep -q "Host github.com" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_runpod
  IdentitiesOnly yes
EOF
    chmod 600 ~/.ssh/config
  fi
  echo "GitHub deploy key installed from ${DEPLOY_KEY}"
}

_sync_git() {
  if [[ "${SKIP_GIT_SYNC:-0}" == "1" ]]; then
    echo "Skipping git sync (SKIP_GIT_SYNC=1)."
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not a git repo; skipping git sync."
    return 0
  fi
  git fetch origin
  if [[ -n "${GIT_BRANCH:-}" ]]; then
    echo "Checking out GIT_BRANCH=${GIT_BRANCH}..."
    git checkout "${GIT_BRANCH}"
    git merge --ff-only "origin/${GIT_BRANCH}"
  else
    local branch
    branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -z "${branch}" ]]; then
      echo "Detached HEAD; skipping pull (checkout a branch or set GIT_BRANCH)."
      return 0
    fi
    echo "Fast-forward pull on current branch: ${branch}"
    git merge --ff-only "origin/${branch}"
  fi
  echo "HEAD=$(git rev-parse --short HEAD) ($(git branch --show-current 2>/dev/null || echo detached))"
}

_ensure_venv() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    export PATH="${VENV_DIR}/bin:${PATH}"
    echo "Using venv: ${VENV_DIR} ($(python --version))"
    return 0
  fi
  if [[ "${SKIP_VENV:-0}" == "1" ]]; then
    echo "Using python from PATH (SKIP_VENV=1, no venv at ${VENV_DIR}): $(command -v python || true)"
    return 0
  fi
  echo "Creating venv at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  echo "Using venv: ${VENV_DIR} ($(python --version))"
}

_clean_workspace_caches
_clean_repo_tool_caches

if [[ -f "${ENV_FILE}" ]]; then
  echo "Loading secrets from ${ENV_FILE}..."
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "No env file found at ${ENV_FILE}; continuing without loading secrets."
fi

_reassert_runtime_env

_install_deploy_key
_sync_git
_ensure_venv

echo "Runtime cache dirs (container disk):"
echo "  HF_HOME=${HF_HOME}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "  VENV_DIR=${VENV_DIR}"

echo "Upgrading pip (recommended on fresh pods)..."
python -m pip install --upgrade pip

echo "Installing pinned Python dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "Sanity-checking key dependency versions..."
python - <<'PY'
import sys

def _get(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "UNKNOWN"

vllm_v = _get("vllm")
tx_v = _get("transformers")
torch_v = _get("torch")
print("vllm:", vllm_v)
print("transformers:", tx_v)
print("torch:", torch_v)
if vllm_v != "UNKNOWN" and not vllm_v.startswith("0.19."):
    print("WARNING: expected vllm 0.19.x (update requirements.txt if intentional).", file=sys.stderr)
PY

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
  echo "Pre-downloading tokenizer+weights for MODEL_NAME=${MODEL_NAME} into HF_HOME=${HF_HOME}..."
  python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "${MODEL_NAME}"
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Pre-download done:", model_name)
PY
fi

echo ""
echo "Setup done — pod runtime ready."
echo "  branch: $(git branch --show-current 2>/dev/null || echo n/a)"
echo "  HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "  python: $(command -v python) ($(python --version 2>&1))"
echo "  HF_HOME: ${HF_HOME}"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo ""
echo "New SSH session:"
echo "  source scripts/activate_pod_env.sh"
echo ""
echo "When you need inference: start vLLM serve (see docs/runpod.md)."

_print_disk_layout
