#!/usr/bin/env bash
# Source on RunPod after setup_cloud.sh (new shell / new SSH session).
# Usage: source scripts/activate_pod_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/pod_runtime_env.sh"

export PATH="${VENV_DIR}/bin:${PATH}"

# setup_cloud.sh loads secrets (HF_TOKEN, LANGFUSE_*, ...) from ENV_FILE only into its own
# one-shot shell -- a fresh SSH session (e.g. every later `ssh runpod "..."` invocation) starts
# without them unless reloaded here too.
ENV_FILE="${ENV_FILE:-/workspace/secrets/env.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
