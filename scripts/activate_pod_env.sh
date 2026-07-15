#!/usr/bin/env bash
# Source on RunPod after setup_cloud.sh (new shell / new SSH session).
# Usage: source scripts/activate_pod_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/pod_runtime_env.sh"

export PATH="${VENV_DIR}/bin:${PATH}"
