#!/usr/bin/env bash
# Download pilot/results from a RunPod pod to the local repo.
#
# RunPod documents two SSH modes:
#   • ssh.runpod.io — interactive shell only; NO SCP/SFTP support (downloads will fail).
#   • "SSH over exposed TCP" (root@IP -p PORT) — supports SCP/SFTP — use this for file copy.
#
# Usage (recommended — copy host/port from Pod → Connect → "SSH over exposed TCP"):
#   ./scripts/download_runpod_results.sh --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519
#   ./scripts/download_runpod_results.sh --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519 \
#     --run pilot_20250604_120000
#
# Or env (same values as dashboard):
#   RUNPOD_TCP_HOST=213.192.2.74 RUNPOD_TCP_PORT=40171 ./scripts/download_runpod_results.sh
#
# Optional: DEST_DIR, REMOTE_RESULTS, RUNPOD_TCP_USER (default root), SSH_KEY, RUN_FOLDER (--run)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST_DIR="${DEST_DIR:-${REPO_ROOT}/data/results/runpod_pilot}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/workspace/metacog-llm-compute/data/results}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
RUN_FOLDER=""

usage() {
  echo "Usage (recommended — TCP SSH from RunPod dashboard, supports scp):" >&2
  echo "  $0 --tcp <user> <host> <port> [ssh_key] [--run FOLDER]" >&2
  echo "  Example: $0 --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519" >&2
  echo "  Example: $0 --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519 --run pilot_20250604_120000" >&2
  echo "" >&2
  echo "Or: RUNPOD_TCP_HOST=... RUNPOD_TCP_PORT=... [RUNPOD_TCP_USER=root] $0 [--run FOLDER]" >&2
  echo "" >&2
  echo "Legacy (often broken): gateway user@ssh.runpod.io — RunPod does NOT support SCP on that URL." >&2
  exit 1
}

flatten_nested_results() {
  local dest="$1"
  local nested="${dest}/results"
  if [[ ! -d "${nested}" ]]; then
    return 0
  fi
  echo "Flattening nested ${nested}/ -> ${dest}/"
  shopt -s dotglob nullglob
  local item base
  for item in "${nested}"/*; do
    [[ -e "${item}" ]] || continue
    base="$(basename "${item}")"
    if [[ -e "${dest}/${base}" ]]; then
      echo "  skip ${base} (already exists in ${dest})" >&2
      continue
    fi
    mv "${item}" "${dest}/${base}"
  done
  shopt -u dotglob nullglob
  rmdir "${nested}" 2>/dev/null || true
}

while [[ $# -gt 0 ]]; do
  case "${1}" in
    -h|--help)
      usage
      ;;
    --run)
      RUN_FOLDER="${2:?--run requires a folder name}"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

mkdir -p "${DEST_DIR}"

do_scp() {
  local ssh_target="$1"
  local scp_port="$2"
  local remote_path="$3"
  if command -v rsync >/dev/null 2>&1; then
    echo "Downloading (rsync) ${ssh_target}:${remote_path} -> ${DEST_DIR}/"
    rsync -az -e "ssh -i ${SSH_KEY} -p ${scp_port} -o StrictHostKeyChecking=accept-new" \
      "${ssh_target}:${remote_path}" "${DEST_DIR}/"
  else
    echo "Downloading (scp) ${ssh_target}:${remote_path} -> ${DEST_DIR}/"
    scp -O -i "${SSH_KEY}" -P "${scp_port}" -r "${ssh_target}:${remote_path}" "${DEST_DIR}/"
  fi
}

if [[ "${1:-}" == "--tcp" ]]; then
  TCP_USER="${2:-root}"
  TCP_HOST="${3:?}"
  TCP_PORT="${4:?}"
  SSH_KEY="${5:-${SSH_KEY}}"
  shift 5 || true
  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --run)
        RUN_FOLDER="${2:?--run requires a folder name}"
        shift 2
        ;;
      *)
        echo "Unknown argument: ${1}" >&2
        usage
        ;;
    esac
  done
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_TARGET="${TCP_USER}@${TCP_HOST}"
  if [[ -n "${RUN_FOLDER}" ]]; then
    REMOTE_PATH="${REMOTE_RESULTS%/}/${RUN_FOLDER}"
  else
    REMOTE_PATH="${REMOTE_RESULTS%/}/"
  fi
  if ! do_scp "${SSH_TARGET}" "${TCP_PORT}" "${REMOTE_PATH}"; then
    echo "Download failed. Check pod is running and TCP port matches the dashboard." >&2
    exit 1
  fi
elif [[ -n "${RUNPOD_TCP_HOST:-}" && -n "${RUNPOD_TCP_PORT:-}" ]]; then
  TCP_USER="${RUNPOD_TCP_USER:-root}"
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_TARGET="${TCP_USER}@${RUNPOD_TCP_HOST}"
  if [[ -n "${RUN_FOLDER}" ]]; then
    REMOTE_PATH="${REMOTE_RESULTS%/}/${RUN_FOLDER}"
  else
    REMOTE_PATH="${REMOTE_RESULTS%/}/"
  fi
  if ! do_scp "${SSH_TARGET}" "${RUNPOD_TCP_PORT}" "${REMOTE_PATH}"; then
    echo "Download failed." >&2
    exit 1
  fi
else
  SSH_TARGET="${1:-}"
  SSH_KEY="${2:-${SSH_KEY}}"
  if [[ -z "${SSH_TARGET}" ]]; then
    usage
  fi
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  echo "Downloading (gateway ${SSH_TARGET}) — RunPod may block SCP here; prefer --tcp (see $0 --help)." >&2
  if [[ -n "${RUN_FOLDER}" ]]; then
    REMOTE_PATH="${REMOTE_RESULTS%/}/${RUN_FOLDER}"
  else
    REMOTE_PATH="${REMOTE_RESULTS%/}/"
  fi
  echo "Downloading ${SSH_TARGET}:${REMOTE_PATH} -> ${DEST_DIR}/"
  if ! scp -O -o RequestTTY=yes -i "${SSH_KEY}" -r "${SSH_TARGET}:${REMOTE_PATH}" "${DEST_DIR}/"; then
    echo "scp failed. Use SSH over exposed TCP from the RunPod UI: $0 --tcp root <ip> <port> ${SSH_KEY}" >&2
    exit 1
  fi
fi

flatten_nested_results "${DEST_DIR}"

n="$(find "${DEST_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Done. ${n} file(s) under ${DEST_DIR}/"
if [[ "${n}" -eq 0 ]]; then
  echo "Warning: no files copied — use --tcp if you used ssh.runpod.io (SCP not supported there)." >&2
fi
