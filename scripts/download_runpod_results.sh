#!/usr/bin/env bash
# Download pilot/results from a RunPod pod to the local repo.
#
# RunPod documents two SSH modes:
#   • ssh.runpod.io — interactive shell only; NO SCP/SFTP support (downloads will fail).
#   • "SSH over exposed TCP" (root@IP -p PORT) — supports SCP/SFTP — use this for file copy.
#
# Usage (recommended — copy host/port from Pod → Connect → "SSH over exposed TCP"):
#   ./scripts/download_runpod_results.sh --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519
#
# Or env (same values as dashboard):
#   RUNPOD_TCP_HOST=213.192.2.74 RUNPOD_TCP_PORT=40171 ./scripts/download_runpod_results.sh
#
# Optional: DEST_DIR, REMOTE_RESULTS, RUNPOD_TCP_USER (default root), SSH_KEY

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${DEST_DIR:-${REPO_ROOT}/data/results/runpod_pilot}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/workspace/metacog-llm-compute/data/results}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"

usage() {
  echo "Usage (recommended — TCP SSH from RunPod dashboard, supports scp):" >&2
  echo "  $0 --tcp <user> <host> <port> [ssh_key]" >&2
  echo "  Example: $0 --tcp root 213.192.2.74 40171 ~/.ssh/id_ed25519" >&2
  echo "" >&2
  echo "Or: RUNPOD_TCP_HOST=... RUNPOD_TCP_PORT=... [RUNPOD_TCP_USER=root] $0" >&2
  echo "" >&2
  echo "Legacy (often broken): gateway user@ssh.runpod.io — RunPod does NOT support SCP on that URL." >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

mkdir -p "${DEST_DIR}"

if [[ "${1:-}" == "--tcp" ]]; then
  TCP_USER="${2:-root}"
  TCP_HOST="${3:?}"
  TCP_PORT="${4:?}"
  SSH_KEY="${5:-${SSH_KEY}}"
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_TARGET="${TCP_USER}@${TCP_HOST}"
  echo "Downloading (TCP SSH) ${SSH_TARGET}:${REMOTE_RESULTS}/ -> ${DEST_DIR}/"
  # Real sshd: legacy -O avoids SFTP; -P is port for scp
  if ! scp -O -i "${SSH_KEY}" -P "${TCP_PORT}" -r "${SSH_TARGET}:${REMOTE_RESULTS}/" "${DEST_DIR}/"; then
    echo "scp failed. Check pod is running and TCP port matches the dashboard." >&2
    exit 1
  fi
elif [[ -n "${RUNPOD_TCP_HOST:-}" && -n "${RUNPOD_TCP_PORT:-}" ]]; then
  TCP_USER="${RUNPOD_TCP_USER:-root}"
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_TARGET="${TCP_USER}@${RUNPOD_TCP_HOST}"
  echo "Downloading (TCP SSH) ${SSH_TARGET}:${REMOTE_RESULTS}/ -> ${DEST_DIR}/"
  if ! scp -O -i "${SSH_KEY}" -P "${RUNPOD_TCP_PORT}" -r "${SSH_TARGET}:${REMOTE_RESULTS}/" "${DEST_DIR}/"; then
    echo "scp failed." >&2
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
  echo "Downloading ${SSH_TARGET}:${REMOTE_RESULTS}/ -> ${DEST_DIR}/"
  if ! scp -O -o RequestTTY=yes -i "${SSH_KEY}" -r "${SSH_TARGET}:${REMOTE_RESULTS}/" "${DEST_DIR}/"; then
    echo "scp failed. Use SSH over exposed TCP from the RunPod UI: $0 --tcp root <ip> <port> ${SSH_KEY}" >&2
    exit 1
  fi
fi

n="$(find "${DEST_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Done. ${n} file(s) under ${DEST_DIR}/"
if [[ "${n}" -eq 0 ]]; then
  echo "Warning: no files copied — use --tcp if you used ssh.runpod.io (SCP not supported there)." >&2
fi
