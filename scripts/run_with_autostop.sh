#!/usr/bin/env bash
# scripts/run_with_autostop.sh <python-script> [args...]
set -uo pipefail
STOP_POD="${STOP_POD:-1}"
POD_ID="${RUNPOD_POD_ID:-}"
_stopped=0
_stop_pod() {
  local ec=$1
  [[ "$_stopped" == "1" ]] && return; _stopped=1
  if [[ "$STOP_POD" != "1" ]]; then echo "[autostop] disabled, exit=$ec"; return; fi
  if [[ -z "$POD_ID" ]]; then echo "[autostop] not on a pod, skip (exit=$ec)"; return; fi
  if ! command -v runpodctl >/dev/null 2>&1; then echo "[autostop] runpodctl missing (exit=$ec)"; return; fi
  echo "[autostop] stopping pod $POD_ID (run exit=$ec)"
  runpodctl stop pod "$POD_ID" || echo "[autostop] WARN: stop failed for $POD_ID"
}
trap '_stop_pod 130' INT TERM
python "$@"; RUN_EC=$?
_stop_pod "$RUN_EC"
exit "$RUN_EC"
