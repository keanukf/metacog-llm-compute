#!/usr/bin/env bash
# Restore Cursor plan files from git history into .cursor/plans/ (local only; .cursor/ is gitignored).
# Idempotent: safe to re-run after a merge that removed tracked .cursor/ files from the working tree.
#
# Usage (from repo root): bash scripts/restore_cursor_plans.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RESTORE_REF="${RESTORE_REF:-57baf33^}"
PLANS_DIR=".cursor/plans"

if ! git rev-parse --verify "${RESTORE_REF}" >/dev/null 2>&1; then
  echo "ERROR: git ref '${RESTORE_REF}' not found. Set RESTORE_REF to the parent of the untrack commit." >&2
  exit 1
fi

mkdir -p "${PLANS_DIR}"

count=0
while IFS= read -r git_path; do
  [[ -z "${git_path}" ]] && continue
  base="$(basename "${git_path}")"
  out="${PLANS_DIR}/${base}"
  echo "Restoring ${out} from ${RESTORE_REF}"
  git show "${RESTORE_REF}:${git_path}" >"${out}"
  count=$((count + 1))
done < <(git ls-tree --name-only -r "${RESTORE_REF}" "${PLANS_DIR}/" 2>/dev/null || true)

if [[ "${count}" -eq 0 ]]; then
  echo "No plan files found at ${RESTORE_REF}:${PLANS_DIR}/" >&2
  exit 1
fi

echo "Done. Restored ${count} file(s) under ${PLANS_DIR}/ (gitignored)."
