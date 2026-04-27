#!/usr/bin/env bash
# ============================================================================
# cluster_sync_pull.sh — Pull experiment results from cluster to local
# ============================================================================
# Pulls runs/ and/or artifacts/ from the remote cluster. Merges with local
# dirs (rsync -a). Creates local runs/ and artifacts/ if missing.
#
# Usage:
#   ./scripts/cluster_sync_pull.sh                   # pull runs/ only (default)
#   ./scripts/cluster_sync_pull.sh --only-runs        # explicit: runs/ only
#   ./scripts/cluster_sync_pull.sh --only-artifacts   # artifacts/ only
#   ./scripts/cluster_sync_pull.sh --all              # both runs/ and artifacts/
#   ./scripts/cluster_sync_pull.sh --dry-run          # preview without copying
#
# Requirements: rsync, SSH access to cluster
# SSH config expected:
#   Host gpu
#       Hostname gpu-gw.enst.fr
#       User latoundji-25
#       IdentityFile ~/.ssh/id_rsa
#
# Remote project path (override with REMOTE_PATH):
#   /home/infres/latoundji-25/robocasa/RoboCasa-RL
# ============================================================================

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gpu}"
REMOTE_PATH="${REMOTE_PATH:-/home/infres/latoundji-25/robocasa/RoboCasa-RL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN=()
MODE="runs" # runs | artifacts | both
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-runs) MODE="runs" ;;
    --only-artifacts) MODE="artifacts" ;;
    --all) MODE="both" ;;
    --dry-run) DRY_RUN=(--dry-run) ;;
    --quiet) QUIET=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

RSYNC_BASE=(rsync -a "${DRY_RUN[@]}" --compress --partial --human-readable)
if [[ "${QUIET}" -eq 0 ]]; then
  RSYNC_BASE+=(--itemize-changes --info=NAME2,STATS2)
fi

remote_has_dir() {
  local sub="$1"
  ssh "${REMOTE_HOST}" "test -d \"${REMOTE_PATH}/${sub}\""
}

pull_one() {
  local remote_sub="$1"
  local local_dest="${REPO_ROOT}/${remote_sub}"
  mkdir -p "${local_dest}"
  "${RSYNC_BASE[@]}" "${REMOTE_HOST}:${REMOTE_PATH}/${remote_sub}/" "${local_dest}/"
}

case "${MODE}" in
  runs)
    pull_one runs
    ;;
  artifacts)
    mkdir -p "${REPO_ROOT}/artifacts"
    if remote_has_dir artifacts; then
      pull_one artifacts
    else
      echo "Remote has no artifacts/ directory yet; skipping."
    fi
    ;;
  both)
    pull_one runs
    mkdir -p "${REPO_ROOT}/artifacts"
    if remote_has_dir artifacts; then
      pull_one artifacts
    else
      echo "Remote has no artifacts/ directory yet; skipping."
    fi
    ;;
esac

echo "Done. Local repo: ${REPO_ROOT}"
