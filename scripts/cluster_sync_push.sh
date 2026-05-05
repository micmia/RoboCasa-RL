#!/usr/bin/env bash
# ============================================================================
# cluster_sync_push.sh — Push repo sources from local machine to cluster
# ============================================================================
# Syncs this repository to the remote cluster via rsync over SSH.
# Skips .venv, caches, __pycache__, runs/ and artifacts/ by default.
#
# Usage:
#   ./scripts/cluster_sync_push.sh                  # default: sync sources only
#   ./scripts/cluster_sync_push.sh --with-runs      # also sync runs/
#   ./scripts/cluster_sync_push.sh --with-artifacts # also sync artifacts/
#   ./scripts/cluster_sync_push.sh --dry-run        # preview without copying
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

WITH_RUNS=0
WITH_ARTIFACTS=0
DRY_RUN=()
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-runs) WITH_RUNS=1 ;;
    --with-artifacts) WITH_ARTIFACTS=1 ;;
    --dry-run) DRY_RUN=(--dry-run) ;;
    --quiet) QUIET=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

EXCLUDES=(
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.mypy_cache/'
  --exclude='.pytest_cache/'
  --exclude='.ruff_cache/'
  --exclude='*.egg-info/'
  --exclude='.git/'
)

if [[ "${WITH_RUNS}" -eq 0 ]]; then
  EXCLUDES+=(--exclude='runs/')
fi
if [[ "${WITH_ARTIFACTS}" -eq 0 ]]; then
  EXCLUDES+=(--exclude='artifacts/')
fi

echo "Pushing ${REPO_ROOT}/ -> ${REMOTE_HOST}:${REMOTE_PATH}/"

# Ensure remote path + SLURM log dir exists even when runs/ isn't synced.
ssh "${REMOTE_HOST}" "mkdir -p \"${REMOTE_PATH}\" \"${REMOTE_PATH}/runs/slurm\""

RSYNC_OPTS=(-a "${DRY_RUN[@]}" --compress --partial --human-readable)
if [[ "${QUIET}" -eq 0 ]]; then
  # Show exactly what changed and basic transfer stats.
  RSYNC_OPTS+=(--itemize-changes --info=NAME2,STATS2)
fi

rsync "${RSYNC_OPTS[@]}" \
  "${EXCLUDES[@]}" \
  "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_PATH}/"

echo "Done."
