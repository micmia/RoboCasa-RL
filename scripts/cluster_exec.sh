#!/usr/bin/env bash
# ============================================================================
# cluster_exec.sh — Execute SLURM / shell commands on the remote cluster
# ============================================================================
# Convenience wrapper to submit jobs, check status, and read logs from local.
#
# Usage:
#   ./scripts/cluster_exec.sh submit train.sh                     # sbatch from repo root
#   ./scripts/cluster_exec.sh status                              # squeue --me
#   ./scripts/cluster_exec.sh status 772878                       # sacct for job
#   ./scripts/cluster_exec.sh logs 772878                         # tail stdout (.out)
#   ./scripts/cluster_exec.sh logs 772878 err                     # tail stderr (.err if any)
#   ./scripts/cluster_exec.sh cancel 772878                       # scancel
#   ./scripts/cluster_exec.sh shell                               # interactive SSH
#   ./scripts/cluster_exec.sh cmd "ls -la runs/"                  # arbitrary command in repo
#
# Environment:
#   REMOTE_HOST   default: gpu
#   REMOTE_PATH   default: /home/infres/latoundji-25/robocasa/RoboCasa-RL
# ============================================================================

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gpu}"
REMOTE_PATH="${REMOTE_PATH:-/home/infres/latoundji-25/robocasa/RoboCasa-RL}"

usage() {
  sed -n '2,18p' "$0"
}

remote_cd() {
  printf 'cd %q && ' "${REMOTE_PATH}"
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  submit)
    script="${1:?usage: submit <script.sh e.g. train.sh>}"
    shift || true
    # Remaining args are passed to sbatch (e.g. --export=KEY=val).
    # NOTE: avoid `printf ' %q' "$@"` because with empty "$@" it prints `''`
    # which makes sbatch try to open an empty filename.
    ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_PATH}" "${script}" "$@" <<'EOS'
set -euo pipefail
REMOTE_PATH="$1"; shift
SCRIPT="$1"; shift

cd "${REMOTE_PATH}"

mkdir -p runs/slurm

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: script not found: ${REMOTE_PATH}/${SCRIPT}" >&2
  echo "Hint: run ./scripts/cluster_sync_push.sh then retry." >&2
  ls -la >&2 || true
  exit 1
fi

sbatch "$@" "${SCRIPT}"
EOS
    ;;
  status)
    job="${1:-}"
    if [[ -z "${job}" ]]; then
      ssh "${REMOTE_HOST}" "squeue --me -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"
    else
      ssh "${REMOTE_HOST}" "sacct -j ${job} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS -P"
    fi
    ;;
  logs)
    job="${1:?usage: logs <jobid> [err]'}"
    kind="${2:-out}"
    # Matches runs/slurm/%x_%j.out → e.g. robocasa_train_807901.out
    ext="out"
    [[ "${kind}" == "err" ]] && ext="err"
    ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_PATH}" "${job}" "${ext}" <<'EOS'
set -euo pipefail
REMOTE_PATH="$1"
JOB="$2"
LOG_EXT="$3"
cd "${REMOTE_PATH}"
shopt -s nullglob
matches=(runs/slurm/*_"${JOB}"."${LOG_EXT}")
if [[ ${#matches[@]} -eq 0 ]]; then
  echo "No runs/slurm/*_${JOB}.${LOG_EXT} found under ${REMOTE_PATH}" >&2
  echo "Existing logs (latest 20):" >&2
  ls -1t runs/slurm 2>/dev/null | head -n 20 >&2 || true
  exit 1
fi
f="${matches[0]}"
bytes="$(stat -c '%s' "$f" 2>/dev/null || echo '?')"
echo "==> ${REMOTE_PATH}/${f} (${bytes} bytes)"
cat "$f"
EOS
    ;;
  cancel)
    job="${1:?usage: cancel <jobid>'}"
    ssh "${REMOTE_HOST}" "scancel ${job}"
    ;;
  shell)
    ssh -t "${REMOTE_HOST}" "$(remote_cd) bash -l"
    ;;
  cmd)
    inner="${*:-}"
    [[ -n "${inner}" ]] || { echo "usage: cmd \"command\"" >&2; exit 2; }
    ssh "${REMOTE_HOST}" "$(remote_cd) ${inner}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
