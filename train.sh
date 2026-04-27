#!/usr/bin/env bash
#SBATCH --job-name=robocasa_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out

set -euo pipefail

# SLURM copies this script under /var/spool/slurmd/... — do not use dirname "$0" as project root.
# Use the directory from which sbatch was run (recommended: cd repo && sbatch train.sh).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
elif [[ -n "${PROJECT_ROOT:-}" ]]; then
  cd "${PROJECT_ROOT}"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Optional: avoid oversubscribing CPU threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN_NAME="${RUN_NAME:-slurm_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
N_ENVS="${N_ENVS:-4}"
DEVICE="${DEVICE:-auto}"

echo "Run name: ${RUN_NAME}"
echo "Timesteps: ${TOTAL_TIMESTEPS} | n_envs: ${N_ENVS} | device: ${DEVICE}"

# ---- Python / venv / uv bootstrap (avoid redownloading every job) ------------
# If your cluster uses environment-modules, you can force a specific Python:
#   sbatch --export=PYTHON_MODULE=python/3.11.13 train.sh
if command -v module >/dev/null 2>&1; then
  if [[ -n "${PYTHON_MODULE:-}" ]]; then
    module load "${PYTHON_MODULE}" 2>/dev/null || true
  fi
fi

# Make uv use a persistent cache and a persistent venv (outside the repo).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/robocasa_rl}"
if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
  VENV_DIR="${VENV_DIR}_${SLURM_JOB_PARTITION}"
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$VENV_DIR}"
mkdir -p "${UV_CACHE_DIR}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

echo "uv: $(command -v uv 2>/dev/null || echo '<missing>')"
echo "python3: $(command -v python3 2>/dev/null || echo '<missing>')"
echo "UV_CACHE_DIR: ${UV_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT: ${UV_PROJECT_ENVIRONMENT}"

# Serialize initial env creation/sync to avoid concurrent partial installs.
LOCK_FILE="${PWD}/.uv_sync.lock"
(
  flock -x 200
  if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    py_bin="$(command -v python3 || true)"
    if [[ -z "${py_bin}" ]]; then
      echo "ERROR: python3 not found on PATH. Load a python module or install python3." >&2
      exit 1
    fi
    uv venv --python "${py_bin}" "${UV_PROJECT_ENVIRONMENT}"
  fi
  # Ensure deps from uv.lock are present, but do not re-resolve every run.
  echo "Syncing environment from uv.lock (this can download once)..."
  uv sync --frozen --no-dev
  echo "Environment ready."
) 200>"${LOCK_FILE}"

# Dense + strict + curriculum (recommended)
uv run --no-sync python scripts/train_ppo_curriculum_dense.py \
  --task PnPCounterToCab \
  --headless \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --n_envs "${N_ENVS}" \
  --run_name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --log_debug \
  --checkpoint_freq 200000 \
  --best_eval_freq 200000 \
  --best_eval_episodes 5

