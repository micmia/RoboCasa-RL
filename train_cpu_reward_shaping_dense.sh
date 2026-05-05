#!/usr/bin/env bash
# Entraînement PPO — reward shaping dense (scripts/train_ppo_reward_shaping_dense.py), **CPU uniquement**.
# Usage cluster : depuis la racine du dépôt → sbatch train_cpu_reward_shaping_dense.sh
# Usage local  : bash train_cpu_reward_shaping_dense.sh
#
# Adapter la partition Slurm (#SBATCH --partition) à ton site si besoin.

#SBATCH --job-name=robocasa_dense_cpu
#SBATCH --partition=CPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out

# RUN_NAME=dense_cpu_810314_20260428_122953  RESUME_FROM=models/dense_cpu_810314_20260428_122953/checkpoints/ppo_3200000_steps.zip sbatch train_cpu_reward_shaping_dense.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
elif [[ -n "${PROJECT_ROOT:-}" ]]; then
  cd "${PROJECT_ROOT}"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Forcer l’absence de GPU pour PyTorch / drivers (entraînement SB3 sur CPU).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

RUN_NAME="${RUN_NAME:-dense_cpu_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
N_ENVS="${N_ENVS:-4}"
# Toujours CPU pour ce script (ne pas passer DEVICE=cuda ici).
DEVICE="cpu"
RESUME_FROM="${RESUME_FROM:-models/${RUN_NAME}/ppo_200000_steps.zip}"

RESUME_FROM_FLAG=""
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_FROM_FLAG="--resume_from ${RESUME_FROM}"
fi

echo "Run name: ${RUN_NAME}"
echo "Timesteps: ${TOTAL_TIMESTEPS} | n_envs: ${N_ENVS} | device: ${DEVICE} (forced)"

if command -v module >/dev/null 2>&1; then
  if [[ -n "${PYTHON_MODULE:-}" ]]; then
    module load "${PYTHON_MODULE}" 2>/dev/null || true
  fi
fi

unset PYTHONHOME 2>/dev/null || true

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

export PYTHONPATH="${PWD}:${PWD}/robocasa:${PWD}/robosuite${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

LOCK_FILE="${PWD}/.uv_sync.lock"
(
  flock -x 200
  if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    py_bin=""
    if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      py_bin="$(command -v "${PYTHON_BIN}")"
    elif command -v python3.11 >/dev/null 2>&1; then
      py_bin="$(command -v python3.11)"
    else
      py_bin="$(command -v python3 || true)"
    fi
    if [[ -z "${py_bin}" ]]; then
      echo "ERROR: python3 not found on PATH. Load a python module or install python3." >&2
      exit 1
    fi
    uv venv --python "${py_bin}" "${UV_PROJECT_ENVIRONMENT}"
  fi
  echo "Syncing environment from uv.lock (this can download once)..."
  uv sync --frozen --no-dev --inexact

  EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-termcolor mujoco==3.3.1 numpy==2.2.5 scipy opencv-python pyyaml pillow pygame qpsolvers[quadprog] pynput tqdm rich h5py lxml tensorboard imageio imageio-ffmpeg av matplotlib psutil}"
  read -r -a _extra_pkgs <<< "${EXTRA_PIP_PACKAGES}"
  echo "Ensuring extra runtime deps: ${EXTRA_PIP_PACKAGES}"
  uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "${_extra_pkgs[@]}"
  unset _extra_pkgs EXTRA_PIP_PACKAGES

  echo "Sanity checks (venv python):"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sys, importlib.metadata as m; print('  exe:', sys.executable); import termcolor; print('  termcolor:', m.version('termcolor'))"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sys; print('  sys.prefix:', sys.prefix); print('  sys.base_prefix:', sys.base_prefix)"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import mujoco, numpy; print('  mujoco:', mujoco.__version__); print('  numpy:', numpy.__version__); assert mujoco.__version__ == '3.3.1', mujoco.__version__; assert numpy.__version__ == '2.2.5', numpy.__version__"

  ROBOSUITE_PKG="${PWD}/robosuite/robosuite"
  if [[ -f "${ROBOSUITE_PKG}/macros.py" && ! -f "${ROBOSUITE_PKG}/macros_private.py" ]]; then
    cp "${ROBOSUITE_PKG}/macros.py" "${ROBOSUITE_PKG}/macros_private.py"
    echo "Created ${ROBOSUITE_PKG}/macros_private.py"
  fi

  echo "Environment ready."
) 200>"${LOCK_FILE}"

echo "Sanity checks (uv run python):"
uv run --no-sync python -c "import sys, importlib.metadata as m; print('  exe:', sys.executable); import termcolor; print('  termcolor:', m.version('termcolor'))"

# Pas de --gpu : train_ppo_reward_shaping_dense.py utilise alors args.device (= cpu).
uv run --no-sync python scripts/train_ppo_reward_shaping_dense.py \
  --task "${TASK:-PnPCounterToCab}" \
  --headless \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --n_envs "${N_ENVS}" \
  --run_name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --log_debug \
  --checkpoint_freq "${CHECKPOINT_FREQ:-200000}" \
  --best_eval_freq "${BEST_EVAL_FREQ:-200000}" \
  --best_eval_episodes "${BEST_EVAL_EPISODES:-5}" \
  ${RESUME_FROM_FLAG}
