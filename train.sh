#!/usr/bin/env bash
#SBATCH --job-name=robocasa_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=9
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

# If modules (or site profiles) set PYTHONHOME, it can break venv imports.
unset PYTHONHOME 2>/dev/null || true

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

# Ensure local robocasa/robosuite sources are importable regardless of how Python is launched.
export PYTHONPATH="${PWD}:${PWD}/robocasa:${PWD}/robosuite${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Serialize initial env creation/sync to avoid concurrent partial installs.
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
  # Ensure deps from uv.lock are present, but do not re-resolve every run.
  # --inexact: keep packages added later via `uv pip install` (extras below).
  # Without it, every run prunes the extras, then we re-install them — wasted time.
  echo "Syncing environment from uv.lock (this can download once)..."
  uv sync --frozen --no-dev --inexact

  # robosuite (and robocasa) require a few runtime deps that are not currently declared
  # in the top-level pyproject. Install them here (idempotent; cached by uv).
  # Pass --python explicitly: without it, `uv pip install` may auto-pick `./.venv`
  # in cwd instead of UV_PROJECT_ENVIRONMENT, causing extras to land in the wrong venv.
  # Full runtime dep set — kept in sync with the working local .venv.
  # robocasa/robosuite are vendored as submodules and don't declare their
  # runtime deps, so we must list them explicitly here.
  #
  # Hard pins required by robocasa/__init__.py asserts:
  #   mujoco == 3.3.1
  #   numpy  == 2.2.5
  # DO NOT add mink or robosuite_models — both pull numpy<2 and break the
  # robocasa numpy assert. Their import-time warnings are harmless for tasks
  # that don't use the GR1 / mink controller (e.g. PnPCounterToCab).
  #
  # SB3 logging stack: tensorboard (TB writer) + rich + tqdm (ProgressBarCallback).
  # Video stack:        imageio + imageio-ffmpeg + av (used by eval rollouts).
  EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-termcolor mujoco==3.3.1 numpy==2.2.5 scipy opencv-python pyyaml pillow pygame qpsolvers[quadprog] pynput tqdm rich h5py lxml tensorboard imageio imageio-ffmpeg av matplotlib psutil}"
  read -r -a _extra_pkgs <<< "${EXTRA_PIP_PACKAGES}"
  echo "Ensuring extra runtime deps: ${EXTRA_PIP_PACKAGES}"
  uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "${_extra_pkgs[@]}"
  unset _extra_pkgs EXTRA_PIP_PACKAGES

  echo "Sanity checks (venv python):"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sys, importlib.metadata as m; print('  exe:', sys.executable); import termcolor; print('  termcolor:', m.version('termcolor'))"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sys; print('  sys.prefix:', sys.prefix); print('  sys.base_prefix:', sys.base_prefix)"
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import mujoco, numpy; print('  mujoco:', mujoco.__version__); print('  numpy:', numpy.__version__); assert mujoco.__version__ == '3.3.1', mujoco.__version__; assert numpy.__version__ == '2.2.5', numpy.__version__"

  # Robosuite expects macros_private.py (sibling of macros.py). The upstream
  # setup script is interactive (input() prompt) so we inline the same logic.
  ROBOSUITE_PKG="${PWD}/robosuite/robosuite"
  if [[ -f "${ROBOSUITE_PKG}/macros.py" && ! -f "${ROBOSUITE_PKG}/macros_private.py" ]]; then
    cp "${ROBOSUITE_PKG}/macros.py" "${ROBOSUITE_PKG}/macros_private.py"
    echo "Created ${ROBOSUITE_PKG}/macros_private.py"
  fi

  echo "Environment ready."
) 200>"${LOCK_FILE}"

# Dense + strict + curriculum (recommended)
echo "Sanity checks (uv run python):"
uv run --no-sync python -c "import sys, importlib.metadata as m; print('  exe:', sys.executable); import termcolor; print('  termcolor:', m.version('termcolor'))"
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

