#!/usr/bin/env bash
#SBATCH --job-name=robocasa_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/%x_%j.out

set -euo pipefail

cd "$(dirname "$0")"

# Optional: avoid oversubscribing CPU threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN_NAME="${RUN_NAME:-slurm_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
N_ENVS="${N_ENVS:-4}"
DEVICE="${DEVICE:-auto}"

echo "Run name: ${RUN_NAME}"
echo "Timesteps: ${TOTAL_TIMESTEPS} | n_envs: ${N_ENVS} | device: ${DEVICE}"

# Dense + strict + curriculum (recommended)
uv run python scripts/train_ppo_curriculum_dense.py \
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

