#!/usr/bin/env bash
#SBATCH --job-name=robocasa_eval
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/%x_%j.out

set -euo pipefail

cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-}"
EPISODES="${EPISODES:-10}"
TARGET="${TARGET:-cab}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Example:"
  echo "  sbatch eval.sh --export=MODEL_PATH=models/<run>/ppo_final.zip"
  exit 2
fi

ARGS=(--task PnPCounterToCab --model_path "${MODEL_PATH}" --episodes "${EPISODES}" --target "${TARGET}")
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  ARGS+=(--save_video)
fi

uv run python scripts/eval_robocasa_dense.py "${ARGS[@]}"

