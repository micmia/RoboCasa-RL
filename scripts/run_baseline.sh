bash#!/bin/bash
SEEDS=(0 1 2 3 4)
TIMESTEPS=200000

for SEED in "${SEEDS[@]}"; do
    echo "=============================="
    echo ">>> BASELINE seed=$SEED"
    echo "=============================="
    python train_ppo_baseline.py \
        --headless \
        --seed $SEED \
        --total_timesteps $TIMESTEPS \
        --log_root "logs/baseline_seed${SEED}" \
        --model_dir "models/baseline_seed${SEED}" \
        --run_name "baseline_seed${SEED}"
done

echo "Tous les runs terminés."