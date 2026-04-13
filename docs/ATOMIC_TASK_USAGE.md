# Guide d'utilisation des tâches atomiques RoboCasa (PandaOmron)

Ce document décrit le workflow actuel pour entraîner et évaluer la tâche atomique
`PnPCounterToCab` (sans navigation) avec PPO.

## 1. Préparer l'environnement

Depuis la racine du dépôt :

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## 2. Scripts principaux (à jour)

- Baseline (récompense native) : `scripts/train_ppo_baseline.py`
- Reward shaping (sans curriculum) : `scripts/train_ppo_reward_shaping.py`
- Curriculum learning (sans reward shaping custom) : `scripts/train_ppo_curriculum.py`
- Évaluation : `scripts/eval_robocasa.py`

Configuration par défaut :

- Robot : `PandaOmron`
- Tâche : `PnPCounterToCab`
- Algo : `PPO` (Stable-Baselines3)

## 3. Lancer l'entraînement

### 3.1 Baseline

```bash
uv run python scripts/train_ppo_baseline.py \
  --headless \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name baseline_seed42
```

### 3.2 Reward shaping

```bash
uv run python scripts/train_ppo_reward_shaping.py \
  --task PnPCounterToCab \
  --headless \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name reward_shaping_seed42
```

### 3.3 Curriculum learning

```bash
uv run python scripts/train_ppo_curriculum.py \
  --task PnPCounterToCab \
  --headless \
  --curriculum_window 100 \
  --curriculum_min_timesteps 50000 \
  --curriculum_thresholds 0.70,0.80 \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name curriculum_seed42
```

## 4. Où sont stockés modèle / metrics / logs

Par défaut, chaque run écrit dans :

```text
models/<run_name>/
```

Structure standard :

```text
models/<run_name>/
├── ppo_final.zip
└── logs/
    ├── metrics.csv
    ├── monitor/
    │   └── env_0/monitor.csv
    ├── tensorboard/
    │   └── events.out.tfevents...
```

Notes :

- Baseline, Reward Shaping et Curriculum écrivent tous `logs/metrics.csv` au même format.
- Avec `--n_envs > 1`, `monitor/` contient `env_0`, `env_1`, etc.
- `monitor.csv` enregistre par épisode : `r` (return), `l` (length), `t` (time).
- `metrics.csv` est écrit à la fin de chaque rollout avec les colonnes :
`timestep, ep_rew_mean, ep_len_mean, success_rate, n_episodes`.

## 5. Paramètres utiles

Paramètres communs :

- `--headless`
- `--seed`
- `--horizon`
- `--n_envs`
- `--total_timesteps`
- `--learning_rate`, `--batch_size`, `--n_steps`
- `--model_dir` (par défaut `models`)
- `--run_name`

Paramètres reward shaping (`train_ppo_reward_shaping.py`) :

- `--task` (actuellement `PnPCounterToCab`)
- `--reach_w`, `--grasp_bonus`, `--place_bonus`, `--success_bonus`
- `--device` / `--gpu`

Paramètres curriculum (`train_ppo_curriculum.py`) :

- `--task` (actuellement `PnPCounterToCab`)
- `--device` / `--gpu`
- `--curriculum_window`, `--curriculum_min_timesteps`, `--curriculum_thresholds`

## 6. Évaluer un modèle

```bash
uv run python scripts/eval_robocasa.py \
  --task PnPCounterToCab \
  --model_path models/<run_name>/ppo_final.zip \
  --episodes 10 \
  --save_video
```

Si `--save_video` est activé, les vidéos sont enregistrées dans `eval_videos/`.

## 7. FAQ rapide

### Q1. Pourquoi `total_timesteps=1000` finit à 2048 ?

C'est normal avec PPO : l'apprentissage avance par blocs de rollout de taille `n_steps`
(par défaut `2048`).

### Q2. `metrics.csv` : rollout et episode, c'est la même chose ?

Non.

- Un **episode** = une trajectoire complète, de `reset()` à `done/truncated`.
- Un **rollout** = un bloc de collecte PPO (taille `n_steps`, par défaut 2048 en single-env).
- `metrics.csv` écrit **une ligne par rollout**, mais `ep_rew_mean`, `ep_len_mean`,
`success_rate`, `n_episodes` sont calculés sur les episodes terminés du buffer SB3.

### Q3. Warnings robosuite (macro / mimicgen / mink) ?

Souvent non bloquants. Si l'entraînement démarre et produit les logs PPO, vous pouvez continuer.