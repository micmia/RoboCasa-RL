# Guide d'utilisation des tâches atomiques RoboCasa (PandaOmron)

Ce document décrit le workflow pour entraîner et évaluer la tâche atomique **`PnPCounterToCab`** (pick-and-place comptoir → placard, sans navigation) avec PPO (Stable-Baselines3).

Pour le détail du flux (wrappers, récompense, curriculum), voir [`docs/ALGORITHM_FLOW.md`](ALGORITHM_FLOW.md).

**Configuration `MyPnPCounterToCab` dans ce dépôt** : l’objet à manipuler est fixé à la pomme **`apple_1`** sur le plan de travail ; un distractor `bowl_1` peut être présent. La méthode `reward()` de cette classe renvoie **0** ; le signal utile pour PPO vient soit du **reward shaping** (`train_ppo_reward_shaping.py`), soit de la dynamique / exploration seule (baseline et curriculum).

---

## 1. Préparer l'environnement

Depuis la racine du dépôt :

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Python **≥ 3.11** (voir `pyproject.toml`).

---

## 2. Scripts principaux

| Script | Rôle |
| ------ | ---- |
| `scripts/train_ppo_baseline.py` | PPO sans `AtomicRewardShapingWrapper` |
| `scripts/train_ppo_reward_shaping.py` | PPO + récompense dense custom (reach / grasp / place / success) |
| `scripts/train_ppo_curriculum.py` | PPO + `CurriculumWrapper` + `SuccessInfoWrapper` (pas de shaping custom) |
| `scripts/eval_robocasa.py` | Chargement de `ppo_final.zip`, épisodes, vidéo multi-caméras optionnelle |
| `scripts/visualize_custom_env.py` | Aperçu interactif / caméras de `MyPnPCounterToCab` (hors PPO) |

Configuration par défaut : robot **`PandaOmron`**, contrôle **`control_freq=20`**, horizon d’épisode **`500`** (sauf modification `--horizon`).

---

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

`--task` doit rester **`PnPCounterToCab`** (seule valeur acceptée par ce script).

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

---

## 4. Où sont stockés modèle / metrics / logs

Par défaut, chaque run écrit sous **`models/<run_name>/`** (si `--run_name` est vide, un nom horodaté est généré).

```text
models/<run_name>/
├── ppo_final.zip
└── logs/
    ├── metrics.csv
    ├── monitor/
    │   └── env_0/monitor.csv
    └── tensorboard/
        └── events.out.tfevents...
```

**`--log_root`** : uniquement dans `train_ppo_reward_shaping.py` et `train_ppo_curriculum.py`. Si défini, remplace le répertoire `logs/` (les chemins TensorBoard / monitor / `metrics.csv` suivent ce préfixe). Le baseline utilise toujours `models/<run_name>/logs/`.

Autres précisions :

- Avec **`--n_envs > 1`**, `monitor/` contient `env_0`, `env_1`, …
- `monitor.csv` : une ligne par épisode avec notamment `r` (return), `l` (longueur), `t` (temps).
- `metrics.csv` : une ligne **à la fin de chaque rollout** PPO, colonnes  
  `timestep, ep_rew_mean, ep_len_mean, success_rate, n_episodes`.  
  La colonne `success_rate` lit `is_success` dans les stats d’épisode agrégées par SB3 ; selon les wrappers, elle peut rester à **0** même si la tâche progresse — le taux de succès fiable reste l’**évaluation** (`eval_robocasa.py`) ou un suivi TensorBoard selon votre configuration.

---

## 5. Paramètres utiles

**Communs** (baseline, shaping, curriculum) :

- `--headless`, `--seed`, `--horizon`, `--n_envs`, `--total_timesteps`
- `--learning_rate`, `--batch_size`, `--n_steps`
- `--model_dir` (défaut `models`), `--run_name`

**`train_ppo_reward_shaping.py` uniquement** :

- `--task` (`PnPCounterToCab` uniquement)
- `--reach_reward`, `--grasp_reward`, `--place_reward`, `--success_reward` (défauts : `0.25`, `0.5`, `1.0`, `5.0`)
- `--device`, `--gpu`, `--log_root`

**`train_ppo_curriculum.py` uniquement** :

- `--task` (`PnPCounterToCab`)
- `--device`, `--gpu`, `--log_root`
- `--curriculum_window`, `--curriculum_min_timesteps`, `--curriculum_thresholds` (liste séparée par des virgules)

Le baseline n’expose pas **`--device` / `--gpu`** (appareil PyTorch laissé au défaut SB3, en pratique souvent CPU si non précisé ailleurs).

---

## 6. Évaluer un modèle

```bash
uv run python scripts/eval_robocasa.py \
  --task PnPCounterToCab \
  --model_path models/<run_name>/ppo_final.zip \
  --episodes 10 \
  --save_video
```

- **`--video_path`** (défaut `eval_videos`) : dossier racine des vidéos.
- Avec `--save_video`, les fichiers vont sous **`eval_videos/<run_name>/`**, où `<run_name>` est le **nom du dossier parent** de `ppo_final.zip` (ex. `models/reward_shaping_seed42/ppo_final.zip` → `eval_videos/reward_shaping_seed42/<run_name>_ep_0.mp4`, …).

Le retour cumulé à l’évaluation ne correspond pas nécessairement aux retours vus à l’entraînement avec reward shaping ; le **booléen de succès** affiché repose sur `_check_success()` de l’environnement brut.

---

## 7. FAQ rapide

### Q1. Pourquoi `total_timesteps=1000` semble « sauter » à 2048 ?

C’est lié à PPO : la collecte se fait par rollouts de **`n_steps`** pas par environnement (défaut **2048**). Le premier `learn` peut donc dépasser légèrement la cible si elle est inférieure à `n_steps`.

### Q2. Rollout et épisode, c’est la même chose ?

Non.

- **Épisode** : de `reset()` jusqu’à `terminated` ou `truncated`.
- **Rollout** : bloc de collecte PPO de taille `n_steps` (× nombre d’envs).

`metrics.csv` ajoute **une ligne par rollout** ; les moyennes portent sur les épisodes terminés présents dans le buffer SB3 sur cette fenêtre.

### Q3. Warnings robosuite / dépendances ?

Souvent non bloquants. Si l’entraînement démarre et que les logs PPO défilent, vous pouvez en général continuer.

### Q4. Visualiser l’environnement sans entraîner ?

```bash
uv run python scripts/visualize_custom_env.py
```

(nécessite un affichage graphique ; voir le script pour les caméras et le rendu.)
