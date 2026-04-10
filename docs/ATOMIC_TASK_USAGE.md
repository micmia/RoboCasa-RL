# Guide d'utilisation des tâches atomiques RoboCasa (PandaOmron)

Ce document explique comment entraîner et évaluer une **tâche atomique** RoboCasa dans ce dépôt, **sans navigation**.  
La tâche par défaut est : `PnPCounterToCab` (prendre un objet sur le plan de travail et le placer dans un placard).

## 1. Préparer l'environnement

Depuis la racine du dépôt :

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Si `robosuite` / `robocasa` ne sont pas encore installés, suivez d'abord les étapes dans `README.md`.

## 2. Scripts principaux

- Entraînement : `scripts/train_robocasa.py`
- Évaluation : `scripts/eval_robocasa.py`
- Environnement personnalisé : `env/custom_pnp_counter_to_cab.py`

Configuration actuelle :

- Robot : `PandaOmron`
- Tâche : `PnPCounterToCab`
- Algorithme : `PPO` (Stable-Baselines3)
- Récompense : option `custom_reward_shaping` (récompense dense)

## 3. Lancer l'entraînement

Commande recommandée (sans rendu, plus rapide) :

```bash
uv run python scripts/train_robocasa.py \
  --task PnPCounterToCab \
  --headless \
  --custom_reward_shaping \
  --total_timesteps 300000 \
  --n_envs 1
```

Le modèle final est sauvegardé par défaut dans :

```text
models/<run_name>/ppo_final.zip
```

## 4. Paramètres d'entraînement utiles

- `--task` : nom de la tâche (actuellement `PnPCounterToCab`)
- `--headless` : désactive le rendu temps réel
- `--custom_reward_shaping` : active la récompense dense
- `--total_timesteps` : nombre total de pas d'entraînement
- `--n_envs` : nombre d'environnements parallèles
- `--seed` : graine aléatoire
- `--horizon` : longueur max d'un épisode
- `--learning_rate` / `--batch_size` / `--n_steps` : hyperparamètres PPO

Paramètres de reward shaping (actifs avec `--custom_reward_shaping`) :

- `--reach_w` : poids du terme d'approche de l'objet
- `--grasp_bonus` : bonus de saisie
- `--place_bonus` : bonus de placement dans la zone cible
- `--success_bonus` : bonus final en cas de succès

## 5. Évaluer un modèle

Exemple :

```bash
uv run python scripts/eval_robocasa.py \
  --task PnPCounterToCab \
  --model_path models/<run_name>/ppo_final.zip \
  --episodes 10 \
  --save_video
```

Si `--save_video` est activé, les vidéos sont écrites dans `eval_videos/`.

## 6. Problèmes fréquents (FAQ)

### Q1 : `No module named 'stable_baselines3'`

Le package n'est pas présent dans l'environnement courant. Exécutez :

```bash
uv add stable-baselines3
```

Puis lancez avec `uv run python ...` pour éviter un mauvais interpréteur Python.

### Q2 : `AttributeError: 'NoneType' object has no attribute 'robot_model'`

C'est un problème d'ordre d'initialisation de l'environnement.  
Dans ce dépôt, le script d'entraînement a déjà été corrigé (reset avant `GymWrapper`).

### Q3 : Warnings robosuite (macro / mink / mimicgen)

Ce sont souvent des avertissements non bloquants.  
Si l'entraînement démarre et affiche les logs PPO, vous pouvez continuer.

## 7. Workflow expérimental recommandé

1. Tester d'abord avec `--total_timesteps 20000`  
2. Vérifier que la récompense et le taux de succès montent  
3. Augmenter ensuite à 300k / 500k / 1M  
4. Comparer plusieurs seeds avec les meilleurs hyperparamètres

## 8. Exemple de pipeline complet

```bash
uv run python scripts/train_robocasa.py --task PnPCounterToCab --headless --custom_reward_shaping --total_timesteps 300000 --n_envs 1
uv run python scripts/eval_robocasa.py --task PnPCounterToCab --model_path models/<run_name>/ppo_final.zip --episodes 10 --save_video
```

