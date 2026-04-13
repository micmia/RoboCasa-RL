# RoboCasa-RL

Reinforcement learning experiments built on top of [RoboCasa](https://github.com/robocasa/robocasa) and [robosuite](https://github.com/ARISE-Initiative/robosuite).

## Installation

These instructions use [`uv`](https://docs.astral.sh/uv/) for environment and package management. Install it first if you haven't:

```shell
pip install uv
```

### 1. Create and activate a virtual environment

```shell
uv venv --python 3.11
```

```shell
# macOS/Linux:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate
```

### 2. Install robosuite

```shell
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
uv pip install -e .
cd ..
```

### 3. Install robocasa

```shell
git clone https://github.com/robocasa/robocasa
cd robocasa
uv pip install -e .
cd ..
```

### 4. Install project dependencies

```shell
uv pip install -e .
```

### 5. Set up macros and download assets

```shell
cd robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
cd ..
```

> **Note:** The kitchen assets download is approximately 10 GB.

## Usage

### Demo

```shell
# Windows/Linux:
python robocasa/robocasa/demos/demo_kitchen_scenes.py
# macOS:
mjpython robocasa/robocasa/demos/demo_kitchen_scenes.py
```

### Training (PPO, atomic `PnPCounterToCab`)

From the repo root, with the venv activated. The atomic task **`PnPCounterToCab`** uses the kitchen **counter** as a fixture; in this repo the **object placed on that counter** is always the apple **`apple_1`** (see `env/custom_pnp_counter_to_cab.py`). The goal is to pick it up and place it in the cabinet. Use `--headless` on servers without a display.

**Baseline** (no custom dense shaping):

```shell
uv run python scripts/train_ppo_baseline.py \
  --headless \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name baseline_seed42
```

**Reward shaping** (dense reach / grasp / place / success bonuses):

```shell
uv run python scripts/train_ppo_reward_shaping.py \
  --task PnPCounterToCab \
  --headless \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name reward_shaping_seed42
```

**Curriculum** (staged difficulty; optional):

```shell
uv run python scripts/train_ppo_curriculum.py \
  --task PnPCounterToCab \
  --headless \
  --total_timesteps 300000 \
  --n_envs 1 \
  --run_name curriculum_seed42
```

Checkpoints and logs are written under `models/<run_name>/` (e.g. `ppo_final.zip`, `logs/metrics.csv`, TensorBoard under `logs/tensorboard/`). More flags and behavior are documented in [`docs/ATOMIC_TASK_USAGE.md`](docs/ATOMIC_TASK_USAGE.md).

### Evaluation

Load a trained policy and run rollouts (deterministic actions by default). Replace `<run_name>` with the directory you used when training.

```shell
uv run python scripts/eval_robocasa.py \
  --task PnPCounterToCab \
  --model_path models/<run_name>/ppo_final.zip \
  --episodes 10 \
  --save_video
```

Videos are saved under `eval_videos/<run_name>/` when `--save_video` is set (see `--video_path` to change the root).
