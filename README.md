# RoboCasa-RL

Reinforcement learning experiments on RoboCasa kitchen manipulation: **PPO** with dense reward shaping (v1–v3) and **SAC + HER** with curriculum. Built on [RoboCasa](https://github.com/robocasa/robocasa) and [robosuite](https://github.com/ARISE-Initiative/robosuite).

**Report:** [experiment_report.pdf](reports/experiment_report.pdf) (PPO vs SAC+HER, tasks, metrics, and failure modes).

## Repository layout

| Path | Role |
|------|------|
| `scripts/` | Training and evaluation CLIs (`train_ppo_reward_shaping_*.py`, `train_sac_her_bowl.py`, `eval_robocasa.py`, `eval_sac_her.py`, …) |
| `env/` | Custom gym-style stacks and task wrappers used by the trainers |
| `reports/` | LaTeX/PDF coursework report and figures |
| `models/` | Saved checkpoints (created when you train; examples below use run folders that match this repo when present) |

## Installation

These instructions use [uv](https://docs.astral.sh/uv/) for environment and package management. Install it first if you haven't:

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

Run everything from the **repository root** with the venv activated (`source .venv/bin/activate` or `uv run …`).

### Demo

```shell
# Windows/Linux:
python robocasa/robocasa/demos/demo_kitchen_scenes.py
# macOS:
mjpython robocasa/robocasa/demos/demo_kitchen_scenes.py
```

### Training — PPO v1 and v2 (cabinet, `PnPCounterToCab`)

The atomic task **PnPCounterToCab** uses the kitchen counter; the manipulated object is the apple **apple_1** (`env/custom_pnp_counter_to_cab.py`). Goal: pick and place into the cabinet. Use `--headless` on servers without a display.

**v1** (default horizon **700**, **3M** steps):

```shell
uv run python scripts/train_ppo_reward_shaping_v1.py \
  --task PnPCounterToCab \
  --headless \
  --total_timesteps 3000000 \
  --n_envs 1 \
  --run_name ppo_reward_shaping_v1_20260425_155559
```

**v2** (default horizon **700**; contact / grasp-hold / lift-sustain terms; **3M** steps in this example):

```shell
uv run python scripts/train_ppo_reward_shaping_v2.py \
  --headless \
  --total_timesteps 3000000 \
  --n_envs 1 \
  --run_name ppo_reward_shaping_v2_20260424_170825
```

Artifacts: `models/<run_name>/ppo_final.zip`, `vec_normalize.pkl` (unless `--no_vecnorm`), periodic `checkpoints/ppo_ckpt_*_steps.zip`, `logs/metrics.csv`, `logs/tensorboard/`. In evaluation, pass `--stack shaping_v1` or `--stack shaping_v2` to match training.

### Training — PPO v3 (apple-to-bowl, `train_ppo_reward_shaping_v3.py`)

Default horizon **900**. The script **defaults to loading** a prior checkpoint and VecNormalize stats for fine-tuning; to train **from scratch**, clear those paths (empty string). Example:

```shell
uv run python scripts/train_ppo_reward_shaping_v3.py \
  --headless \
  --horizon 900 \
  --total_timesteps 3000000 \
  --n_envs 1 \
  --run_name ppo_reward_shaping_v3_my_run
```

To **fine-tune** instead, omit `--load_model` / `--load_vecnorm` or set them to your `.zip` / `.pkl` pair.

GUI (single env): `--no-headless` instead of `--headless`.

### Evaluation — PPO (`eval_robocasa.py`)

`--stack` must match training (`shaping_v1`, `shaping_v2`, `shaping_v3`). With `--stack auto`, the script infers from the model path and observation size when possible.

**v1 — cabinet** (horizon **700**):

```shell
uv run python scripts/eval_robocasa.py \
  --model_path models/ppo_reward_shaping_v1_20260425_155559/ppo_final.zip \
  --stack shaping_v1 \
  --task counter_to_cab \
  --horizon 700 \
  --episodes 10 \
  --save_video
```

**v2 — cabinet** (horizon **700**):

```shell
uv run python scripts/eval_robocasa.py \
  --model_path models/ppo_reward_shaping_v2_20260424_170825/ppo_final.zip \
  --stack shaping_v2 \
  --task counter_to_cab \
  --horizon 700 \
  --episodes 10 \
  --save_video
```

**v3 — apple-to-bowl** (horizon **900**). If you have `models/<run>/ppo_final.zip` and `vec_normalize.pkl` in the same directory, `--vecnorm_path` is optional. Many local runs only keep periodic checkpoints under `checkpoints/`; then pass a **matching** policy zip and VecNormalize pickle (same step count `<N>`):

```shell
uv run python scripts/eval_robocasa.py \
  --model_path models/ppo_reward_shaping_v3_20260427_000723/checkpoints/ppo_ckpt_900000_steps.zip \
  --vecnorm_path models/ppo_reward_shaping_v3_20260427_000723/checkpoints/ppo_ckpt_vecnormalize_900000_steps.pkl \
  --stack shaping_v3 \
  --task apple_to_bowl \
  --horizon 900 \
  --episodes 5 \
  --save_video \
  --video_dir eval_videos
```

Videos: under `--video_dir`, in a subfolder named after the parent directory of the model file (often `…/eval_videos/checkpoints/ep_00.mp4` when loading from `…/checkpoints/*.zip`).

### Training — SAC + HER (bowl, `train_sac_her_bowl.py`)

Default `--total_timesteps` is **200000**; built-in curricula (reward phase, pre-grasp, success radius, base DOF) use milestones up to **700k** steps, so a full run should use a larger budget, for example:

```shell
uv run python scripts/train_sac_her_bowl.py \
  --headless \
  --horizon 400 \
  --total_timesteps 700000 \
  --run_name sac_her_bowl_my_run
```

Outputs: `models/<run_name>/sac_her_final.zip` and `models/<run_name>/checkpoints/sac_her_<steps>_steps.zip`.

### Evaluation — SAC + HER (`eval_sac_her.py`)

Default checkpoint in the script points at `models/sac_her/checkpoints/sac_her_350000_steps.zip` if you do not pass `--model_path`. Typical invocation:

```shell
uv run python scripts/eval_sac_her.py \
  --model_path models/sac_her/checkpoints/sac_her_350000_steps.zip \
  --reward_phase 1C \
  --pre_grasp_mode partial \
  --horizon 400 \
  --episodes 10 \
  --save_video
```

With `--save_video`, files go under `eval_videos/checkpoints/<checkpoint_stem>/` (e.g. `eval_videos/checkpoints/sac_her_350000_steps/ep_00.mp4`).