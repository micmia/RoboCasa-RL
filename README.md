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

### 4. Set up macros and download assets

```shell
cd robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

> **Note:** The kitchen assets download is approximately 10 GB.
