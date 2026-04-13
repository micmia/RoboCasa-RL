"""Train PPO with curriculum learning (no custom reward shaping)."""

import argparse
import csv
import os
import sys
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.curriculum import CurriculumWrapper
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab


class SuccessInfoWrapper(gym.Wrapper):
    """Attach success booleans for metrics/curriculum."""

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = self._raw_env()
        try:
            info["success"] = bool(raw_env._check_success())
            info["is_success"] = bool(info["success"])
        except Exception:
            info.setdefault("success", False)
            info.setdefault("is_success", False)
        return obs, reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    def __init__(self, window_size=100, min_timesteps_per_stage=50_000, thresholds=(0.70, 0.80), verbose=0):
        super().__init__(verbose=verbose)
        self.window_size = int(window_size)
        self.min_timesteps_per_stage = int(min_timesteps_per_stage)
        self.thresholds = tuple(float(x) for x in thresholds)
        self._recent = deque(maxlen=self.window_size)
        self._stage = 0
        self._stage_start_timesteps = 0

    def _on_training_start(self):
        self._stage = 0
        self._stage_start_timesteps = 0
        self._recent.clear()
        try:
            self.training_env.env_method("set_stage", self._stage)
        except Exception:
            pass

    def _maybe_advance(self):
        if self._stage >= len(self.thresholds):
            return
        if (self.num_timesteps - self._stage_start_timesteps) < self.min_timesteps_per_stage:
            return
        if len(self._recent) < self.window_size:
            return
        if float(np.mean(self._recent)) < self.thresholds[self._stage]:
            return
        self._stage += 1
        self._stage_start_timesteps = int(self.num_timesteps)
        self._recent.clear()
        self.training_env.env_method("set_stage", self._stage)

    def _on_step(self):
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and "success" in info:
                self._recent.append(1.0 if bool(info["success"]) else 0.0)
        if len(self._recent) > 0:
            self.logger.record("curriculum/stage", float(self._stage))
            self.logger.record("curriculum/success_rate_window", float(np.mean(self._recent)))
            self.logger.record("curriculum/episodes_in_window", float(len(self._recent)))
        self._maybe_advance()
        return True


class MetricsLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.writer = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.csv_file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["timestep", "ep_rew_mean", "ep_len_mean", "success_rate", "n_episodes"],
        )
        self.writer.writeheader()

    def _on_rollout_end(self):
        if not self.model.ep_info_buffer:
            return
        ep_rews = [ep["r"] for ep in self.model.ep_info_buffer]
        ep_lens = [ep["l"] for ep in self.model.ep_info_buffer]
        successes = [ep.get("is_success", 0) for ep in self.model.ep_info_buffer]
        self.writer.writerow(
            {
                "timestep": self.num_timesteps,
                "ep_rew_mean": float(np.mean(ep_rews)),
                "ep_len_mean": float(np.mean(ep_lens)),
                "success_rate": float(np.mean(successes)),
                "n_episodes": len(ep_rews),
            }
        )
        self.csv_file.flush()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


def make_env(args, rank, monitor_root):
    def _init():
        if args.task != "PnPCounterToCab":
            raise ValueError("Use PnPCounterToCab for this curriculum script.")
        robots = "PandaOmron"
        controller_config = load_composite_controller_config(controller=None, robot=robots)
        env = MyPnPCounterToCab(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=False,
            has_renderer=not args.headless,
            has_offscreen_renderer=False,
            reward_shaping=True,
            control_freq=20,
            renderer="mjviewer",
            ignore_done=False,
            seed=args.seed + rank,
            horizon=args.horizon,
        )
        env.reset()
        env = CurriculumWrapper(env, initial_stage=0)
        env = GymWrapper(env, keys=None)
        env = SuccessInfoWrapper(env)
        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with curriculum on RoboCasa.")
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--log_root", type=str, default=None, help="Optional logs root. Default: models/<run_name>/logs")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--curriculum_window", type=int, default=100)
    parser.add_argument("--curriculum_min_timesteps", type=int, default=50_000)
    parser.add_argument("--curriculum_thresholds", type=str, default="0.70,0.80")
    args = parser.parse_args()

    train_device = "cuda" if args.gpu else args.device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.task}_Curriculum_PandaOmron_{timestamp}"
    run_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_root = args.log_root if args.log_root is not None else os.path.join(run_dir, "logs")
    tensorboard_dir = os.path.join(log_root, "tensorboard")
    monitor_root = os.path.join(log_root, "monitor")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(monitor_root, exist_ok=True)

    env_fns = [make_env(args, i, monitor_root) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tensorboard_dir,
        device=train_device,
    )
    thresholds = tuple(float(x) for x in args.curriculum_thresholds.split(",") if x.strip() != "")
    callbacks = CallbackList(
        [
            MetricsLoggerCallback(log_path=os.path.join(log_root, "metrics.csv")),
            CurriculumCallback(
                window_size=args.curriculum_window,
                min_timesteps_per_stage=args.curriculum_min_timesteps,
                thresholds=thresholds,
            ),
        ]
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callbacks)
    save_path = os.path.join(run_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
