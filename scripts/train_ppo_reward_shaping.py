"""Train PPO with custom reward shaping (no curriculum)."""

import argparse
import csv
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
from robocasa.utils import object_utils as OU
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab


class AtomicRewardShapingWrapper(gym.Wrapper):
    """Adds dense reward terms and success signal."""

    def __init__(self, env, reach_reward=0.25, grasp_reward=0.5, place_reward=1.0, success_reward=5.0):
        super().__init__(env)
        self.reach_reward = reach_reward
        self.grasp_reward = grasp_reward
        self.place_reward = place_reward
        self.success_reward = success_reward

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = self._raw_env()
        shaped = 0.0
        try:
            info["success"] = bool(raw_env._check_success())
        except Exception:
            info.setdefault("success", False)
        try:
            obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]]
            eef_pos = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]]
            shaped += self.reach_reward * float(np.exp(-4.0 * np.linalg.norm(eef_pos - obj_pos)))
        except Exception:
            pass
        try:
            if OU.check_obj_grasped(raw_env, "obj"):
                shaped += self.grasp_reward
        except Exception:
            pass
        try:
            if OU.obj_inside_of(raw_env, "obj", "cab", partial_check=True, th=0.0):
                shaped += self.place_reward
        except Exception:
            pass
        try:
            if bool(raw_env._check_success()):
                shaped += self.success_reward
        except Exception:
            pass
        info["sparse_reward"] = float(reward)
        info["dense_reward"] = float(shaped)
        return obs, float(reward + shaped), terminated, truncated, info


class MetricsLoggerCallback(BaseCallback):
    """Log ep_rew_mean, ep_len_mean, success_rate in a CSV on each rollout."""

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
            raise ValueError(
                f"Task {args.task} is unsupported in this script. "
                "Use PnPCounterToCab for an atomic non-navigation task."
            )

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

        # Initialize robosuite internals before GymWrapper queries robot metadata.
        env.reset()
        env = GymWrapper(env, keys=None)
        env = AtomicRewardShapingWrapper(
            env,
            reach_reward=args.reach_reward,
            grasp_reward=args.grasp_reward,
            place_reward=args.place_reward,
            success_reward=args.success_reward,
        )

        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with reward shaping on RoboCasa.")
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Optional logs root. Default: models/<run_name>/logs",
    )
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for PPO (auto, cpu, cuda, cuda:0, ...). Default auto picks GPU if available.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Train PPO on CUDA (shortcut for --device cuda).",
    )

    parser.add_argument("--reach_reward", type=float, default=0.25)
    parser.add_argument("--grasp_reward", type=float, default=0.5)
    parser.add_argument("--place_reward", type=float, default=1.0)
    parser.add_argument("--success_reward", type=float, default=5.0)

    args = parser.parse_args()

    train_device = "cuda" if args.gpu else args.device

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"reward_shaping_{timestamp}"
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

    callback = MetricsLoggerCallback(log_path=os.path.join(log_root, "metrics.csv"))
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callback,
    )
    save_path = os.path.join(run_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
