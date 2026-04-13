"""Baseline PPO on RoboCasa atomic tasks with PandaOmron — native reward only."""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
import csv
from stable_baselines3.common.callbacks import BaseCallback

class MetricsLoggerCallback(BaseCallback):
    """Logue ep_rew_mean, ep_len_mean, success_rate dans un CSV à chaque rollout."""

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
        self.writer.writerow({
            "timestep":    self.num_timesteps,
            "ep_rew_mean": float(np.mean(ep_rews)),
            "ep_len_mean": float(np.mean(ep_lens)),
            "success_rate": float(np.mean(successes)),
            "n_episodes":  len(ep_rews),
        })
        self.csv_file.flush()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()
            

def make_env(args, rank, log_root):
    def _init():
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
        env = GymWrapper(env, keys=None)
        # ✂️ DIFF 1 : pas de AtomicRewardShapingWrapper ici

        log_dir = os.path.join(log_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Baseline PPO on RoboCasa atomic task with PandaOmron."
    )
    # ✂️ DIFF 2 : pas de --task (toujours PnPCounterToCab)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")
    # ✂️ DIFF 3 : pas de --custom_reward_shaping ni ses hyperparamètres

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"baseline_{timestamp}"
    run_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_root = os.path.join(run_dir, "logs")
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
    )

    #model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    metrics_cb = MetricsLoggerCallback(
        log_path=os.path.join(log_root, "metrics.csv")
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=metrics_cb,
        progress_bar=True,
    )
    save_path = os.path.join(run_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()