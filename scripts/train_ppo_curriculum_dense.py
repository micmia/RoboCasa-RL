"""Train PPO with curriculum + dense/strict reward shaping."""

import os
import sys

# Ensure we import the vendored RoboCasa / robosuite packages as proper packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.curriculum import CurriculumWrapper
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab

import argparse
from collections import deque
from datetime import datetime

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from scripts.train_ppo_reward_shaping_dense import DenseStrictRewardWrapper, StrictStateCfg, MetricsLoggerCallback


class CurriculumCallback(BaseCallback):
    """
    Advances curriculum stages based on rolling success rate.
    Reads `info["success"]` at episode end.
    """

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

        strict_cfg = StrictStateCfg(
            grasp_lift_m=args.strict_grasp_lift_m,
            grasp_min_consecutive=args.strict_grasp_min_consecutive,
            grasp_follow_dist_m=args.strict_grasp_follow_dist_m,
            gripper_open_qpos=args.strict_gripper_open_qpos,
            grasp_min_close=args.strict_grasp_min_close,
            inside_min_consecutive=args.strict_inside_min_consecutive,
            inside_max_obj_speed=args.strict_inside_max_obj_speed,
            inside_require_release=(not args.allow_inside_while_grasped),
        )
        env = DenseStrictRewardWrapper(
            env,
            target=args.target,
            strict_cfg=strict_cfg,
            w_reach=args.w_reach,
            w_grasp=args.w_grasp,
            w_lift=args.w_lift,
            w_carry=args.w_carry,
            w_inside=args.w_inside,
            w_success=args.w_success,
            reach_temp=args.reach_temp,
            carry_temp=args.carry_temp,
        )

        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir, info_keywords=("success", "is_success", "dense_reward"))
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with curriculum + dense/strict shaping on RoboCasa.")
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=200_000)

    # PPO hyperparams (add knobs to reduce gripper jitter / improve stability)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    parser.add_argument("--log_root", type=str, default=None, help="Optional logs root. Default: models/<run_name>/logs")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu", action="store_true")

    # Curriculum
    parser.add_argument("--curriculum_window", type=int, default=100)
    parser.add_argument("--curriculum_min_timesteps", type=int, default=50_000)
    parser.add_argument("--curriculum_thresholds", type=str, default="0.70,0.80")

    # Target location
    parser.add_argument("--target", type=str, default="cab", choices=("cab", "bowl"))

    # Dense weights
    parser.add_argument("--w_reach", type=float, default=0.25)
    parser.add_argument("--w_grasp", type=float, default=0.35)
    parser.add_argument("--w_lift", type=float, default=0.20)
    parser.add_argument("--w_carry", type=float, default=0.35)
    parser.add_argument("--w_inside", type=float, default=0.75)
    parser.add_argument("--w_success", type=float, default=5.0)
    parser.add_argument("--reach_temp", type=float, default=4.0)
    parser.add_argument("--carry_temp", type=float, default=4.0)

    # Strict grasp gates
    parser.add_argument("--strict_grasp_lift_m", type=float, default=0.015)
    parser.add_argument("--strict_grasp_min_consecutive", type=int, default=4)
    parser.add_argument("--strict_grasp_follow_dist_m", type=float, default=0.035)
    parser.add_argument("--strict_gripper_open_qpos", type=float, default=0.040)
    parser.add_argument("--strict_grasp_min_close", type=float, default=0.60)

    # Strict inside gates
    parser.add_argument("--strict_inside_min_consecutive", type=int, default=6)
    parser.add_argument("--strict_inside_max_obj_speed", type=float, default=0.15)
    parser.add_argument("--allow_inside_while_grasped", action="store_true")

    args = parser.parse_args()

    train_device = "cuda" if args.gpu else args.device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"curriculum_dense_{args.target}_{timestamp}"
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
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tensorboard_dir,
        device=train_device,
        target_kl=args.target_kl,
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

