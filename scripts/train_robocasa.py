"""Train PPO on RoboCasa atomic tasks with PandaOmron and shaped rewards."""

import argparse
import os
import sys
from datetime import datetime
from collections import deque

import gymnasium as gym
import numpy as np
from robocasa.utils import object_utils as OU
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent directory to path to import local env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from env.curriculum import CurriculumWrapper


class AtomicRewardShapingWrapper(gym.Wrapper):
    """
    Adds dense reward terms for atomic manipulation:
    - reach object
    - grasp object
    - move object to goal fixture
    - complete task (sparse success)
    """

    def __init__(
        self,
        env,
        reach_w=0.25,
        grasp_bonus=0.5,
        place_bonus=1.0,
        success_bonus=5.0,
    ):
        super().__init__(env)
        self.reach_w = reach_w
        self.grasp_bonus = grasp_bonus
        self.place_bonus = place_bonus
        self.success_bonus = success_bonus

    def _raw_env(self):
        # GymWrapper -> robocasa task env
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = self._raw_env()

        shaped = 0.0
        # Provide an explicit boolean success signal for curriculum / logging.
        try:
            info["success"] = bool(raw_env._check_success())
        except Exception:
            # Best-effort: if unavailable, don't break training.
            info.setdefault("success", False)
        try:
            obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]]
            eef_pos = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]]
            dist = np.linalg.norm(eef_pos - obj_pos)
            shaped += self.reach_w * float(np.exp(-4.0 * dist))
        except Exception:
            pass

        try:
            if OU.check_obj_grasped(raw_env, "obj"):
                shaped += self.grasp_bonus
        except Exception:
            pass

        # For this script's default atomic task, target fixture ref is "cab".
        try:
            if OU.obj_inside_of(raw_env, "obj", "cab", partial_check=True, th=0.0):
                shaped += self.place_bonus
        except Exception:
            pass

        try:
            if bool(raw_env._check_success()):
                shaped += self.success_bonus
        except Exception:
            pass

        info["sparse_reward"] = float(reward)
        info["dense_reward"] = float(shaped)
        return obs, float(reward + shaped), terminated, truncated, info


class CurriculumCallback(BaseCallback):
    """
    Hybrid curriculum progression:
    - enforce a minimum number of timesteps per stage
    - then advance when success rate over a rolling window exceeds a threshold
    """

    def __init__(
        self,
        window_size: int = 100,
        min_timesteps_per_stage: int = 50_000,
        thresholds: tuple[float, ...] = (0.70, 0.80),
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.window_size = int(window_size)
        self.min_timesteps_per_stage = int(min_timesteps_per_stage)
        self.thresholds = tuple(float(x) for x in thresholds)

        self._recent = deque(maxlen=self.window_size)
        self._stage = 0
        self._stage_start_timesteps = 0

    def _on_training_start(self) -> None:
        self._stage = 0
        self._stage_start_timesteps = 0
        self._recent.clear()
        # ensure envs start at stage 0
        try:
            self.training_env.env_method("set_stage", self._stage)
        except Exception:
            pass

    def _maybe_advance(self) -> None:
        if self._stage >= len(self.thresholds):
            return
        if (self.num_timesteps - self._stage_start_timesteps) < self.min_timesteps_per_stage:
            return
        if len(self._recent) < self.window_size:
            return

        success_rate = float(np.mean(self._recent))
        if success_rate < self.thresholds[self._stage]:
            return

        self._stage += 1
        self._stage_start_timesteps = int(self.num_timesteps)
        self._recent.clear()
        self.training_env.env_method("set_stage", self._stage)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        # Record success only when an episode ends.
        for done, info in zip(dones, infos):
            if not done:
                continue
            if isinstance(info, dict) and "success" in info:
                self._recent.append(1.0 if bool(info["success"]) else 0.0)

        # Log for TensorBoard
        if len(self._recent) > 0:
            self.logger.record("curriculum/stage", float(self._stage))
            self.logger.record("curriculum/success_rate_window", float(np.mean(self._recent)))
            self.logger.record("curriculum/episodes_in_window", float(len(self._recent)))

        self._maybe_advance()
        return True


def make_env(args, rank):
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
        if args.curriculum:
            env = CurriculumWrapper(env, initial_stage=0)
        env = GymWrapper(env, keys=None)
        if args.custom_reward_shaping:
            env = AtomicRewardShapingWrapper(
                env,
                reach_w=args.reach_w,
                grasp_bonus=args.grasp_bonus,
                place_bonus=args.place_bonus,
                success_bonus=args.success_bonus,
            )

        log_dir = os.path.join(args.log_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on RoboCasa atomic task with PandaOmron."
    )
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--log_root", type=str, default="/tmp/robocasa_rl_logs")
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

    parser.add_argument("--custom_reward_shaping", action="store_true")
    parser.add_argument("--reach_w", type=float, default=0.25)
    parser.add_argument("--grasp_bonus", type=float, default=0.5)
    parser.add_argument("--place_bonus", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=5.0)

    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (phase 3).")
    parser.add_argument("--curriculum_window", type=int, default=100)
    parser.add_argument("--curriculum_min_timesteps", type=int, default=50_000)
    parser.add_argument(
        "--curriculum_thresholds",
        type=str,
        default="0.70,0.80",
        help="Comma-separated success thresholds for stage transitions (e.g. '0.70,0.80').",
    )

    args = parser.parse_args()

    train_device = "cuda" if args.gpu else args.device

    env_fns = [make_env(args, i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_root,
        device=train_device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.task}_PandaOmron_{timestamp}"
    save_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    callback = None
    if args.curriculum:
        thresholds = tuple(float(x) for x in args.curriculum_thresholds.split(",") if x.strip() != "")
        callback = CurriculumCallback(
            window_size=args.curriculum_window,
            min_timesteps_per_stage=args.curriculum_min_timesteps,
            thresholds=thresholds,
        )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback)
    save_path = os.path.join(save_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
