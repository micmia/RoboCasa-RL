"""Train PPO on RoboCasa atomic tasks with PandaOmron and shaped rewards."""

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
from robocasa.utils import object_utils as OU
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent directory to path to import local env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab


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

    parser.add_argument("--custom_reward_shaping", action="store_true")
    parser.add_argument("--reach_w", type=float, default=0.25)
    parser.add_argument("--grasp_bonus", type=float, default=0.5)
    parser.add_argument("--place_bonus", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=5.0)

    args = parser.parse_args()

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
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.task}_PandaOmron_{timestamp}"
    save_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    save_path = os.path.join(save_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
