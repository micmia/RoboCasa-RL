"""
PPO Training Script for Robocasa PnPCounterToCab
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyPnPCounterToCab
import gymnasium as gym
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback


def make_env(args, rank):
    """
    Utility function for multiprocessed env.
    
    :param task_name: the task class name
    :param rank: index of the subprocess
    :param seed: the inital seed for RNG
    """
    def _init():
        # Define Environment
        if args.task == "PnPCounterToCab":
            env_cls = MyPnPCounterToCab
        else:
             raise ValueError(f"Task {args.task} not supported yet in this script")

        robots = "PandaOmron" # Default robot
        controller_config = load_composite_controller_config(
            controller=None,
            robot=robots,
        )
        if args.headless:
            has_renderer = False
        else:
            has_renderer = True
        env = env_cls(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=False, 
            has_renderer=has_renderer, 
            has_offscreen_renderer=False,
            reward_shaping=True, 
            control_freq=20,
            renderer="mjviewer",
            # render_camera="robot0_robotview", # avoiding 'robot0_agentview_center' error
            ignore_done=False,
            seed=args.seed, 
            horizon=500,
        )
        
        # Wrap with GymWrapper
        # NOTE: Using keys=None due to a bug in RoboCasa's GymWrapper where explicit keys
        # cause observation space to not match actual observations. With keys=None, the
        # wrapper correctly handles the observation space.
        env = GymWrapper(env, keys=None)
        # Fix observation space mismatch (RoboCasa bug workaround)
        # env = FixedObservationWrapper(env)
        
        # Wrap with Monitor for SB3 logging
        log_dir = f"/tmp/gym/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=args.seed)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Training for Robocasa")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")

    
    args = parser.parse_args()
    
    # Create Vectorized Environment
    # Create a list of environment factories, one for each parallel environment
    env_fns = [make_env(args, i) for i in range(args.n_envs)]
    
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Define Model
    model = ...


if __name__ == "__main__":
    main()
