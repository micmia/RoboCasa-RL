"""
PPO Evaluation Script for Robocasa PnPCounterToCab
"""

import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import ...
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MyPnPCounterToCab

# Camera names used for multi-view visualisation
VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame( raw_env, camera_names: list[str] = VIZ_CAMERAS, width: int = 256, height: int = 256):
    """
    Render one frame from each camera and stitch them into a single
    tiled image arranged in a 2-column grid.

    Returns
    -------
    np.ndarray  shape (rows*height, cols*width, 3)  uint8
    """
    cols = 2
    rows = (len(camera_names) + cols - 1) // cols
    tile_rows = []
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(camera_names):
                frame = raw_env.sim.render(
                    camera_name=camera_names[idx],
                    width=width,
                    height=height,
                    depth=False,
                )
                frame = np.flipud(frame)
            else:
                # Pad with a black tile if camera count is odd
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="Task name: PnPCounterToCab or TurnOnMicrowave")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_video", action="store_true", help="Save video of evaluation")
    parser.add_argument("--video_path", type=str, default="eval_videos", help="Directory to save videos")
    args = parser.parse_args()

    # Select environment class based on task
    if args.task == "PnPCounterToCab":
        env_cls = MyPnPCounterToCab

    # Environment for evaluation (enable renderer if saving video)
    # Note: For video saving we need offscreen renderer
    has_offscreen = args.save_video
    
    robots = "PandaOmron"  # Default robot
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )
    env = env_cls(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=has_offscreen,
        control_freq=20,
        ignore_done=False,
        horizon=500,
        camera_names=VIZ_CAMERAS,          # register all four cameras
        camera_heights=256,
        camera_widths=256,
        seed=args.seed,
        render_camera=VIZ_CAMERAS[0],      # primary render camera
    )
    
    # Use same wrapper setup as training
    env = GymWrapper(env, keys=None)
    
    folder_name = args.model_path.split("/")[1]
    model = ...
    
    if args.save_video:
        os.makedirs(args.video_path, exist_ok=True)
        video_folder = os.path.join(args.video_path, folder_name)
        os.makedirs(video_folder, exist_ok=True)

    success_count = 0
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        frames = []
        episode_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if args.save_video:
                # Render all four cameras and stitch into one composite frame.
                # env stack: FixedObservationWrapper -> GymWrapper -> raw env
                raw_env = env.env.env
                composite = render_tiled_frame(raw_env, VIZ_CAMERAS, width=256, height=256)
                frames.append(composite)
        
        # Check success (if info has it, otherwise we might need custom check)
        # Note: Robosuite infos usually don't have 'is_success' standardly in GymWrapper unless added
        # But we can check internal logic if needed. 
        # For now, let's assume if return > threshold it might be good, 
        # OR we can manually check environment success
        is_success = env.env._check_success()
        if is_success:
            success_count += 1
            
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Success = {is_success}")
        
        if args.save_video and len(frames) > 0:
            vid_path = os.path.join(video_folder, f"eval_ep_{ep}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"Saved multi-camera video ({len(VIZ_CAMERAS)} views) to {vid_path}")

    print(f"Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.2f}%)")
    env.close()

if __name__ == "__main__":
    main()
