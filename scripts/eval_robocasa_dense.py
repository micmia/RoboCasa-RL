"""
Evaluation script that applies the same dense + strict reward wrapper as training.
This avoids the common confusion where the raw env reward is sparse (often 0.0).
"""

import argparse
import os
import sys

import imageio
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from scripts.train_ppo_reward_shaping_dense import DenseStrictRewardWrapper, StrictStateCfg

from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers import load_composite_controller_config


VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame(raw_env, camera_names: list[str] = VIZ_CAMERAS, width: int = 256, height: int = 256):
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
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="eval_videos")
    parser.add_argument("--target", type=str, default="cab", choices=("cab", "bowl"))

    # Strict gates (match training defaults)
    parser.add_argument("--strict_grasp_lift_m", type=float, default=0.015)
    parser.add_argument("--strict_grasp_min_consecutive", type=int, default=4)
    parser.add_argument("--strict_grasp_follow_dist_m", type=float, default=0.035)
    parser.add_argument("--strict_gripper_open_qpos", type=float, default=0.040)
    parser.add_argument("--strict_grasp_min_close", type=float, default=0.60)
    parser.add_argument("--strict_inside_min_consecutive", type=int, default=6)
    parser.add_argument("--strict_inside_max_obj_speed", type=float, default=0.15)
    parser.add_argument("--allow_inside_while_grasped", action="store_true")

    args = parser.parse_args()

    if args.task != "PnPCounterToCab":
        raise ValueError(f"Unsupported task: {args.task}")

    has_offscreen = args.save_video

    robots = "PandaOmron"
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    env = MyPnPCounterToCab(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=has_offscreen,
        control_freq=20,
        ignore_done=False,
        horizon=1000,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        seed=args.seed,
        render_camera=VIZ_CAMERAS[0],
    )
    env.reset()
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
    env = DenseStrictRewardWrapper(env, target=args.target, strict_cfg=strict_cfg)

    model = PPO.load(args.model_path)

    run_name = os.path.basename(os.path.dirname(args.model_path)) or "eval_run"
    if args.save_video:
        os.makedirs(args.video_path, exist_ok=True)
        video_folder = os.path.join(args.video_path, run_name)
        os.makedirs(video_folder, exist_ok=True)

    success_count = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        frames = []
        episode_reward = 0.0
        last_terms = None

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            last_terms = dict(info.get("dense_terms", {}))

            if args.save_video:
                raw_env = env.env.env  # DenseStrictRewardWrapper -> GymWrapper -> raw env
                frames.append(render_tiled_frame(raw_env, VIZ_CAMERAS, width=256, height=256))

        is_success = bool(info.get("success", False))
        success_count += int(is_success)

        terms_str = ""
        if isinstance(last_terms, dict) and last_terms:
            terms_str = " | " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(last_terms.items()))
        print(f"Episode {ep+1}: Return={episode_reward:.2f}, Success={is_success}{terms_str}")

        if args.save_video and frames:
            vid_path = os.path.join(video_folder, f"{run_name}_ep_{ep}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"Saved multi-camera video ({len(VIZ_CAMERAS)} views) to {vid_path}")

    print(f"Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.2f}%)")


if __name__ == "__main__":
    main()

