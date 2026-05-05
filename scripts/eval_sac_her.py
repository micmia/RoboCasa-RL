"""Evaluate SAC+HER bowl checkpoint with the same goal/wrapper stack as training.

Default checkpoint:
    models/sac_her/checkpoints/sac_her_350000_steps.zip

Example:
    .venv/bin/python scripts/eval_sac_her.py --episodes 20 --save_video
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import gymnasium as gym
import imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from scripts.train_sac_her_bowl import (
    BowlGoalWrapper,
    DOF_SCHEDULE,
    GripperOverrideWrapper,
    ScaledBaseWrapper,
    set_reward_phase,
)

VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame(raw_env, camera_names=VIZ_CAMERAS, width=256, height=256):
    cols = 2
    rows = (len(camera_names) + cols - 1) // cols
    tile_rows = []
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(camera_names):
                frame = np.flipud(
                    raw_env.sim.render(
                        camera_name=camera_names[idx],
                        width=width,
                        height=height,
                        depth=False,
                    )
                )
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def _extract_steps_from_ckpt(path: str) -> int | None:
    name = os.path.basename(path)
    m = re.search(r"_(\d+)_steps\.zip$", name)
    if m:
        return int(m.group(1))
    return None


def _dof_scales_for_steps(steps: int) -> dict[str, float]:
    chosen = {"xy": 0.0, "rot": 0.0, "torso": 0.0}
    for ts, scales in sorted(DOF_SCHEDULE, key=lambda x: x[0]):
        if steps >= ts:
            chosen = dict(scales)
    return chosen


def make_eval_env(seed: int, horizon: int, save_video: bool, pre_grasp_mode: str):
    robots = "PandaOmron"
    cc = load_composite_controller_config(controller=None, robot=robots)
    raw = MyPnPCounterToCab(
        robots=robots,
        controller_configs=cc,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=save_video,
        reward_shaping=False,
        control_freq=20,
        ignore_done=False,
        seed=seed,
        horizon=horizon,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        render_camera=VIZ_CAMERAS[0],
    )
    raw.curriculum_pre_grasp = pre_grasp_mode if pre_grasp_mode != "none" else None
    raw.reset()
    env: gym.Env = GymWrapper(raw, keys=None)
    env = GripperOverrideWrapper(env)
    env = ScaledBaseWrapper(env)
    env = BowlGoalWrapper(env)
    return env, raw


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC+HER bowl model checkpoint.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/sac_her/checkpoints/sac_her_350000_steps.zip",
        help="Path to SAC+HER checkpoint (.zip).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--reward_phase",
        type=str,
        default="1C",
        choices=["1A", "1B", "1C"],
        help="Evaluation reward/success phase used inside BowlGoalWrapper.",
    )
    parser.add_argument(
        "--pre_grasp_mode",
        type=str,
        default="partial",
        choices=["full", "partial", "none"],
        help="Initial curriculum mode at reset.",
    )
    parser.add_argument(
        "--dof_schedule_mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="auto: infer scales from checkpoint step and DOF_SCHEDULE; manual: use args below.",
    )
    parser.add_argument("--base_xy_scale", type=float, default=0.1)
    parser.add_argument("--base_rot_scale", type=float, default=0.05)
    parser.add_argument("--torso_scale", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy for evaluation (default stochastic if not set).",
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", type=str, default="eval_videos/checkpoints")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    set_reward_phase(args.reward_phase)
    env, raw = make_eval_env(
        seed=args.seed,
        horizon=args.horizon,
        save_video=args.save_video,
        pre_grasp_mode=args.pre_grasp_mode,
    )

    if args.dof_schedule_mode == "auto":
        steps = _extract_steps_from_ckpt(args.model_path)
        if steps is None:
            scales = {
                "xy": args.base_xy_scale,
                "rot": args.base_rot_scale,
                "torso": args.torso_scale,
            }
            print("Could not infer step from checkpoint name; fallback to manual scales.")
        else:
            scales = _dof_scales_for_steps(steps)
            print(f"Inferred checkpoint steps={steps}; using DOF scales={scales}")
    else:
        scales = {
            "xy": args.base_xy_scale,
            "rot": args.base_rot_scale,
            "torso": args.torso_scale,
        }
        print(f"Using manual DOF scales={scales}")

    # Wrapper chain: BowlGoalWrapper -> ScaledBaseWrapper -> GripperOverrideWrapper -> GymWrapper -> raw
    env.env.set_dof_scales(scales["xy"], scales["rot"], scales["torso"])

    model = SAC.load(args.model_path, env=env)

    run_name = os.path.splitext(os.path.basename(args.model_path))[0]
    if args.save_video:
        out_dir = os.path.join(args.video_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Video output: {os.path.abspath(out_dir)}")

    success_count = 0
    returns = []
    min_dxy_list = []
    min_d3_list = []
    release_dxy_list = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        frames = []
        min_dxy = float("inf")
        min_d3 = float("inf")
        prev_open = 0.0
        first_release_dxy = None
        info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)

            dxy = float(info.get("d_xy", np.nan))
            if np.isfinite(dxy):
                min_dxy = min(min_dxy, dxy)

            d3 = float(info.get("place_dist", np.nan))
            if np.isfinite(d3):
                min_d3 = min(min_d3, d3)

            open_now = float(info.get("gripper_open", 0.0))
            if open_now > 0.5 and prev_open <= 0.5 and np.isfinite(dxy) and first_release_dxy is None:
                first_release_dxy = dxy
            prev_open = open_now

            if args.save_video:
                frames.append(render_tiled_frame(raw))

        success = bool(info.get("is_success", 0.0) > 0.5)
        success_count += int(success)
        returns.append(ep_return)
        min_dxy_list.append(min_dxy if np.isfinite(min_dxy) else np.nan)
        min_d3_list.append(min_d3 if np.isfinite(min_d3) else np.nan)
        if first_release_dxy is not None:
            release_dxy_list.append(first_release_dxy)

        print(
            f"Ep {ep+1:02d} | return={ep_return:8.2f} | success={success} | "
            f"min_dxy={100*min_dxy:5.1f}cm | min_d3={100*min_d3:5.1f}cm | "
            f"release_dxy={(100*first_release_dxy):5.1f}cm" if first_release_dxy is not None
            else f"Ep {ep+1:02d} | return={ep_return:8.2f} | success={success} | "
            f"min_dxy={100*min_dxy:5.1f}cm | min_d3={100*min_d3:5.1f}cm | release_dxy=  N/A"
        )

        if args.save_video and frames:
            video_path = os.path.join(out_dir, f"ep_{ep:02d}.mp4")
            imageio.mimsave(video_path, frames, fps=20)
            print(f"  saved: {video_path}")

    success_rate = 100.0 * success_count / max(args.episodes, 1)
    print("\n=== Evaluation Summary ===")
    print(f"Model:         {args.model_path}")
    print(f"Episodes:      {args.episodes}")
    print(f"Success rate:  {success_count}/{args.episodes} ({success_rate:.1f}%)")
    print(f"Return mean:   {np.nanmean(returns):.2f}")
    print(f"Min d_xy mean: {100*np.nanmean(min_dxy_list):.2f} cm")
    print(f"Min d3 mean:   {100*np.nanmean(min_d3_list):.2f} cm")
    if release_dxy_list:
        print(f"Release d_xy:  {100*np.nanmean(release_dxy_list):.2f} cm (first release event)")
    else:
        print("Release d_xy:  N/A (no release events detected)")


if __name__ == "__main__":
    main()

