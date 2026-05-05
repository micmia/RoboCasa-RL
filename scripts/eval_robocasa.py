"""PPO evaluation for RoboCasa training stacks (baseline, curriculum, reward shaping v0–v3)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Literal, Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.curriculum import CurriculumWrapper
from env.custom_pnp_apple_to_bowl import MyPnPAppleToBowl
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from train_ppo_reward_shaping_v1 import ObservationAugmentWrapper as CabObservationAugmentWrapper
from train_ppo_reward_shaping_v3 import ObservationAugmentWrapper as BowlObservationAugmentWrapper

VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

StackName = Literal[
    "auto",
    "baseline",
    "curriculum",
    "shaping_v0",
    "shaping_v1",
    "shaping_v2",
    "shaping_v3",
]
TaskName = Literal["counter_to_cab", "apple_to_bowl"]


class FreezeBaseActionWrapper(gym.Wrapper):
    """Match training: zero mobile-base action dims after the policy output (PandaOmron hybrid)."""

    def __init__(self, env: gym.Env, base_action_start: int = 8):
        super().__init__(env)
        self.base_action_start = int(base_action_start)

    def step(self, action):
        a = np.array(action, dtype=np.float32)
        if self.base_action_start < len(a):
            a[self.base_action_start :] = 0.0
        return self.env.step(a)


def render_tiled_frame(raw_env, camera_names=VIZ_CAMERAS, width=256, height=256):
    """Render each camera and stitch into a 2-column tiled image."""
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


def _path_suggests_v3(d: str) -> bool:
    return "v3" in d and "ppo" in d


def _infer_stack_from_path(model_path: str) -> Optional[StackName]:
    d = os.path.basename(os.path.dirname(os.path.abspath(model_path))).lower()
    if "curriculum" in d:
        return "curriculum"
    if "baseline" in d:
        return "baseline"
    if "reward_shaping_v3" in d or "shaping_v3" in d or _path_suggests_v3(d):
        return "shaping_v3"
    if "reward_shaping_v2" in d or "shaping_v2" in d or "_v2_" in d:
        return "shaping_v2"
    if "reward_shaping_v1" in d or "shaping_v1" in d or "_v1_" in d:
        return "shaping_v1"
    if "reward_shaping_v0" in d or "shaping_v0" in d or "_v0_" in d:
        return "shaping_v0"
    return None


def _heuristic_freeze_base(model_path: str) -> bool:
    d = os.path.basename(os.path.dirname(os.path.abspath(model_path))).lower()
    return "freeze" in d or "frozen" in d


def _make_raw_counter(seed: int, horizon: int, save_video: bool) -> MyPnPCounterToCab:
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    return MyPnPCounterToCab(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=save_video,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        seed=seed,
        render_camera=VIZ_CAMERAS[0],
    )


def _make_raw_apple(seed: int, horizon: int, save_video: bool) -> MyPnPAppleToBowl:
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    return MyPnPAppleToBowl(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=save_video,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        seed=seed,
        render_camera=VIZ_CAMERAS[0],
    )


def build_gym_env(
    stack: StackName,
    task: TaskName,
    *,
    seed: int,
    horizon: int,
    table_height: float,
    save_video: bool,
    curriculum_stage: int,
    freeze_base: bool,
    base_action_start: int,
) -> Tuple[gym.Env, object]:
    """Return (gym_env, raw_robosuite_env) matching the training wrapper order (without Monitor / reward shaping)."""
    if task == "apple_to_bowl":
        raw = _make_raw_apple(seed, horizon, save_video)
    else:
        raw = _make_raw_counter(seed, horizon, save_video)

    raw.reset()
    inner = raw
    if stack == "curriculum":
        inner = CurriculumWrapper(inner, initial_stage=curriculum_stage)
    gym_env: gym.Env = GymWrapper(inner, keys=None)

    if stack in ("shaping_v2", "shaping_v3") and freeze_base:
        gym_env = FreezeBaseActionWrapper(gym_env, base_action_start=base_action_start)

    if stack in ("shaping_v1", "shaping_v2"):
        gym_env = CabObservationAugmentWrapper(gym_env, table_height=table_height)
    elif stack == "shaping_v3":
        gym_env = BowlObservationAugmentWrapper(gym_env, table_height=table_height)

    return gym_env, raw


def _obs_dim(env: gym.Env) -> int:
    return int(np.prod(env.observation_space.shape))


def _resolve_stack_auto(
    model_path: str,
    expected_obs_dim: int,
    *,
    seed: int,
    horizon: int,
    table_height: float,
    save_video: bool,
    curriculum_stage: int,
    base_action_start: int,
    user_freeze_base: Optional[bool],
) -> Tuple[StackName, TaskName, bool]:
    """Pick stack/task/freeze_base from path heuristics, else observation-dimension search."""
    hinted = _infer_stack_from_path(model_path)
    if hinted is not None:
        task: TaskName = "apple_to_bowl" if hinted == "shaping_v3" else "counter_to_cab"
        freeze = _heuristic_freeze_base(model_path) if hinted in ("shaping_v2", "shaping_v3") else False
        g, _ = build_gym_env(
            hinted,
            task,
            seed=seed,
            horizon=horizon,
            table_height=table_height,
            save_video=save_video,
            curriculum_stage=curriculum_stage,
            freeze_base=freeze,
            base_action_start=base_action_start,
        )
        if _obs_dim(g) == expected_obs_dim:
            if user_freeze_base is not None and hinted in ("shaping_v2", "shaping_v3"):
                freeze = user_freeze_base
            return hinted, task, freeze
        print(
            f"Warning: path suggests stack={hinted} but obs dim {_obs_dim(g)} != model {expected_obs_dim}; "
            "falling back to dim search."
        )

    path_l = os.path.basename(os.path.dirname(os.path.abspath(model_path))).lower()
    candidates: list[Tuple[StackName, TaskName, bool]] = []

    def prefer_v3_first():
        for fz in (False, True):
            candidates.append(("shaping_v3", "apple_to_bowl", fz))

    def prefer_v2_v1():
        for fz in (False, True):
            candidates.append(("shaping_v2", "counter_to_cab", fz))
        candidates.append(("shaping_v1", "counter_to_cab", False))

    if "v3" in path_l or "apple" in path_l or "bowl" in path_l:
        prefer_v3_first()
        prefer_v2_v1()
    else:
        prefer_v2_v1()
        prefer_v3_first()

    path_l2 = path_l
    if "v0" in path_l2 or "shaping_v0" in path_l2:
        tie = [
            ("shaping_v0", "counter_to_cab", False),
            ("baseline", "counter_to_cab", False),
            ("curriculum", "counter_to_cab", False),
        ]
    elif "curriculum" in path_l2:
        tie = [
            ("curriculum", "counter_to_cab", False),
            ("baseline", "counter_to_cab", False),
            ("shaping_v0", "counter_to_cab", False),
        ]
    elif "baseline" in path_l2:
        tie = [
            ("baseline", "counter_to_cab", False),
            ("shaping_v0", "counter_to_cab", False),
            ("curriculum", "counter_to_cab", False),
        ]
    else:
        tie = [
            ("baseline", "counter_to_cab", False),
            ("shaping_v0", "counter_to_cab", False),
            ("curriculum", "counter_to_cab", False),
        ]
    candidates.extend(tie)

    seen: set[Tuple[StackName, TaskName, bool]] = set()
    ordered: list[Tuple[StackName, TaskName, bool]] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for st, tk, fz in ordered:
        g, _ = build_gym_env(
            st,
            tk,
            seed=seed,
            horizon=horizon,
            table_height=table_height,
            save_video=save_video,
            curriculum_stage=curriculum_stage,
            freeze_base=fz,
            base_action_start=base_action_start,
        )
        if _obs_dim(g) == expected_obs_dim:
            if user_freeze_base is not None and st in ("shaping_v2", "shaping_v3"):
                fz = user_freeze_base
            return st, tk, fz

    raise ValueError(
        f"Could not match observation dim {expected_obs_dim} to any known stack. "
        "Pass an explicit --stack (baseline|curriculum|shaping_v0|shaping_v1|shaping_v2|shaping_v3) "
        "and matching --task if needed."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint with the same env stack as training (obs + optional base freeze)."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument(
        "--stack",
        type=str,
        default="auto",
        choices=[
            "auto",
            "baseline",
            "curriculum",
            "shaping_v0",
            "shaping_v1",
            "shaping_v2",
            "shaping_v3",
        ],
        help="Training stack to mirror. 'auto' uses the run folder name and then observation-dim matching.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        choices=["", "counter_to_cab", "apple_to_bowl"],
        help="Raw env task; default is counter_to_cab except shaping_v3 (apple_to_bowl). Required if auto-match is ambiguous.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=700, help="Episode horizon (should match training).")
    parser.add_argument("--table_height", type=float, default=0.88, help="Must match training when using obs augment.")
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        default=2,
        help="Curriculum stage index when stack=curriculum (0 easiest …; default 2 = hardest in DEFAULT_STAGES).",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Zero base wheel actions (match v2/v3 training with --freeze_base). Also toggled by 'freeze' in run folder name in auto mode.",
    )
    parser.add_argument(
        "--no_freeze_base",
        action="store_true",
        help="Force-disable freeze_base even if the run folder name suggests it.",
    )
    parser.add_argument(
        "--base_action_start",
        type=int,
        default=8,
        help="Index from which base commands are zeroed (PandaOmron hybrid default 8).",
    )
    parser.add_argument(
        "--vecnorm_path",
        type=str,
        default="",
        help="Path to VecNormalize stats. Defaults to <model_dir>/vec_normalize.pkl.",
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", type=str, default="eval_videos")
    args = parser.parse_args()

    model_probe = PPO.load(args.model_path)
    expected_obs_dim = int(np.prod(model_probe.observation_space.shape))
    del model_probe

    stack_arg = args.stack
    if stack_arg == "auto":
        user_fb: Optional[bool] = None
        if args.no_freeze_base:
            user_fb = False
        elif args.freeze_base:
            user_fb = True
        stack, task, freeze_auto = _resolve_stack_auto(
            args.model_path,
            expected_obs_dim,
            seed=args.seed,
            horizon=args.horizon,
            table_height=args.table_height,
            save_video=args.save_video,
            curriculum_stage=args.curriculum_stage,
            base_action_start=args.base_action_start,
            user_freeze_base=user_fb,
        )
        freeze_base = freeze_auto
        print(f"Auto-selected stack={stack} task={task} freeze_base={freeze_base}")
    else:
        stack = stack_arg  # type: ignore[assignment]
        if args.task:
            task = args.task  # type: ignore[assignment]
        else:
            task = "apple_to_bowl" if stack == "shaping_v3" else "counter_to_cab"
        freeze_base = bool(args.freeze_base)
        if stack in ("shaping_v2", "shaping_v3") and not args.freeze_base and not args.no_freeze_base:
            if _heuristic_freeze_base(args.model_path):
                freeze_base = True
                print("Inferred freeze_base=True from model directory name.")
        if args.no_freeze_base:
            freeze_base = False
        g_chk, _ = build_gym_env(
            stack,
            task,
            seed=args.seed,
            horizon=args.horizon,
            table_height=args.table_height,
            save_video=args.save_video,
            curriculum_stage=args.curriculum_stage,
            freeze_base=freeze_base,
            base_action_start=args.base_action_start,
        )
        if _obs_dim(g_chk) != expected_obs_dim:
            raise ValueError(
                f"stack={stack} task={task} freeze_base={freeze_base} yields obs dim {_obs_dim(g_chk)}, "
                f"but model expects {expected_obs_dim}. Try --stack auto, another --task, or --freeze_base / --no_freeze_base."
            )

    gym_env, raw_env = build_gym_env(
        stack,
        task,
        seed=args.seed,
        horizon=args.horizon,
        table_height=args.table_height,
        save_video=args.save_video,
        curriculum_stage=args.curriculum_stage,
        freeze_base=freeze_base,
        base_action_start=args.base_action_start,
    )
    vec_env = DummyVecEnv([lambda: gym_env])

    vecnorm_path = args.vecnorm_path or os.path.join(
        os.path.dirname(os.path.abspath(args.model_path)), "vec_normalize.pkl"
    )
    if os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from: {vecnorm_path}")
    else:
        print("VecNormalize stats not found; evaluating without normalization.")

    model = PPO.load(args.model_path, env=vec_env)

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.model_path))) or "eval"
    if args.save_video:
        video_dir = os.path.join(args.video_dir, run_name)
        os.makedirs(video_dir, exist_ok=True)
        print(f"Video save directory: {os.path.abspath(video_dir)}")

    success_count = 0
    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        frames = []
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            episode_reward += float(rewards[0])
            if args.save_video:
                frames.append(render_tiled_frame(raw_env))

        is_success = bool(raw_env._check_success())
        success_count += is_success
        print(f"Episode {ep + 1:2d}: reward={episode_reward:7.2f}  success={is_success}")

        if args.save_video and frames:
            vid_path = os.path.join(video_dir, f"ep_{ep:02d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"  Saved video → {vid_path}")

    print(f"\nSuccess rate: {success_count}/{args.episodes} ({100 * success_count / args.episodes:.1f}%)")
    vec_env.close()


if __name__ == "__main__":
    main()
