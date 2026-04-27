"""Train PPO with denser, clearer reward shaping + stricter state gates."""

import os
import sys

# Ensure we import the vendored RoboCasa / robosuite packages as proper packages,
# not as namespace packages from the workspace root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.utils import object_utils as OU
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


@dataclass(frozen=True)
class StrictStateCfg:
    # Grasp stability
    grasp_lift_m: float = 0.015
    grasp_min_consecutive: int = 4
    grasp_follow_dist_m: float = 0.035
    gripper_open_qpos: float = 0.040  # used to normalize closure; approximate for Panda gripper
    grasp_min_close: float = 0.60  # closure in [0,1]
    # Inside stability
    inside_min_consecutive: int = 6
    inside_max_obj_speed: float = 0.15
    # When object is inside, prefer it released
    inside_require_release: bool = True


def _get_gripper_close_fraction(raw_env, open_qpos: float) -> float:
    """
    Returns a closure fraction in [0, 1], where 1 means fully closed.
    Uses the same finger joints as OU.check_obj_grasped().
    """
    joints = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
    qpos = []
    for j in joints:
        addr = raw_env.sim.model.get_joint_qpos_addr(j)
        qpos.append(float(raw_env.sim.data.qpos[addr]))
    mean_qpos = float(np.mean(qpos))
    denom = max(float(open_qpos), 1e-6)
    close = 1.0 - (mean_qpos / denom)
    return float(np.clip(close, 0.0, 1.0))


def _cabinet_int_center(raw_env, fixture_id: str = "cab") -> np.ndarray | None:
    try:
        fixture = raw_env.get_fixture(fixture_id)
        regions = fixture.get_int_sites(relative=False)
        if not regions:
            return None
        (p0, px, py, pz) = next(iter(regions.values()))
        return np.mean(np.stack([p0, px, py, pz], axis=0), axis=0)
    except Exception:
        return None


def _obj_world_pos(raw_env, obj_key: str = "obj") -> np.ndarray:
    return np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id[obj_key]], dtype=np.float64)


def _obj_world_quat_xyzw(raw_env, obj_key: str) -> np.ndarray:
    body_id = raw_env.obj_body_id[obj_key]
    quat_wxyz = np.array(raw_env.sim.data.body_xquat[body_id], dtype=np.float64)
    return np.array(T.convert_quat(quat_wxyz, to="xyzw"), dtype=np.float64)


def _point_in_aabb(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> bool:
    return bool(np.all(p >= (lo - margin)) and np.all(p <= (hi + margin)))


def _obj_inside_object_aabb(
    raw_env,
    obj_key: str,
    container_key: str,
    *,
    partial_check: bool = True,
    margin: float = 0.002,
) -> bool:
    """
    Approximate 'obj inside container' using container's world AABB expressed in the
    container frame, then checking obj position (or bbox corners) in that frame.
    Works for object receptacles like a bowl (coarse but stable).
    """
    obj = raw_env.objects[obj_key]
    container = raw_env.objects[container_key]

    c_pos = _obj_world_pos(raw_env, container.name)
    c_quat = _obj_world_quat_xyzw(raw_env, container.name)
    R_wc = T.quat2mat(c_quat)

    # Container AABB in container frame (compute from its bbox points)
    c_corners_w = container.get_bbox_points(trans=c_pos, rot=c_quat)
    c_corners_c = (R_wc.T @ (np.asarray(c_corners_w).T - c_pos.reshape(3, 1))).T
    lo = np.min(c_corners_c, axis=0)
    hi = np.max(c_corners_c, axis=0)

    obj_pos_w = _obj_world_pos(raw_env, obj.name)
    obj_quat = _obj_world_quat_xyzw(raw_env, obj.name)

    if partial_check:
        points_w = [obj_pos_w]
    else:
        points_w = obj.get_bbox_points(trans=obj_pos_w, rot=obj_quat)

    for p_w in points_w:
        p_c = R_wc.T @ (np.asarray(p_w) - c_pos)
        if not _point_in_aabb(p_c, lo, hi, margin=margin):
            return False
    return True


class DenseStrictRewardWrapper(gym.Wrapper):
    """
    Denser shaping with explicit strict gates for:
    - grasped (stability / lift / follow + close)
    - inside_target (stability streak + low speed [+ optional release])
    """

    def __init__(
        self,
        env,
        *,
        target: str = "cab",
        strict_cfg: StrictStateCfg = StrictStateCfg(),
        # reward weights
        w_reach: float = 0.25,
        w_grasp: float = 0.35,
        w_lift: float = 0.20,
        w_carry: float = 0.35,
        w_inside: float = 0.75,
        w_success: float = 5.0,
        reach_temp: float = 4.0,
        carry_temp: float = 4.0,
    ):
        super().__init__(env)
        self.target = str(target)
        self.cfg = strict_cfg

        self.w_reach = float(w_reach)
        self.w_grasp = float(w_grasp)
        self.w_lift = float(w_lift)
        self.w_carry = float(w_carry)
        self.w_inside = float(w_inside)
        self.w_success = float(w_success)
        self.reach_temp = float(reach_temp)
        self.carry_temp = float(carry_temp)

        # streaks / baselines (set on reset)
        self._obj_z0 = 0.0
        self._raw_grasp_streak = 0
        self._inside_streak = 0

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_env = self._raw_env()
        try:
            self._obj_z0 = float(_obj_world_pos(raw_env, "obj")[2])
        except Exception:
            self._obj_z0 = 0.0
        self._raw_grasp_streak = 0
        self._inside_streak = 0
        return obs, info

    def _target_center(self, raw_env) -> np.ndarray | None:
        if self.target == "cab":
            return _cabinet_int_center(raw_env, "cab")
        if self.target == "bowl":
            # distractor bowl in this custom env is named "distr_counter"
            try:
                return _obj_world_pos(raw_env, "distr_counter")
            except Exception:
                return None
        return None

    def _raw_inside(self, raw_env) -> bool:
        if self.target == "cab":
            return bool(OU.obj_inside_of(raw_env, "obj", "cab", partial_check=True, th=0.0))
        if self.target == "bowl":
            return bool(_obj_inside_object_aabb(raw_env, "obj", "distr_counter", partial_check=True, margin=0.002))
        return False

    def _raw_success(self, raw_env, inside_target: bool, grasped: bool, obj_speed: float) -> bool:
        if self.target == "cab":
            try:
                return bool(raw_env._check_success())
            except Exception:
                return False
        if self.target == "bowl":
            if not inside_target:
                return False
            if self.cfg.inside_require_release and grasped:
                return False
            return bool(obj_speed < self.cfg.inside_max_obj_speed)
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = self._raw_env()

        shaped_terms: dict[str, float] = {}

        # Core poses
        obj_pos = None
        eef_pos = None
        dist_eef_obj = None
        obj_speed = 0.0
        height = 0.0
        try:
            obj_pos = _obj_world_pos(raw_env, "obj")
            eef_pos = np.array(raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]], dtype=np.float64)
            dist_eef_obj = float(np.linalg.norm(eef_pos - obj_pos))
            height = float(obj_pos[2] - float(self._obj_z0))
        except Exception:
            pass
        try:
            v = np.array(raw_env.sim.data.body_xvelp[raw_env.obj_body_id["obj"]], dtype=np.float64)
            obj_speed = float(np.linalg.norm(v))
        except Exception:
            obj_speed = 0.0

        # Reach shaping
        if dist_eef_obj is not None:
            shaped_terms["reach"] = self.w_reach * float(np.exp(-self.reach_temp * dist_eef_obj))
        else:
            shaped_terms["reach"] = 0.0

        # Raw grasp + strict transportable gate
        raw_grasped = False
        grip_close = 0.0
        try:
            raw_grasped = bool(OU.check_obj_grasped(raw_env, "obj"))
        except Exception:
            raw_grasped = False
        try:
            grip_close = _get_gripper_close_fraction(raw_env, self.cfg.gripper_open_qpos)
        except Exception:
            grip_close = 0.0

        if raw_grasped:
            self._raw_grasp_streak += 1
        else:
            self._raw_grasp_streak = 0

        # Require either slight object lift OR stable finger closure + proximity so brush contact doesn't count.
        transportable = bool(
            height > self.cfg.grasp_lift_m
            or (
                self._raw_grasp_streak >= self.cfg.grasp_min_consecutive
                and dist_eef_obj is not None
                and dist_eef_obj < self.cfg.grasp_follow_dist_m
                and grip_close >= self.cfg.grasp_min_close
            )
        )
        grasped = bool(raw_grasped and transportable)

        shaped_terms["grasp"] = self.w_grasp * float(grasped)
        shaped_terms["lift"] = self.w_lift * float(np.clip(height / max(self.cfg.grasp_lift_m, 1e-6), 0.0, 1.0))

        # Carry shaping (move object toward target center)
        target_center = self._target_center(raw_env)
        if obj_pos is not None and target_center is not None:
            d = float(np.linalg.norm(obj_pos - target_center))
            shaped_terms["carry"] = self.w_carry * float(np.exp(-self.carry_temp * d))
        else:
            shaped_terms["carry"] = 0.0

        # Inside target with stability gate
        raw_inside = False
        try:
            raw_inside = bool(self._raw_inside(raw_env))
        except Exception:
            raw_inside = False

        inside_step_ok = raw_inside and (obj_speed < self.cfg.inside_max_obj_speed)
        if self.cfg.inside_require_release:
            inside_step_ok = bool(inside_step_ok and (not grasped))

        if inside_step_ok:
            self._inside_streak += 1
        else:
            self._inside_streak = 0

        inside_target = bool(self._inside_streak >= self.cfg.inside_min_consecutive)
        shaped_terms["inside"] = self.w_inside * float(inside_target)

        # Success
        success = bool(self._raw_success(raw_env, inside_target=inside_target, grasped=grasped, obj_speed=obj_speed))
        shaped_terms["success"] = self.w_success * float(success)

        dense = float(sum(shaped_terms.values()))
        info["dense_terms"] = shaped_terms
        info["dense_reward"] = dense
        info["sparse_reward"] = float(reward)
        info["success"] = bool(success)
        info["is_success"] = bool(success)
        info["strict_grasped"] = bool(grasped)
        info["raw_grasped"] = bool(raw_grasped)
        info["raw_inside"] = bool(raw_inside)
        info["inside_target"] = bool(inside_target)
        info["obj_speed"] = float(obj_speed)
        info["obj_height"] = float(height)
        info["grip_close"] = float(grip_close)
        info["raw_grasp_streak"] = int(self._raw_grasp_streak)
        info["inside_streak"] = int(self._inside_streak)

        return obs, float(reward + dense), terminated, truncated, info


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
            raise ValueError("Use PnPCounterToCab for this script.")

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
        env = Monitor(env, log_dir)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with dense + strict reward shaping on RoboCasa.")
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
        help="Torch device for PPO (auto, cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument("--gpu", action="store_true", help="Train PPO on CUDA (shortcut for --device cuda).")

    # Target location
    parser.add_argument("--target", type=str, default="cab", choices=("cab", "bowl"))

    # Dense weights (clear decomposition)
    parser.add_argument("--w_reach", type=float, default=0.25)
    parser.add_argument("--w_grasp", type=float, default=0.35)
    parser.add_argument("--w_lift", type=float, default=0.20)
    parser.add_argument("--w_carry", type=float, default=0.35)
    parser.add_argument("--w_inside", type=float, default=0.75)
    parser.add_argument("--w_success", type=float, default=5.0)
    parser.add_argument("--reach_temp", type=float, default=4.0)
    parser.add_argument("--carry_temp", type=float, default=4.0)

    # Strict grasp gates (your idea, made configurable)
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
    run_name = args.run_name or f"dense_reward_{args.target}_{timestamp}"
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
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback)
    save_path = os.path.join(run_dir, "ppo_final")
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()

