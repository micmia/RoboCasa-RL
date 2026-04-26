"""Train PPO with phase-aware reward shaping for pick-apple-and-place-into-bowl."""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Callable

import gymnasium as gym
import numpy as np
from robocasa.utils import object_utils as OU
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_apple_to_bowl import MyPnPAppleToBowl


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return max(final_value, final_value + progress_remaining * (initial_value - final_value))
    return func


def _get_bowl_pos(raw_env) -> np.ndarray | None:
    """Return bowl (container) body position, or None if unavailable."""
    try:
        body_id = raw_env.obj_body_id["container"]
        return raw_env.sim.data.body_xpos[body_id].copy()
    except Exception:
        return None


def _compute_grip_open(raw_env, gripper_qpos_open_is_high: bool = True) -> float | None:
    try:
        gripper_qpos = raw_env.sim.data.qpos[raw_env.robots[0].gripper.qpos_indices]
        grip_raw = float(np.mean(gripper_qpos))
        grip_open = float(0.5 + 0.5 * np.tanh(5.0 * grip_raw))
        if not gripper_qpos_open_is_high:
            grip_open = 1.0 - grip_open
        return float(np.clip(grip_open, 0.0, 1.0))
    except Exception:
        return None


def _resolve_active_gripper(raw_env):
    robot = raw_env.robots[0]
    return robot.gripper["right"] if isinstance(robot.gripper, dict) and "right" in robot.gripper else robot.gripper


def _is_obj_between_fingers(raw_env, obj_name: str = "obj", enclosure_margin_m: float = 0.004) -> bool:
    try:
        gripper = _resolve_active_gripper(raw_env)
        left_geoms = gripper.important_geoms.get("left_fingerpad", [])
        right_geoms = gripper.important_geoms.get("right_fingerpad", [])
        if not left_geoms or not right_geoms:
            return False
        left_pts = [raw_env.sim.data.geom_xpos[raw_env.sim.model.geom_name2id(g)].copy() for g in left_geoms]
        right_pts = [raw_env.sim.data.geom_xpos[raw_env.sim.model.geom_name2id(g)].copy() for g in right_geoms]
        left_c = np.mean(np.asarray(left_pts), axis=0)
        right_c = np.mean(np.asarray(right_pts), axis=0)
        axis = right_c - left_c
        span = float(np.linalg.norm(axis))
        if span < 1e-6:
            return False
        axis_u = axis / span
        obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id[obj_name]].copy()
        t = float(np.dot(obj_pos - left_c, axis_u))
        return (-enclosure_margin_m) <= t <= (span + enclosure_margin_m)
    except Exception:
        return False


def _check_obj_grasped_robust(
    raw_env,
    obj_name: str = "obj",
    gripper_qpos_open_is_high: bool = True,
    min_grip_close: float = 0.55,
    require_enclosure: bool = True,
) -> bool:
    try:
        obj = raw_env.objects[obj_name]
        gripper = _resolve_active_gripper(raw_env)
        in_contact = bool(raw_env._check_grasp(gripper=gripper, object_geoms=obj))
    except Exception:
        return False
    if not in_contact:
        return False
    grip_open = _compute_grip_open(raw_env, gripper_qpos_open_is_high=gripper_qpos_open_is_high)
    if grip_open is None:
        return bool((not require_enclosure) or _is_obj_between_fingers(raw_env, obj_name=obj_name))
    grip_close = 1.0 - grip_open
    if grip_close < min_grip_close:
        return False
    if require_enclosure and (not _is_obj_between_fingers(raw_env, obj_name=obj_name)):
        return False
    return True


class ObservationAugmentWrapper(gym.Wrapper):
    """Append grasp-critical and task-progress features to flattened observations.

    Extra features (9-dim):
      rel_obj_eef xyz | gripper openness | obj_height_above_table | grasped_flag | rel_obj_bowl xyz
    """

    _EXTRA_DIM = 9

    def __init__(self, env, table_height=0.88):
        super().__init__(env)
        self.table_height = table_height
        self._cached_bowl_pos = None
        base_space = env.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("ObservationAugmentWrapper expects Box observation space.")
        low = np.concatenate([base_space.low, -np.inf * np.ones(self._EXTRA_DIM, dtype=base_space.dtype)])
        high = np.concatenate([base_space.high, np.inf * np.ones(self._EXTRA_DIM, dtype=base_space.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=base_space.dtype)

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def _extra_features(self):
        raw_env = self._raw_env()
        rel_eef = np.zeros(3, dtype=np.float32)
        grip_open = np.float32(0.5)
        height_above_table = np.float32(0.0)
        grasped_flag = np.float32(0.0)
        rel_bowl = np.zeros(3, dtype=np.float32)
        obj_pos = None
        try:
            obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]]
            eef_pos = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]]
            rel_eef = (obj_pos - eef_pos).astype(np.float32)
            height_above_table = np.float32(max(0.0, float(obj_pos[2]) - self.table_height))
        except Exception:
            pass
        try:
            _grip_open = _compute_grip_open(raw_env, gripper_qpos_open_is_high=True)
            if _grip_open is not None:
                grip_open = np.float32(_grip_open)
        except Exception:
            pass
        try:
            grasped_flag = np.float32(1.0 if _check_obj_grasped_robust(raw_env, "obj") else 0.0)
        except Exception:
            pass
        try:
            if self._cached_bowl_pos is None:
                self._cached_bowl_pos = _get_bowl_pos(raw_env)
            if self._cached_bowl_pos is not None and obj_pos is not None:
                rel_bowl = (self._cached_bowl_pos - obj_pos).astype(np.float32)
        except Exception:
            pass
        return np.array([*rel_eef, grip_open, height_above_table, grasped_flag, *rel_bowl], dtype=np.float32)

    def _augment_obs(self, obs):
        return np.concatenate([obs.astype(np.float32), self._extra_features()], dtype=np.float32)

    def reset(self, **kwargs):
        self._cached_bowl_pos = None
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


class AtomicRewardShapingWrapper(gym.Wrapper):
    """Phase-aware dense reward shaping: Reach → Grasp → Lift → Transport → Place into bowl.

    Each step returns sparse env reward plus dense shaping (curriculum-scaled, clipped),
    with phase_cap optionally limiting which terms apply (e.g. lift-only pretraining).
    """

    STRICT_GRASP_MIN_CONSECUTIVE = 1
    STRICT_GRASP_LIFT_M = 0.005
    STRICT_GRASP_FOLLOW_DIST_M = 0.06
    STRICT_GRASP_MIN_CLOSE = 0.45

    def __init__(
        self,
        env,
        reach_reward=3.0,
        reach_abs_coef=0.25,
        pre_grasp_reward=0.3,
        grasp_reward=3.0,
        drop_penalty=2.0,
        gripper_open_reward=0.1,
        gripper_close_reward=0.3,
        lift_reward=2.0,
        lift_threshold=0.05,
        contact_reward=0.5,
        grasp_hold_reward=0.3,
        lift_sustain_reward=0.5,
        grasp_lift_bonus=5.0,
        table_height=0.88,
        transport_reward=2.0,
        place_reward=3.0,
        success_reward=10.0,
        release_reward=0.5,
        action_penalty_weight=0.005,
        gripper_open_threshold=0.14,
        gripper_close_threshold=0.06,
        gripper_qpos_open_is_high=True,
        curriculum_steps=150_000,
        reward_clip=15.0,
        premature_lift_penalty=0.3,
        premature_lift_dist_m=0.07,
        premature_lift_exempt_grip_close=0.45,
        premature_lift_z_scale=8.0,
        ungrasped_object_lift_penalty=0.1,
        air_close_grip_threshold=0.70,
        air_close_dist_m=0.10,
        air_close_penalty=0.05,
        phase_cap="full",
        retraction_reward=3.0,
        retraction_dist_m=0.05,
    ):
        super().__init__(env)
        self.reach_reward = reach_reward
        self.reach_abs_coef = float(reach_abs_coef)
        self.retraction_reward = float(retraction_reward)
        self.retraction_dist_m = float(retraction_dist_m)
        self.pre_grasp_reward = pre_grasp_reward
        self.grasp_reward = grasp_reward
        self.drop_penalty = drop_penalty
        self.gripper_open_reward = gripper_open_reward
        self.gripper_close_reward = gripper_close_reward
        self.contact_reward = float(contact_reward)
        self.grasp_hold_reward = float(grasp_hold_reward)
        self.lift_reward = lift_reward
        self.lift_threshold = lift_threshold
        self.lift_sustain_reward = float(lift_sustain_reward)
        self.grasp_lift_bonus = float(grasp_lift_bonus)
        self.table_height = table_height
        self.transport_reward = transport_reward
        self.place_reward = place_reward
        self.success_reward = success_reward
        self.release_reward = release_reward
        self.action_penalty_weight = action_penalty_weight
        self.gripper_open_threshold = gripper_open_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_qpos_open_is_high = gripper_qpos_open_is_high
        self.curriculum_steps = max(1, int(curriculum_steps))
        self.reward_clip = float(reward_clip)
        self.premature_lift_penalty = float(premature_lift_penalty)
        self.premature_lift_dist_m = float(premature_lift_dist_m)
        self.premature_lift_exempt_grip_close = float(premature_lift_exempt_grip_close)
        self.premature_lift_z_scale = float(premature_lift_z_scale)
        self.ungrasped_object_lift_penalty = float(ungrasped_object_lift_penalty)
        self.air_close_grip_threshold = float(air_close_grip_threshold)
        self.air_close_dist_m = float(air_close_dist_m)
        self.air_close_penalty = float(air_close_penalty)
        self.phase_cap = str(phase_cap).lower()
        self.global_step = 0
        self._prev_dist_eef_obj = None
        self._prev_dist_obj_bowl = None
        self._prev_grasped = False
        self._episode_step = 0
        self._first_grasp_step = 0
        self._grasp_transitions = 0
        self._max_height = 0.0
        self._ever_contacted = False
        self._ever_grasped = False
        self._ever_grasp_lifted = False
        self._ever_inside_bowl = False
        self._raw_grasp_streak = 0
        self._min_dist_eef_obj = float("inf")
        self._cached_bowl_pos = None
        self._prev_eef_z = None
        self._obj_z0 = None
        self._init_inside_bowl = False

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def _curriculum_scale(self):
        progress = np.clip(self.global_step / self.curriculum_steps, 0.0, 1.0)
        return 0.3 + 0.7 * float(progress)

    def _bowl_pos_cached(self, raw_env):
        if self._cached_bowl_pos is None:
            self._cached_bowl_pos = _get_bowl_pos(raw_env)
        return self._cached_bowl_pos

    def reset(self, **kwargs):
        self._prev_dist_eef_obj = None
        self._prev_dist_obj_bowl = None
        self._prev_grasped = False
        self._episode_step = 0
        self._first_grasp_step = 0
        self._grasp_transitions = 0
        self._max_height = 0.0
        self._ever_contacted = False
        self._ever_grasped = False
        self._ever_grasp_lifted = False
        self._ever_inside_bowl = False
        self._raw_grasp_streak = 0
        self._min_dist_eef_obj = float("inf")
        self._cached_bowl_pos = None
        self._prev_eef_z = None
        self._obj_z0 = None
        self._init_inside_bowl = False
        obs, info = self.env.reset(**kwargs)
        try:
            raw_env = self._raw_env()
            obj_pos0 = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]].copy()
            self._obj_z0 = float(obj_pos0[2])
        except Exception:
            self._obj_z0 = None
        try:
            self._init_inside_bowl = bool(OU.check_obj_in_receptacle(self._raw_env(), "obj", "container"))
        except Exception:
            self._init_inside_bowl = False
        return obs, info

    def step(self, action):
        # Env sparse task reward + dense shaped term (curriculum-scaled). Returned reward is reward + shaped.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.global_step += 1
        self._episode_step += 1
        # Ramps dense shaping from 0.3→1.0 over curriculum_steps so early training is not dominated by shaping.
        scale = self._curriculum_scale()
        raw_env = self._raw_env()
        shaped = 0.0
        dist_eef_obj = None
        obj_pos = None
        eef_pos = None

        try:
            info["success"] = bool(raw_env._check_success())
        except Exception:
            info.setdefault("success", False)

        try:
            obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]].copy()
            eef_pos = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]].copy()
            dist_eef_obj = float(np.linalg.norm(eef_pos - obj_pos))
        except Exception:
            eef_pos = None

        raw_grasped = False
        try:
            raw_grasped = _check_obj_grasped_robust(
                raw_env, "obj",
                gripper_qpos_open_is_high=self.gripper_qpos_open_is_high,
                min_grip_close=0.45,
                require_enclosure=False,
            )
        except Exception:
            pass
        self._raw_grasp_streak = self._raw_grasp_streak + 1 if raw_grasped else 0

        lifted = False
        height = 0.0
        if obj_pos is not None:
            if self._obj_z0 is not None:
                height = max(0.0, float(obj_pos[2]) - self._obj_z0)
            else:
                height = max(0.0, float(obj_pos[2]) - self.table_height)
            self._max_height = max(self._max_height, float(height))
            lifted = height > self.lift_threshold

        grip_open = 0.5
        try:
            _grip_open = _compute_grip_open(raw_env, gripper_qpos_open_is_high=self.gripper_qpos_open_is_high)
            if _grip_open is not None:
                grip_open = _grip_open
        except Exception:
            pass
        grip_close = 1.0 - grip_open

        # Require either slight object lift or stable finger closure + proximity so "brush contact" does not count as grasp.
        transportable = (
            height > self.STRICT_GRASP_LIFT_M
            or (
                self._raw_grasp_streak >= self.STRICT_GRASP_MIN_CONSECUTIVE
                and dist_eef_obj is not None
                and dist_eef_obj < self.STRICT_GRASP_FOLLOW_DIST_M
                and grip_close >= self.STRICT_GRASP_MIN_CLOSE
            )
        )
        grasped = bool(raw_grasped and transportable)

        # Compute inside_bowl early so drop_penalty can be suppressed for intentional release.
        inside_bowl = False
        if self.phase_cap != "lift":
            try:
                inside_bowl = bool(OU.check_obj_in_receptacle(raw_env, "obj", "container"))
            except Exception:
                inside_bowl = False

        in_contact = raw_grasped
        if in_contact and not self._ever_contacted:
            shaped += scale * self.contact_reward
        self._ever_contacted = self._ever_contacted or bool(in_contact)

        # --- Not grasped: reach / pre-grasp / gripper schedule, anti-cheese (premature lift, air-close), drop vs release ---
        if not grasped:
            if dist_eef_obj is not None:
                if self._prev_dist_eef_obj is not None:
                    delta = self._prev_dist_eef_obj - dist_eef_obj
                    reach_progress = float(np.clip(delta * 20.0, -0.5, 1.0))
                    shaped += scale * self.reach_reward * (1.0 - self.reach_abs_coef) * reach_progress
                reach_abs = float(1.0 - np.tanh(4.0 * dist_eef_obj))
                shaped += scale * self.reach_reward * self.reach_abs_coef * reach_abs
                self._prev_dist_eef_obj = dist_eef_obj
                self._prev_dist_obj_bowl = None

                if dist_eef_obj < 0.05:
                    proximity = 1.0 - dist_eef_obj / 0.05
                    shaped += scale * self.pre_grasp_reward * proximity

                if dist_eef_obj >= self.gripper_open_threshold:
                    shaped += scale * self.gripper_open_reward * grip_open
                    shaped -= scale * self.gripper_open_reward * grip_close
                elif dist_eef_obj <= self.gripper_close_threshold:
                    shaped += scale * self.gripper_close_reward * grip_close
                    shaped -= scale * self.gripper_close_reward * grip_open
                else:
                    alpha = (self.gripper_open_threshold - dist_eef_obj) / max(
                        1e-6, self.gripper_open_threshold - self.gripper_close_threshold
                    )
                    shaped += scale * self.gripper_open_reward * (1.0 - alpha) * grip_open
                    shaped += scale * self.gripper_close_reward * alpha * grip_close

                close_and_squeezing = (dist_eef_obj < self.premature_lift_dist_m) and (
                    grip_close >= self.premature_lift_exempt_grip_close
                )
                if eef_pos is not None and (not close_and_squeezing) and self._prev_eef_z is not None:
                    dz = float(eef_pos[2] - self._prev_eef_z)
                    if dz > 0:
                        shaped -= scale * self.premature_lift_penalty * self.premature_lift_z_scale * dz

                if grip_close > self.air_close_grip_threshold and dist_eef_obj > self.air_close_dist_m:
                    shaped -= scale * self.air_close_penalty

            if lifted:
                shaped -= scale * self.ungrasped_object_lift_penalty

            # Only penalise drops that are NOT intentional bowl placements.
            # _ever_inside_bowl catches the case where the apple was placed then fell out.
            if self._prev_grasped and not inside_bowl and not self._ever_inside_bowl:
                shaped -= self.drop_penalty

            # Sustained reward for apple resting in bowl with gripper open (post-release).
            if inside_bowl and self._ever_contacted:
                shaped += scale * self.release_reward * grip_open
                # Reward retracting the arm away so gripper_obj_far can trigger success.
                if dist_eef_obj is not None:
                    retraction = float(np.tanh(4.0 * max(0.0, dist_eef_obj - self.retraction_dist_m)))
                    shaped += scale * self.retraction_reward * retraction

        # --- Grasped: first-grasp bonus, optional transport shaping, lift/hold/sustain, release-in-bowl ---
        else:
            self._prev_dist_eef_obj = None

            if not self._prev_grasped:
                self._grasp_transitions += 1
                if not self._ever_grasped:
                    shaped += scale * self.grasp_reward
                    self._first_grasp_step = self._episode_step

            if self.phase_cap != "lift" and lifted:
                # Transport reward only when the apple is lifted — prevents a "drag along counter"
                # local optimum and preserves the lift-then-transport skill from pretraining.
                bowl_pos = self._bowl_pos_cached(raw_env)
                if bowl_pos is not None and obj_pos is not None:
                    dist_obj_bowl = float(np.linalg.norm(obj_pos - bowl_pos))
                    transport_dense = float(1.0 - np.tanh(0.8 * dist_obj_bowl))
                    shaped += scale * self.transport_reward * 0.5 * transport_dense
                    if self._prev_dist_obj_bowl is not None:
                        delta_bowl = self._prev_dist_obj_bowl - dist_obj_bowl
                        transport_progress = float(np.clip(delta_bowl * 10.0, -0.5, 1.0))
                        shaped += scale * self.transport_reward * 0.5 * transport_progress
                    self._prev_dist_obj_bowl = dist_obj_bowl
                else:
                    self._prev_dist_obj_bowl = None

            if not lifted:
                shaped += scale * self.grasp_hold_reward
                if obj_pos is not None:
                    shaped += scale * self.lift_reward * min(1.0, height / max(1e-6, self.lift_threshold))
            else:
                if not self._ever_grasp_lifted:
                    shaped += scale * self.grasp_lift_bonus
                shaped += scale * self.lift_sustain_reward

            # Encourage gripper release once apple is inside the bowl.
            if inside_bowl:
                shaped += scale * self.release_reward * grip_open

        # --- Placement & success: full task gates place/success on contact; lift-only cap uses grasp+lift+env success ---
        if self.phase_cap != "lift":
            # Only credit placement if the agent actually contacted the object,
            # filtering out episodes where the apple spawned inside the bowl.
            agent_placed = inside_bowl and self._ever_contacted
            if agent_placed and not self._ever_inside_bowl:
                shaped += scale * self.place_reward
            env_success = info.get("success", False) and self._ever_contacted
            info["success"] = env_success
            if env_success:
                shaped += self.success_reward
        else:
            env_success = info.get("success", False) and self._ever_contacted
            info["success"] = env_success
            if grasped and lifted and env_success:
                shaped += self.success_reward

        # L2 action norm: discourages jitter; subtracted after task terms so success spike still dominates when clip allows.
        try:
            shaped -= self.action_penalty_weight * float(np.linalg.norm(action))
        except Exception:
            pass

        if dist_eef_obj is not None:
            self._min_dist_eef_obj = min(self._min_dist_eef_obj, dist_eef_obj)
        # Episode-level metric: stable "grasped while lifted" (used in info / callbacks, not added again as reward here).
        grasp_lifted = bool(grasped and lifted)
        shaped = float(np.clip(shaped, -self.reward_clip, self.reward_clip))
        self._ever_grasped = self._ever_grasped or bool(grasped)
        self._ever_grasp_lifted = self._ever_grasp_lifted or grasp_lifted
        self._ever_inside_bowl = self._ever_inside_bowl or (inside_bowl and self._ever_contacted)
        # Logging: sparse vs dense split; flags mirror curriculum success criteria for analysis.
        info.update(
            sparse_reward=float(reward),
            dense_reward=float(shaped),
            grasped=bool(grasped),
            raw_grasped=bool(raw_grasped),
            lifted=bool(lifted),
            inside_bowl=bool(inside_bowl),
            first_grasp_step=int(self._first_grasp_step),
            grasp_transitions=int(self._grasp_transitions),
            max_height=float(self._max_height),
            contact_success=float(self._ever_contacted),
            grasp_success=float(self._ever_grasped),
            grasp_lift_success=float(self._ever_grasp_lifted),
            place_success=float(self._ever_inside_bowl),
            min_dist_eef_obj=float(self._min_dist_eef_obj) if self._min_dist_eef_obj != float("inf") else -1.0,
        )
        self._prev_grasped = bool(grasped)
        if eef_pos is not None:
            self._prev_eef_z = float(eef_pos[2])
        # Terminate the episode immediately on success so PPO gets a clean credit assignment.
        if info.get("success", False):
            terminated = True
        return obs, float(reward + shaped), terminated, truncated, info


class MetricsLoggerCallback(BaseCallback):
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
            fieldnames=[
                "timestep", "ep_rew_mean", "ep_len_mean", "success_rate",
                "contact_success_rate", "grasp_success_rate", "grasp_lift_success_rate",
                "place_success_rate", "first_grasp_step_mean", "grasp_transitions_mean",
                "max_height_mean", "min_dist_eef_obj_mean", "n_episodes",
            ],
        )
        self.writer.writeheader()

    def _on_rollout_end(self):
        if not self.model.ep_info_buffer:
            return
        ep_rews = [ep["r"] for ep in self.model.ep_info_buffer]
        ep_lens = [ep["l"] for ep in self.model.ep_info_buffer]
        successes = [ep.get("success", ep.get("is_success", 0)) for ep in self.model.ep_info_buffer]
        contact_successes = [ep.get("contact_success", 0.0) for ep in self.model.ep_info_buffer]
        grasp_successes = [ep.get("grasp_success", 0.0) for ep in self.model.ep_info_buffer]
        grasp_lift_successes = [ep.get("grasp_lift_success", 0.0) for ep in self.model.ep_info_buffer]
        place_successes = [ep.get("place_success", 0.0) for ep in self.model.ep_info_buffer]
        first_grasp_steps = [ep.get("first_grasp_step", 0) for ep in self.model.ep_info_buffer]
        grasp_transitions = [ep.get("grasp_transitions", 0) for ep in self.model.ep_info_buffer]
        max_heights = [ep.get("max_height", 0.0) for ep in self.model.ep_info_buffer]
        min_dists = [ep.get("min_dist_eef_obj", -1.0) for ep in self.model.ep_info_buffer]
        valid_dists = [d for d in min_dists if d >= 0]
        self.writer.writerow({
            "timestep": self.num_timesteps,
            "ep_rew_mean": float(np.mean(ep_rews)),
            "ep_len_mean": float(np.mean(ep_lens)),
            "success_rate": float(np.mean(successes)),
            "contact_success_rate": float(np.mean(contact_successes)),
            "grasp_success_rate": float(np.mean(grasp_successes)),
            "grasp_lift_success_rate": float(np.mean(grasp_lift_successes)),
            "place_success_rate": float(np.mean(place_successes)),
            "first_grasp_step_mean": float(np.mean(first_grasp_steps)),
            "grasp_transitions_mean": float(np.mean(grasp_transitions)),
            "max_height_mean": float(np.mean(max_heights)),
            "min_dist_eef_obj_mean": float(np.mean(valid_dists)) if valid_dists else -1.0,
            "n_episodes": len(ep_rews),
        })
        self.csv_file.flush()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


class FreezeBaseWrapper(gym.Wrapper):
    def __init__(self, env, base_action_start: int = 8):
        super().__init__(env)
        self.base_action_start = base_action_start

    def step(self, action):
        if self.base_action_start < len(action):
            action = np.array(action, dtype=np.float32)
            action[self.base_action_start:] = 0.0
        return self.env.step(action)


def make_env(args, rank, monitor_root):
    def _init():
        robots = "PandaOmron"
        controller_config = load_composite_controller_config(controller=None, robot=robots)
        env = MyPnPAppleToBowl(
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
        if args.freeze_base:
            env = FreezeBaseWrapper(env, base_action_start=args.base_action_start)
        env = ObservationAugmentWrapper(env, table_height=args.table_height)
        env = AtomicRewardShapingWrapper(
            env,
            reach_reward=args.reach_reward,
            reach_abs_coef=args.reach_abs_coef,
            pre_grasp_reward=args.pre_grasp_reward,
            contact_reward=args.contact_reward,
            grasp_hold_reward=args.grasp_hold_reward,
            grasp_reward=args.grasp_reward,
            drop_penalty=args.drop_penalty,
            gripper_open_reward=args.gripper_open_reward,
            gripper_close_reward=args.gripper_close_reward,
            lift_reward=args.lift_reward,
            lift_threshold=args.lift_threshold,
            lift_sustain_reward=args.lift_sustain_reward,
            grasp_lift_bonus=args.grasp_lift_bonus,
            table_height=args.table_height,
            transport_reward=args.transport_reward,
            place_reward=args.place_reward,
            success_reward=args.success_reward,
            release_reward=args.release_reward,
            action_penalty_weight=args.action_penalty_weight,
            gripper_open_threshold=args.gripper_open_threshold,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_qpos_open_is_high=args.gripper_qpos_open_is_high,
            curriculum_steps=args.curriculum_steps,
            reward_clip=args.reward_clip,
            premature_lift_penalty=args.premature_lift_penalty,
            premature_lift_dist_m=args.premature_lift_dist_m,
            premature_lift_exempt_grip_close=args.premature_lift_exempt_grip_close,
            premature_lift_z_scale=args.premature_lift_z_scale,
            ungrasped_object_lift_penalty=args.ungrasped_object_lift_penalty,
            air_close_grip_threshold=args.air_close_grip_threshold,
            air_close_dist_m=args.air_close_dist_m,
            air_close_penalty=args.air_close_penalty,
            phase_cap=args.phase_cap,
            retraction_reward=args.retraction_reward,
            retraction_dist_m=args.retraction_dist_m,
        )
        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(
            env, log_dir,
            info_keywords=(
                "success", "contact_success", "grasp_success", "grasp_lift_success",
                "place_success", "first_grasp_step", "grasp_transitions",
                "max_height", "min_dist_eef_obj",
            ),
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO to pick apple and place into bowl.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_vecnorm", action="store_true")
    parser.add_argument("--checkpoint_freq", type=int, default=200_000)
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a .zip model checkpoint to fine-tune from (e.g. v2 final model).")
    parser.add_argument("--load_vecnorm", type=str, default=None,
                        help="Path to a VecNormalize .pkl to initialise normalisation stats from.")

    # Reward shaping
    parser.add_argument("--reach_reward", type=float, default=3.0)
    parser.add_argument("--reach_abs_coef", type=float, default=0.25)
    parser.add_argument("--contact_reward", type=float, default=0.5)
    parser.add_argument("--grasp_hold_reward", type=float, default=0.3)
    parser.add_argument("--pre_grasp_reward", type=float, default=0.3)
    parser.add_argument("--grasp_reward", type=float, default=3.0)
    parser.add_argument("--drop_penalty", type=float, default=2.0)
    parser.add_argument("--release_reward", type=float, default=0.5)
    parser.add_argument("--gripper_open_reward", type=float, default=0.1)
    parser.add_argument("--gripper_close_reward", type=float, default=0.3)
    parser.add_argument("--lift_reward", type=float, default=2.0)
    parser.add_argument("--lift_threshold", type=float, default=0.05)
    parser.add_argument("--lift_sustain_reward", type=float, default=0.5)
    parser.add_argument("--grasp_lift_bonus", type=float, default=5.0)
    parser.add_argument("--phase_cap", type=str, default="full", choices=["full", "lift"])
    parser.add_argument("--freeze_base", action="store_true", default=False)
    parser.add_argument("--base_action_start", type=int, default=8)
    parser.add_argument("--transport_reward", type=float, default=2.0)
    parser.add_argument("--place_reward", type=float, default=3.0)
    parser.add_argument("--success_reward", type=float, default=10.0)
    parser.add_argument("--action_penalty_weight", type=float, default=0.005)
    parser.add_argument("--table_height", type=float, default=0.88)
    parser.add_argument("--gripper_open_threshold", type=float, default=0.14)
    parser.add_argument("--gripper_close_threshold", type=float, default=0.06)
    parser.add_argument("--gripper_qpos_open_is_high", action="store_true", default=True)
    parser.add_argument("--gripper_qpos_open_is_low", action="store_false", dest="gripper_qpos_open_is_high")
    parser.add_argument("--curriculum_steps", type=int, default=150_000)
    parser.add_argument("--reward_clip", type=float, default=15.0)
    parser.add_argument("--premature_lift_penalty", type=float, default=0.3)
    parser.add_argument("--premature_lift_dist_m", type=float, default=0.07)
    parser.add_argument("--premature_lift_exempt_grip_close", type=float, default=0.45)
    parser.add_argument("--premature_lift_z_scale", type=float, default=8.0)
    parser.add_argument("--ungrasped_object_lift_penalty", type=float, default=0.1)
    parser.add_argument("--air_close_grip_threshold", type=float, default=0.70)
    parser.add_argument("--air_close_dist_m", type=float, default=0.10)
    parser.add_argument("--air_close_penalty", type=float, default=0.05)
    parser.add_argument("--retraction_reward", type=float, default=3.0)
    parser.add_argument("--retraction_dist_m", type=float, default=0.05)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_reward_shaping_v3_{timestamp}"
    run_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_root = args.log_root if args.log_root is not None else os.path.join(run_dir, "logs")
    tensorboard_dir = os.path.join(log_root, "tensorboard")
    monitor_root = os.path.join(log_root, "monitor")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(monitor_root, exist_ok=True)

    env_fns = [make_env(args, i, monitor_root) for i in range(args.n_envs)]
    base_vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    if not args.no_vecnorm:
        if args.load_vecnorm:
            vec_env = VecNormalize.load(args.load_vecnorm, base_vec_env)
            vec_env.training = True
            vec_env.norm_obs = True
            vec_env.norm_reward = True
            vec_env.clip_obs = 10.0
            print(f"Loaded VecNormalize stats from: {args.load_vecnorm}")
        else:
            vec_env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
    else:
        vec_env = base_vec_env

    lr_schedule = linear_schedule(args.learning_rate, args.lr_final)

    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=vec_env,
            custom_objects={
                "learning_rate": lr_schedule,
                "ent_coef": args.ent_coef,
                "clip_range": args.clip_range,
                "n_epochs": args.n_epochs,
                "max_grad_norm": args.max_grad_norm,
                "tensorboard_log": tensorboard_dir,
            },
            device=args.device,
            verbose=1,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
            device=args.device,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_ckpt",
        save_vecnormalize=not args.no_vecnorm,
        verbose=1,
    )
    metrics_cb = MetricsLoggerCallback(log_path=os.path.join(log_root, "metrics.csv"))
    callback = CallbackList([checkpoint_cb, metrics_cb])

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback)

    save_path = os.path.join(run_dir, "ppo_final")
    model.save(save_path)
    if not args.no_vecnorm:
        vecnorm_path = os.path.join(run_dir, "vec_normalize.pkl")
        vec_env.save(vecnorm_path)
        print(f"Saved VecNormalize stats to: {vecnorm_path}")
    vec_env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
