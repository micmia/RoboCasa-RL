"""Train PPO with phase-aware reward shaping for pick-and-place-into-cabinet."""

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
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linearly decay LR from initial_value to final_value over training."""
    def func(progress_remaining: float) -> float:
        return max(final_value, final_value + progress_remaining * (initial_value - final_value))
    return func


def _get_cabinet_pos(raw_env) -> np.ndarray | None:
    """Return cabinet body position, or None if unavailable."""
    try:
        for name, fixture in raw_env.fixtures.items():
            if "cab" in name.lower():
                if hasattr(fixture, "root_body"):
                    body_id = raw_env.sim.model.body_name2id(fixture.root_body)
                    return raw_env.sim.data.body_xpos[body_id].copy()
                if hasattr(fixture, "pos"):
                    return np.array(fixture.pos, dtype=np.float64)
    except Exception:
        pass
    try:
        for i in range(raw_env.sim.model.nbody):
            name = raw_env.sim.model.body_id2name(i)
            if "cab" in name.lower() and "door" not in name.lower() and "handle" not in name.lower():
                return raw_env.sim.data.body_xpos[i].copy()
    except Exception:
        pass
    return None


def _compute_grip_open(raw_env, gripper_qpos_open_is_high: bool = True) -> float | None:
    """Return gripper openness in [0,1], or None if unavailable."""
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
    """Return active gripper model for single-arm setups."""
    robot = raw_env.robots[0]
    return robot.gripper["right"] if isinstance(robot.gripper, dict) and "right" in robot.gripper else robot.gripper


def _is_obj_between_fingers(
    raw_env,
    obj_name: str = "obj",
    enclosure_margin_m: float = 0.004,
) -> bool:
    """Check if object center projects between left and right fingerpad centroids."""
    try:
        gripper = _resolve_active_gripper(raw_env)
        left_geoms = gripper.important_geoms.get("left_fingerpad", [])
        right_geoms = gripper.important_geoms.get("right_fingerpad", [])
        if not left_geoms or not right_geoms:
            return False

        left_pts = []
        for g in left_geoms:
            gid = raw_env.sim.model.geom_name2id(g)
            left_pts.append(raw_env.sim.data.geom_xpos[gid].copy())
        right_pts = []
        for g in right_geoms:
            gid = raw_env.sim.model.geom_name2id(g)
            right_pts.append(raw_env.sim.data.geom_xpos[gid].copy())
        if not left_pts or not right_pts:
            return False

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
    """Robot-agnostic grasp check with optional between-fingers enclosure."""
    try:
        obj = raw_env.objects[obj_name]
        gripper = _resolve_active_gripper(raw_env)
        # Requires both fingerpads to touch the object (more strict than generic contact).
        in_contact = bool(raw_env._check_grasp(gripper=gripper, object_geoms=obj))
    except Exception:
        return False

    if not in_contact:
        return False

    grip_open = _compute_grip_open(raw_env, gripper_qpos_open_is_high=gripper_qpos_open_is_high)
    if grip_open is None:
        # Fall back to contact-only when joint state is unavailable.
        return bool((not require_enclosure) or _is_obj_between_fingers(raw_env, obj_name=obj_name))
    grip_close = 1.0 - grip_open
    if grip_close < min_grip_close:
        return False
    if require_enclosure and (not _is_obj_between_fingers(raw_env, obj_name=obj_name)):
        return False
    return True


def _check_obj_contact(raw_env, obj_name: str = "obj") -> bool:
    """Return True if any gripper geom is in contact with the object (softer than _check_grasp)."""
    try:
        obj = raw_env.objects[obj_name]
        gripper = _resolve_active_gripper(raw_env)
        return bool(raw_env._check_grasp(gripper=gripper, object_geoms=obj))
    except Exception:
        return False


class ObservationAugmentWrapper(gym.Wrapper):
    """Append grasp-critical and task-progress features to flattened observations."""

    # rel_obj_eef xyz + gripper openness + obj_height_above_table + grasped_flag + rel_obj_cab xyz
    _EXTRA_DIM = 9

    def __init__(self, env, table_height=0.88):
        super().__init__(env)
        self.table_height = table_height
        self._cached_cab_pos = None
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
        rel_cab = np.zeros(3, dtype=np.float32)
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
            if self._cached_cab_pos is None:
                self._cached_cab_pos = _get_cabinet_pos(raw_env)
            if self._cached_cab_pos is not None and obj_pos is not None:
                rel_cab = (self._cached_cab_pos - obj_pos).astype(np.float32)
        except Exception:
            pass
        return np.array([*rel_eef, grip_open, height_above_table, grasped_flag, *rel_cab], dtype=np.float32)

    def _augment_obs(self, obs):
        return np.concatenate([obs.astype(np.float32), self._extra_features()], dtype=np.float32)

    def reset(self, **kwargs):
        self._cached_cab_pos = None
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


class AtomicRewardShapingWrapper(gym.Wrapper):
    """Phase-aware dense reward shaping: Reach → Grasp → Lift → Transport → Place.

    Anti-exploitation design:
    - Reach reward is potential-based (progress-only), not absolute proximity.
    - No per-step hold reward. A one-time grasp transition bonus + drop penalty
      encourages maintaining grasp without making it infinitely farmable.
    - Transport potential delta is capped per step to prevent instability.
    - Success reward is not curriculum-scaled so it always dominates at the end.
    - Premature lift: while not grasped, upward EEF motion is penalized unless
      the gripper is very close AND already squeezing.
    - Air-close: penalize near-closed gripper while still far from the object.
    - Strict grasp: require raw contact+closed plus transportable evidence
      (object lifted off table, or persistent tight contact over several steps).
    """

    # Strict-grasp thresholds (fixed to avoid CLI noise; tune here if needed).
    STRICT_GRASP_MIN_CONSECUTIVE = 1     # steps of raw_grasped before "transportable"
    STRICT_GRASP_LIFT_M = 0.005          # object lifted this far above table
    STRICT_GRASP_FOLLOW_DIST_M = 0.06    # eef stays within this of object
    STRICT_GRASP_MIN_CLOSE = 0.45        # grip_close in [0 open, 1 closed]

    def __init__(
        self,
        env,
        reach_reward=3.0,
        reach_abs_coef=0.25,     # weight of always-on tanh(distance) component vs potential-only
        pre_grasp_reward=0.3,    # kept small to prevent hover exploitation
        grasp_reward=3.0,
        drop_penalty=2.0,
        gripper_open_reward=0.1,
        gripper_close_reward=0.3,
        lift_reward=2.0,
        lift_threshold=0.05,
        contact_reward=0.5,       # one-time reward per episode for first gripper-object contact
        grasp_hold_reward=0.3,    # per-step reward while grasped (before lift) — incentivize maintaining grip
        lift_sustain_reward=0.5,  # per-step reward for maintaining grasp+lift
        grasp_lift_bonus=5.0,     # one-time bonus for first grasp+lift in episode
        table_height=0.88,
        transport_reward=2.0,
        place_reward=3.0,
        success_reward=10.0,
        release_reward=0.5,      # bonus for opening gripper when already inside cabinet
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
        phase_cap="full",         # "lift" = only reward up to grasp+lift; "full" = all phases
    ):
        super().__init__(env)
        self.reach_reward = reach_reward
        self.pre_grasp_reward = pre_grasp_reward
        self.grasp_reward = grasp_reward
        self.drop_penalty = drop_penalty
        self.gripper_open_reward = gripper_open_reward
        self.gripper_close_reward = gripper_close_reward
        self.reach_abs_coef = float(reach_abs_coef)
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
        self._prev_dist_eef_obj = None   # for potential-based reach
        self._prev_dist_obj_cab = None   # for potential-based transport
        self._prev_grasped = False
        self._episode_step = 0
        self._first_grasp_step = 0
        self._grasp_transitions = 0
        self._max_height = 0.0
        self._ever_contacted = False      # any gripper-object contact this episode
        self._ever_grasped = False
        self._ever_grasp_lifted = False   # grasped AND lifted above threshold
        self._ever_inside_cab = False
        self._raw_grasp_streak = 0
        self._min_dist_eef_obj = float("inf")  # closest the EEF got to object this episode
        self._cached_cab_pos = None      # cabinet position, fixed per episode
        self._prev_eef_z = None
        self._obj_z0 = None              # object z at episode start (for robust lift checks)

    def _raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def _curriculum_scale(self):
        """Linearly ramp from 0.3 → 1.0 over curriculum_steps."""
        progress = np.clip(self.global_step / self.curriculum_steps, 0.0, 1.0)
        return 0.3 + 0.7 * float(progress)

    def _cabinet_pos_cached(self, raw_env):
        if self._cached_cab_pos is None:
            self._cached_cab_pos = _get_cabinet_pos(raw_env)
        return self._cached_cab_pos

    def reset(self, **kwargs):
        self._prev_dist_eef_obj = None
        self._prev_dist_obj_cab = None
        self._prev_grasped = False
        self._episode_step = 0
        self._first_grasp_step = 0
        self._grasp_transitions = 0
        self._max_height = 0.0
        self._ever_contacted = False
        self._ever_grasped = False
        self._ever_grasp_lifted = False
        self._ever_inside_cab = False
        self._raw_grasp_streak = 0
        self._min_dist_eef_obj = float("inf")
        self._cached_cab_pos = None
        self._prev_eef_z = None
        self._obj_z0 = None
        obs, info = self.env.reset(**kwargs)
        # Capture per-episode object baseline height so lift logic is scene-agnostic.
        try:
            raw_env = self._raw_env()
            obj_pos0 = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]].copy()
            self._obj_z0 = float(obj_pos0[2])
        except Exception:
            self._obj_z0 = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.global_step += 1
        self._episode_step += 1
        scale = self._curriculum_scale()
        raw_env = self._raw_env()
        shaped = 0.0
        dist_eef_obj = None
        obj_pos = None
        eef_pos = None

        # ---------- success flag ----------
        try:
            info["success"] = bool(raw_env._check_success())
        except Exception:
            info.setdefault("success", False)

        # ---------- geometry ----------
        try:
            obj_pos = raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]].copy()
            eef_pos = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id["right"]].copy()
            dist_eef_obj = float(np.linalg.norm(eef_pos - obj_pos))
        except Exception:
            eef_pos = None

        # ---------- grasp state ----------
        raw_grasped = False
        try:
            raw_grasped = _check_obj_grasped_robust(
                raw_env,
                "obj",
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
            # Prefer relative-to-reset height; fallback to table-height if unavailable.
            if self._obj_z0 is not None:
                height = max(0.0, float(obj_pos[2]) - self._obj_z0)
            else:
                height = max(0.0, float(obj_pos[2]) - self.table_height)
            self._max_height = max(self._max_height, float(height))
            lifted = height > self.lift_threshold

        # ---------- gripper openness ----------
        grip_open = 0.5
        try:
            _grip_open = _compute_grip_open(
                raw_env,
                gripper_qpos_open_is_high=self.gripper_qpos_open_is_high,
            )
            if _grip_open is not None:
                grip_open = _grip_open
        except Exception:
            pass
        grip_close = 1.0 - grip_open

        # Strict grasp: object either slightly lifted, or gripper has held it
        # tight and followed it for several consecutive steps.
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

        # One-time contact reward: fires on the first step any gripper-object contact
        # is detected this episode. Bridges the gap between "near" and "grasped".
        in_contact = raw_grasped  # raw_grasped already requires _check_grasp contact
        if in_contact and not self._ever_contacted:
            shaped += scale * self.contact_reward
        self._ever_contacted = self._ever_contacted or bool(in_contact)

        # ============================================================
        # Phase-based reward (anti-exploitation design)
        # ============================================================
        if not grasped:
            # ---- Phase 1: Reach (potential-based + small absolute proximity) ----
            if dist_eef_obj is not None:
                if self._prev_dist_eef_obj is not None:
                    delta = self._prev_dist_eef_obj - dist_eef_obj
                    reach_progress = float(np.clip(delta * 20.0, -0.5, 1.0))
                    shaped += scale * self.reach_reward * (1.0 - self.reach_abs_coef) * reach_progress
                # Absolute component: always-on, tanh-shaped; decays with distance but never zero.
                # Prevents pure potential-based silence when policy is still near-random.
                reach_abs = float(1.0 - np.tanh(4.0 * dist_eef_obj))
                shaped += scale * self.reach_reward * self.reach_abs_coef * reach_abs
                self._prev_dist_eef_obj = dist_eef_obj
                self._prev_dist_obj_cab = None

                # ---- Phase 2: Pre-grasp proximity bonus (< 5 cm) ----
                if dist_eef_obj < 0.05:
                    proximity = 1.0 - dist_eef_obj / 0.05
                    shaped += scale * self.pre_grasp_reward * proximity

                # ---- Gripper shaping: open far, close near ----
                if dist_eef_obj >= self.gripper_open_threshold:
                    shaped += scale * self.gripper_open_reward * grip_open
                    shaped -= scale * self.gripper_open_reward * grip_close   # soft penalty for closed-far
                elif dist_eef_obj <= self.gripper_close_threshold:
                    shaped += scale * self.gripper_close_reward * grip_close
                    shaped -= scale * self.gripper_close_reward * grip_open   # soft penalty for open-near
                else:
                    alpha = (self.gripper_open_threshold - dist_eef_obj) / max(
                        1e-6, self.gripper_open_threshold - self.gripper_close_threshold
                    )
                    shaped += scale * self.gripper_open_reward * (1.0 - alpha) * grip_open
                    shaped += scale * self.gripper_close_reward * alpha * grip_close

                # ---- Penalize raising the arm before grasp ----
                # Exempt only when very close AND fingers are clearly closing.
                close_and_squeezing = (dist_eef_obj < self.premature_lift_dist_m) and (
                    grip_close >= self.premature_lift_exempt_grip_close
                )
                if eef_pos is not None and (not close_and_squeezing) and self._prev_eef_z is not None:
                    dz = float(eef_pos[2] - self._prev_eef_z)
                    if dz > 0:
                        shaped -= (
                            scale
                            * self.premature_lift_penalty
                            * self.premature_lift_z_scale
                            * dz
                        )

                # ---- Discourage closing gripper in the air before contact ----
                if grip_close > self.air_close_grip_threshold and dist_eef_obj > self.air_close_dist_m:
                    shaped -= scale * self.air_close_penalty

            # Object off the table but not grasped (scooping/accidental lift).
            if lifted:
                shaped -= scale * self.ungrasped_object_lift_penalty

            # ---- Drop penalty: was grasped last step, not now (non-scaled) ----
            if self._prev_grasped:
                shaped -= self.drop_penalty

        else:
            # ---- Phase 3+: Grasped (reach tracker no longer needed) ----
            self._prev_dist_eef_obj = None

            # One-time-per-episode grasp transition bonus (gated by _ever_grasped
            # to prevent drop-regrasp farming).
            if not self._prev_grasped:
                self._grasp_transitions += 1
                if not self._ever_grasped:
                    shaped += scale * self.grasp_reward
                    self._first_grasp_step = self._episode_step

            if not lifted:
                # ---- Phase 3: Lift (progressive toward threshold) ----
                # Per-step hold reward: incentivize keeping the grip closed.
                shaped += scale * self.grasp_hold_reward
                if obj_pos is not None:
                    shaped += scale * self.lift_reward * min(
                        1.0, height / max(1e-6, self.lift_threshold)
                    )
            else:
                # ---- Grasp+Lift achieved: one-time bonus ----
                if not self._ever_grasp_lifted:
                    shaped += scale * self.grasp_lift_bonus

                # ---- Per-step sustain reward for maintaining grasp+lift ----
                shaped += scale * self.lift_sustain_reward

                # ---- Phase 4: Transport (potential-based + dense) — skipped in lift mode ----
                if self.phase_cap != "lift":
                    cab_pos = self._cabinet_pos_cached(raw_env)
                    if cab_pos is not None and obj_pos is not None:
                        dist_obj_cab = float(np.linalg.norm(obj_pos - cab_pos))
                        transport_dense = float(1.0 - np.tanh(0.8 * dist_obj_cab))
                        shaped += scale * self.transport_reward * 0.5 * transport_dense
                        if self._prev_dist_obj_cab is not None:
                            delta_cab = self._prev_dist_obj_cab - dist_obj_cab
                            transport_progress = float(np.clip(delta_cab * 10.0, -0.5, 1.0))
                            shaped += scale * self.transport_reward * 0.5 * transport_progress
                        self._prev_dist_obj_cab = dist_obj_cab
                    else:
                        self._prev_dist_obj_cab = None

        # ---- Phase 5: Place & Success — skipped in lift mode ----
        inside_cab = False
        if self.phase_cap != "lift":
            try:
                inside_cab = bool(OU.obj_inside_of(raw_env, "obj", "cab", partial_check=True, th=0.0))
            except Exception:
                inside_cab = False
            if inside_cab and not self._ever_inside_cab:
                shaped += scale * self.place_reward
            if inside_cab and grasped:
                shaped += scale * self.release_reward * grip_open
            if info.get("success", False):
                shaped += self.success_reward
        else:
            # In lift mode treat grasp+lift as the episode success signal.
            if grasped and lifted and info.get("success", False):
                shaped += self.success_reward

        # ---- Action smoothness penalty ----
        try:
            shaped -= self.action_penalty_weight * float(np.linalg.norm(action))
        except Exception:
            pass

        if dist_eef_obj is not None:
            self._min_dist_eef_obj = min(self._min_dist_eef_obj, dist_eef_obj)
        grasp_lifted = bool(grasped and lifted)
        shaped = float(np.clip(shaped, -self.reward_clip, self.reward_clip))
        self._ever_grasped = self._ever_grasped or bool(grasped)
        self._ever_grasp_lifted = self._ever_grasp_lifted or grasp_lifted
        self._ever_inside_cab = self._ever_inside_cab or bool(inside_cab)
        info.update(
            sparse_reward=float(reward),
            dense_reward=float(shaped),
            grasped=bool(grasped),
            raw_grasped=bool(raw_grasped),
            lifted=bool(lifted),
            inside_cab=bool(inside_cab),
            first_grasp_step=int(self._first_grasp_step),
            grasp_transitions=int(self._grasp_transitions),
            max_height=float(self._max_height),
            contact_success=float(self._ever_contacted),
            grasp_success=float(self._ever_grasped),
            grasp_lift_success=float(self._ever_grasp_lifted),
            place_success=float(self._ever_inside_cab),
            min_dist_eef_obj=float(self._min_dist_eef_obj) if self._min_dist_eef_obj != float("inf") else -1.0,
        )
        self._prev_grasped = bool(grasped)
        if eef_pos is not None:
            self._prev_eef_z = float(eef_pos[2])
        return obs, float(reward + shaped), terminated, truncated, info


class MetricsLoggerCallback(BaseCallback):
    """Log training metrics to CSV on each rollout end."""

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
                "timestep",
                "ep_rew_mean",
                "ep_len_mean",
                "success_rate",
                "contact_success_rate",
                "grasp_success_rate",
                "grasp_lift_success_rate",
                "place_success_rate",
                "first_grasp_step_mean",
                "grasp_transitions_mean",
                "max_height_mean",
                "min_dist_eef_obj_mean",
                "n_episodes",
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
        self.writer.writerow(
            {
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
            }
        )
        self.csv_file.flush()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


class FreezeBaseWrapper(gym.Wrapper):
    """Zero out mobile base action dimensions so the policy only learns arm control.

    Action layout for PandaOmron (HYBRID_MOBILE_BASE):
      [0:6]  arm OSC_POSE (dx,dy,dz,drx,dry,drz)
      [6]    gripper
      [7]    torso
      [8:]   Omron base wheel velocities
    """

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
        if args.task != "PnPCounterToCab":
            raise ValueError(f"Task {args.task} is unsupported. Use PnPCounterToCab.")

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
        )

        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(
            env,
            log_dir,
            info_keywords=(
                "success",
                "contact_success",
                "grasp_success",
                "grasp_lift_success",
                "place_success",
                "first_grasp_step",
                "grasp_transitions",
                "max_height",
                "min_dist_eef_obj",
            ),
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with reward shaping on RoboCasa.")
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5, help="Final LR at end of training (linear schedule).")
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
    parser.add_argument("--no_vecnorm", action="store_true", help="Disable VecNormalize.")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200_000,
        help="Save a checkpoint every N *total* timesteps (across all envs).",
    )

    # Reward shaping
    parser.add_argument("--reach_reward", type=float, default=3.0)
    parser.add_argument("--reach_abs_coef", type=float, default=0.25,
                        help="Fraction of reach_reward applied as always-on absolute proximity (tanh-shaped).")
    parser.add_argument("--contact_reward", type=float, default=0.5,
                        help="One-time reward per episode for first gripper-object contact.")
    parser.add_argument("--grasp_hold_reward", type=float, default=0.3,
                        help="Per-step reward while grasped (before lift) to incentivize maintaining grip.")
    parser.add_argument("--pre_grasp_reward", type=float, default=0.3)
    parser.add_argument("--grasp_reward", type=float, default=3.0)
    parser.add_argument("--drop_penalty", type=float, default=2.0)
    parser.add_argument("--release_reward", type=float, default=0.5)
    parser.add_argument("--gripper_open_reward", type=float, default=0.1)
    parser.add_argument("--gripper_close_reward", type=float, default=0.3)
    parser.add_argument("--lift_reward", type=float, default=2.0)
    parser.add_argument("--lift_threshold", type=float, default=0.05)
    parser.add_argument("--lift_sustain_reward", type=float, default=0.5,
                        help="Per-step reward for maintaining grasped+lifted state.")
    parser.add_argument("--grasp_lift_bonus", type=float, default=5.0,
                        help="One-time bonus when first achieving grasped+lifted in an episode.")
    parser.add_argument("--phase_cap", type=str, default="full", choices=["full", "lift"],
                        help="'lift' focuses rewards on grasp+lift only; 'full' enables all phases.")
    parser.add_argument("--freeze_base", action="store_true", default=False,
                        help="Zero out mobile base action dims so policy only learns arm control.")
    parser.add_argument("--base_action_start", type=int, default=8,
                        help="Index where base actions begin in the action vector (default 8 for PandaOmron).")
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
    parser.add_argument(
        "--premature_lift_penalty",
        type=float,
        default=0.3,
        help="Scale for penalizing upward EEF motion while not grasped (unless close+closing; see dist/grip args).",
    )
    parser.add_argument(
        "--premature_lift_dist_m",
        type=float,
        default=0.07,
        help="With not grasped: exempt upward-z penalty only if ||eef-obj|| is below this (m) AND grip is closing.",
    )
    parser.add_argument(
        "--premature_lift_exempt_grip_close",
        type=float,
        default=0.45,
        help="With not grasped: exempt upward-z penalty only if grip_close >= this (0=open,1=closed).",
    )
    parser.add_argument(
        "--premature_lift_z_scale",
        type=float,
        default=8.0,
        help="Multiplies dz (m/step) with premature_lift_penalty; increase if the arm still rises before grasping.",
    )
    parser.add_argument(
        "--ungrasped_object_lift_penalty",
        type=float,
        default=0.1,
        help="Per-step penalty (curriculum-scaled) when object is lifted off table but grasp is still false.",
    )
    parser.add_argument(
        "--air_close_grip_threshold",
        type=float,
        default=0.70,
        help="If not grasped and grip_close exceeds this while farther than air_close_dist_m, apply air_close_penalty.",
    )
    parser.add_argument(
        "--air_close_dist_m",
        type=float,
        default=0.10,
        help="Minimum eef-obj distance (m) before air-close penalty applies.",
    )
    parser.add_argument(
        "--air_close_penalty",
        type=float,
        default=0.05,
        help="Per-step penalty (curriculum-scaled) for closing gripper before contact.",
    )

    args = parser.parse_args()

    train_device = args.device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_reward_shaping_v2_{timestamp}"
    run_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_root = args.log_root if args.log_root is not None else os.path.join(run_dir, "logs")
    tensorboard_dir = os.path.join(log_root, "tensorboard")
    monitor_root = os.path.join(log_root, "monitor")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(monitor_root, exist_ok=True)

    env_fns = [make_env(args, i, monitor_root) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    if not args.no_vecnorm:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma
        )

    lr_schedule = linear_schedule(args.learning_rate, args.lr_final)

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
        device=train_device,
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
