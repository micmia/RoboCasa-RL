"""SAC + HER for the bowl pick-and-place task.

The previous attempt with pure SAC (sac_FINAL_170k) hit a local optimum where
the robot reaches but doesn't lift. HER (Hindsight Experience Replay) solves
the chicken-and-egg by relabeling failed trajectories as successes for nearby
goals — providing learning signal even before the policy can solve the task.

Key design:
  - Sparse reward: -1 if not in goal zone, 0 if in goal zone
  - Goal: bowl interior (target_z = 1.10m, just above rim)
  - achieved_goal: apple position
  - HER strategy: 'future' (relabel with future achieved_goals from same episode)

Usage:
  .venv/bin/python scripts/train_sac_her_bowl.py --headless --total_timesteps 200000
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


# Distance threshold for "success" (apple within this distance of bowl target).
# This is a CURRICULUM threshold — the actual threshold used in compute_reward
# can be widened early in training (see THRESHOLD_SCHEDULE) and tightened later.
SUCCESS_THRESHOLD = 0.10  # 10cm — final threshold (3D, used in 1B/1C)
SUCCESS_THRESHOLD_XY = 0.15  # 15cm — looser for 1A bootstrap (helps HER fire more often)

# Phase 1A vs 1B split (user's insight: avoid policy that "hovers" without releasing).
# - 1A (approach):  reward based on XY distance only. Apple can be at any height,
#                    we just want it OVER the bowl horizontally. Easy bootstrap.
# - 1B+ (place):    reward based on full 4D distance. Apple MUST be low (is_low=1)
#                    AND xy-aligned. Forces actual descent (release encouraged).
# - Phases 2, 3 use 1B+ logic (still 4D, hardest setups).
CURRENT_REWARD_PHASE = "1A"  # mutable, updated by PhaseCallback

# Goal enrichment: achieved_goal and desired_goal are 4D = [x, y, z, is_low].
# is_low ∈ [0, 1] is a smooth indicator that apple_z is in the bowl region.
# The ramp is calibrated so that is_low(bowl_target_z) = 1.0 EXACTLY, and
# is_low(pre_grasp_z) = 0. HER relabels with apple in air (high z) get
# is_low≈0; the desired_goal always has is_low=1.0; so the L2 distance
# between achieved (in air) and desired (low) is large → reward = -1, and
# the relabel is mathematically NOT a success. This kills the "fly up"
# failure mode caused by HER's biased achieved_goal distribution.
LOW_Z_TOP = 1.30    # apple_z >= here → is_low = 0
LOW_Z_BOT = 1.00    # apple_z <= here → is_low = 1 (widened: smoother gradient)

def compute_is_low(apple_z):
    """Smooth indicator in [0, 1]. =1 at bowl level, =0 in the air."""
    return float(np.clip((LOW_Z_TOP - apple_z) / (LOW_Z_TOP - LOW_Z_BOT), 0.0, 1.0))


# Phase 1C: require apple released in bowl. Goal becomes 5D:
# [x, y, z, is_low, gripper_open]
# (`settled` was dropped — too noisy in sim physics, tended to never fire)
GRIPPER_OPEN_THRESH = 0.025  # finger qpos above this = "open enough"
GRIPPER_CLOSED_THRESH = 0.005  # below this = "closed"

def compute_gripper_open(finger_qpos):
    """Smooth flag [0,1]. 1 when finger pads are open, 0 when closed."""
    return float(np.clip((finger_qpos - GRIPPER_CLOSED_THRESH) /
                         (GRIPPER_OPEN_THRESH - GRIPPER_CLOSED_THRESH), 0.0, 1.0))


class GripperOverrideWrapper(gym.Wrapper):
    """Bypasses the broken default Panda controller for the gripper.

    KNOWN BUG: With HYBRID_MOBILE_BASE + GRIP controller, action[6] does NOT
    open the gripper — fingers stay closed regardless of action sign. Verified
    by direct probing (no action dim opens fingers, but force-setting qpos to
    0.04 makes the apple fall correctly).

    This wrapper intercepts action[6]: if > 0.0, it directly writes finger
    qpos to the open position (0.04 / -0.04 for the two fingers) AFTER the
    underlying env step. The controller's broken "always close" behavior is
    overridden, and physics correctly drops the apple.
    """
    OPEN_QPOS = 0.04
    CLOSED_QPOS_MAG = 0.005

    def __init__(self, env):
        super().__init__(env)
        # Walk to raw env to access sim.model
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        self._raw = raw
        self._finger_addrs_signed = []  # list of (addr, sign) where sign is +1 or -1
        try:
            sim = raw.sim
            for fname, sgn in (
                ("gripper0_right_finger_joint1", +1),
                ("gripper0_right_finger_joint2", -1),
            ):
                if fname in sim.model.joint_names:
                    jid = sim.model.joint_name2id(fname)
                    addr = int(sim.model.jnt_qposadr[jid])
                    self._finger_addrs_signed.append((addr, sgn))
        except Exception:
            pass

    def _force_gripper(self, open_amount):
        """open_amount in [0,1]: 0=closed, 1=fully open."""
        if not self._finger_addrs_signed:
            return
        target_mag = self.CLOSED_QPOS_MAG + open_amount * (self.OPEN_QPOS - self.CLOSED_QPOS_MAG)
        for addr, sign in self._finger_addrs_signed:
            self._raw.sim.data.qpos[addr] = sign * target_mag
        self._raw.sim.forward()

    def step(self, action):
        # GRIPPER OVERRIDE: applied BEFORE and AFTER env.step()
        # - BEFORE: opens/closes fingers so apple physics resolve during substeps
        # - AFTER: re-applies state so gripper_open metric reflects intent
        # NO STICKY ZONE: previously cmd in [-0.5, 0] left fingers in their last
        # state, which made gripper "stick open" after first release (artefact
        # of multi-release pattern). Now: cmd>0 → open, cmd<=0 → close. Policy
        # must actively choose to keep gripper open every step.
        a = np.asarray(action, dtype=np.float32)
        cmd = float(a[6]) if a.shape[0] > 6 else 0.0
        open_amount = max(0.0, min(1.0, (cmd + 1.0) / 2.0))

        if cmd > 0.0:
            self._force_gripper(open_amount)  # BEFORE: open
        else:
            self._force_gripper(0.0)          # BEFORE: close (no sticky zone)

        result = self.env.step(action)

        if cmd > 0.0:
            self._force_gripper(open_amount)  # AFTER: open
        else:
            self._force_gripper(0.0)          # AFTER: close
        return result


class ScaledBaseWrapper(gym.ActionWrapper):
    """Scales base + torso action dimensions by a configurable factor, while
    keeping the FULL 12-dim action space exposed to the policy.

    PandaOmron action layout (12 dims):
        0-5 : arm OSC (dx, dy, dz, drx, dry, drz)
        6   : gripper (close/open)
        7-9 : base (x, y, theta)  ← MOBILE BASE
        10  : torso (z)
        (11 may be unused or aux)

    DOF curriculum: at the start, base+torso scales = 0 → robot body is
    effectively frozen, policy learns manipulation only. As training
    progresses, scales rise → policy learns to coordinate locomotion +
    manipulation. The action space stays 12-dim so the policy can transfer.

    Default scales: base_xy=0, base_rot=0, torso=0 (manipulation-only).
    """
    def __init__(self, env):
        super().__init__(env)
        # Scales for base/torso dims (start at 0, curriculum raises them)
        self.base_xy_scale = 0.0   # action[7], action[8] (forward, lateral)
        self.base_rot_scale = 0.0  # action[9] (rotation)
        self.torso_scale = 0.0     # action[10]
        # Action space unchanged (12-dim), policy keeps full output
        self.action_space = env.action_space

    def set_dof_scales(self, xy=None, rot=None, torso=None):
        if xy is not None: self.base_xy_scale = float(xy)
        if rot is not None: self.base_rot_scale = float(rot)
        if torso is not None: self.torso_scale = float(torso)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).copy()
        if a.shape[0] >= 9:
            a[7] *= self.base_xy_scale
            a[8] *= self.base_xy_scale
        if a.shape[0] >= 10:
            a[9] *= self.base_rot_scale
        if a.shape[0] >= 11:
            a[10] *= self.torso_scale
        return a


# DOF curriculum schedule: scale up base + torso progressively.
# DELAYED to give 1C release time to learn BEFORE adding base-motion noise.
# - 0..300k:    body frozen → learn manipulation + release without base interference
# - 300..500k:  small base motion (0.1×), tiny rotation, no torso
# - 500k..end:  full body control
DOF_SCHEDULE = [
    (0,       {"xy": 0.0, "rot": 0.0,  "torso": 0.0}),
    (300_000, {"xy": 0.1, "rot": 0.05, "torso": 0.0}),
    (500_000, {"xy": 1.0, "rot": 1.0,  "torso": 1.0}),
]


class BowlGoalWrapper(gym.Wrapper):
    """Convert the bowl task into a goal-conditioned env compatible with SB3 HER.

    Observation becomes a Dict with:
      - 'observation': original obs
      - 'achieved_goal': current apple xyz
      - 'desired_goal': bowl target xyz (constant per episode)

    Reward:
      - Sparse: 0 if dist(apple, target) < SUCCESS_THRESHOLD, else -1
    """

    def __init__(self, env):
        super().__init__(env)
        # Probe original obs space
        obs = env.observation_space
        obs_dim = obs.shape[0]

        # Goal is 5D: [x, y, z, is_low, gripper_open]
        # 1A uses [x,y]; 1B uses [x,y,z,is_low]; 1C adds gripper_open.
        # `settled` was dropped — too noisy with sim physics; never fired reliably.
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        })
        self._raw = self._get_raw_env()
        self.desired_goal = None
        # Cache joint addresses for gripper open flag (1C goal component).
        self._finger_qpos_addrs = []
        self._cache_addrs()
        # Track previous gripper_open value for release-event detection (G)
        self._prev_gripper_open = 0.0

    def _cache_addrs(self):
        try:
            for fname in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2"):
                if fname in self._raw.sim.model.joint_names:
                    fjid = self._raw.sim.model.joint_name2id(fname)
                    self._finger_qpos_addrs.append(self._raw.sim.model.jnt_qposadr[fjid])
        except Exception:
            pass

    def _get_raw_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            cur = cur.env
        return cur

    def _apple_pos(self):
        """Returns 5D goal: [x, y, z, is_low, gripper_open]."""
        xyz = self._raw.sim.data.qpos[59:62]
        is_low = compute_is_low(xyz[2])
        if self._finger_qpos_addrs:
            finger_vals = [abs(self._raw.sim.data.qpos[a]) for a in self._finger_qpos_addrs]
            gripper_open = compute_gripper_open(float(np.mean(finger_vals)))
        else:
            gripper_open = 0.0
        return np.array([xyz[0], xyz[1], xyz[2], is_low, gripper_open], dtype=np.float32)

    def _bowl_target(self):
        """Returns 5D target: [bowl_x, bowl_y, bowl_z+0.10, 1.0, 1.0]."""
        bowl_id = self._raw.sim.model.body_name2id("distr_counter_main")
        target = self._raw.sim.data.body_xpos[bowl_id].copy()
        target[2] += 0.10
        return np.array([target[0], target[1], target[2], 1.0, 1.0], dtype=np.float32)

    def _success_mask(self, achieved_goal, desired_goal):
        """Returns boolean mask for success per phase (vectorized)."""
        d_xy = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        d_z = np.abs(achieved_goal[..., 2] - desired_goal[..., 2])
        is_low_a = achieved_goal[..., 3]
        if CURRENT_REWARD_PHASE == "1A":
            return d_xy < SUCCESS_THRESHOLD_XY
        elif CURRENT_REWARD_PHASE == "1B":
            return (d_xy < SUCCESS_THRESHOLD) & (d_z < 0.15) & (is_low_a > 0.5)
        else:
            gripper_open_a = achieved_goal[..., 4]
            return ((d_xy < SUCCESS_THRESHOLD) & (d_z < 0.10)
                    & (is_low_a > 0.7) & (gripper_open_a > 0.7))

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Dense step-wise reward with success bonus and phase-aware shaping.

        Phase 1A (approach, XY only):
            -1.0 (d_xy >= 0.40)
            -0.5 (d_xy <  0.40)
            -0.2 (d_xy <  0.25)
            +1.0 (d_xy <  0.15)   ← success bonus
        Phase 1B (descent, XY + Z + is_low):
            -1.0, -0.5, -0.2, -0.05 partial tiers, +1.0 at success.
        Phase 1C (release + settled, 6D):
            -1.0 → -0.5 (in position) → smooth bonuses for is_low / gripper / settled
            → +1.0 at full success.
            Soft HER filter: only is_low<0.2 (apple in air) → -1.
            Held & moving relabels stay valid (gradient signal preserved).
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        scalar = (achieved_goal.ndim == 1)
        if scalar:
            achieved_goal = achieved_goal[None, :]
            desired_goal = desired_goal[None, :]

        d_xy = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        d_z = np.abs(achieved_goal[..., 2] - desired_goal[..., 2])
        is_low_a = achieved_goal[..., 3]
        success = self._success_mask(achieved_goal, desired_goal)

        if CURRENT_REWARD_PHASE == "1A":
            rewards = -np.ones_like(d_xy)
            rewards = np.where(d_xy < 0.40, -0.5, rewards)
            rewards = np.where(d_xy < 0.25, -0.2, rewards)
            rewards = np.where(success, 1.0, rewards)  # success bonus

        elif CURRENT_REWARD_PHASE == "1B":
            rewards = -np.ones_like(d_xy)
            rewards = np.where(d_xy < 0.30, -0.5, rewards)
            near_xy = d_xy < 0.15
            rewards = np.where(near_xy & (d_z < 0.30), -0.2, rewards)
            rewards = np.where(near_xy & (d_z < 0.15) & (is_low_a > 0.3), -0.05, rewards)
            rewards = np.where(success, 1.0, rewards)  # success bonus

        else:  # "1C" — release via shaping GATED on XY-alignment (anti-hack)
            gripper_open_a = achieved_goal[..., 4]
            rewards = -np.ones_like(d_xy)

            # Near tier: kept binary at 15cm for the proximity feedback.
            near = (d_xy < 0.15) & (d_z < 0.15)
            rewards = np.where(near, -0.5, rewards)

            # CONTINUOUS PROXIMITY-BASED SHAPING (Plan profond) — replaces binary
            # threshold at 12cm. The previous binary cliff created a "no-mans-land"
            # at 12-22cm where the policy had no gradient (couldn't reach bonus zone,
            # plateaued just outside). Continuous proximity gives smooth gradient
            # all the way from d_xy=30cm down to 0cm.
            #
            # prox = 1.0 at d_xy<=10cm (success threshold), 0.0 at d_xy>=30cm
            # Crossover (open == closed reward): d_xy = 20cm
            #   above 20cm: closed better
            #   below 20cm: open better → smooth pull toward release+bowl
            prox = np.clip((0.30 - d_xy) / 0.20, 0.0, 1.0)
            rewards = rewards + 2.0 * is_low_a * gripper_open_a * prox     # bonus scales smoothly with proximity
            rewards = rewards - 2.0 * gripper_open_a * (1.0 - prox)        # penalty scales smoothly with distance
            # Open-high penalty (Beta): unchanged — strict to forbid "release during approach"
            apple_high = (1.0 - is_low_a)
            rewards = rewards - 0.5 * (gripper_open_a * apple_high)

            # Success bonus DOMINATES bonus tier. Earlier value 1.0 was LESS than the
            # bonus tier max (-0.5 + 0.2 + 2.0 = +1.7/step), so the policy preferred
            # camping at d_xy~10-15cm over actually succeeding (Success collapsed to 0%).
            # +5.0 ensures success > any shaping tier (gradient = +3.3/step toward goal).
            rewards = np.where(success, 5.0, rewards)

            # HARD HER filter (Fix #1): relabels valid ONLY when the relabeled goal
            # corresponds to a state matching the success criteria (apple low AND
            # gripper open). Eliminates HER↔shaping conflict for the gripper/low
            # axes — but NOT for the spatial axis (apple position).
            invalid = (desired_goal[..., 3] < 0.5) | (desired_goal[..., 4] < 0.5)
            # SPATIAL HER filter (Fix #3): also reject relabels where the relabeled
            # goal position is far from the REAL bowl (self.desired_goal[:3] is set
            # at reset to the actual bowl target). This blocks the "drop apple at
            # 25cm = success in HER" exploit that maintained the local minimum
            # ReleD~25cm + GripF~0.91. Threshold 20cm = 2x success_threshold,
            # generous enough to keep useful relabels (precise approaches), strict
            # enough to invalidate "release somewhere" patterns.
            try:
                real_bowl_xy = self.desired_goal[:2]  # constant per env (no randomization)
                relabel_offset = np.linalg.norm(desired_goal[..., :2] - real_bowl_xy[None, :], axis=-1)
                spatial_invalid = relabel_offset > 0.12  # was 0.20 — Fix #3b: tighten to bonus zone (12cm) to eliminate the 12-20cm gray zone where HER was still rewarding "drop somewhere" patterns. Aligns HER valid relabels with the shaping bonus tier.
                invalid = invalid | spatial_invalid
            except (AttributeError, TypeError):
                pass  # self.desired_goal not yet set (very first call before reset)
            rewards = np.where(invalid, -1.0, rewards)

        rewards = rewards.astype(np.float32)
        return float(rewards[0]) if scalar else rewards

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        self.desired_goal = self._bowl_target()
        achieved_goal = self._apple_pos()
        # Initialize prev_gripper for release-event detection
        self._prev_gripper_open = float(achieved_goal[4])
        dict_obs = {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal,
            "desired_goal": self.desired_goal,
        }
        return dict_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        achieved_goal = self._apple_pos()
        # Dense step-wise reward — what HER uses for relabeling.
        # IMPORTANT: total reward = compute_reward exactly, no hidden bonus.
        # The release shaping is now done INSIDE compute_reward via the
        # continuous term 0.3 * is_low * gripper_open (HER-consistent).
        # This replaces the previous release_bonus (which was outside compute_reward
        # and broke HER consistency by adding a signal HER couldn't relabel).
        cur_gripper_open = float(achieved_goal[4])
        self._prev_gripper_open = cur_gripper_open  # kept for potential debug logging
        total_reward = float(self.compute_reward(achieved_goal, self.desired_goal, info))
        release_bonus = 0.0  # disabled — replaced by continuous shaping in compute_reward
        # Diagnostics
        d_full = np.linalg.norm(achieved_goal - self.desired_goal)
        d_spatial = np.linalg.norm(achieved_goal[:3] - self.desired_goal[:3])
        d_xy = np.linalg.norm(achieved_goal[:2] - self.desired_goal[:2])  # XY-only for OpenD/ReleD metrics
        success_bool = bool(self._success_mask(achieved_goal[None, :], self.desired_goal[None, :])[0])
        info["is_success"] = float(success_bool)
        info["place_dist"] = float(d_spatial)
        info["d_xy"] = float(d_xy)  # for diagnostic: where does the policy open the gripper?
        info["apple_pos"] = achieved_goal[:3].copy()
        info["gripper_open"] = float(achieved_goal[4])  # 1=open, 0=closed
        info["goal_dist_5d"] = float(d_full)
        info["reward_phase"] = CURRENT_REWARD_PHASE
        info["release_bonus"] = release_bonus
        dict_obs = {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal,
            "desired_goal": self.desired_goal,
        }
        return dict_obs, total_reward, terminated, truncated, info


# Early-stopping milestones. Includes "tolerance" milestones AFTER curriculum
# transitions where success_rate is allowed to drop temporarily (the env just
# changed; the policy needs to adapt).
EARLY_STOP_CRITERIA = [
    # Very soft early-stop — only kill if the run is FUNDAMENTALLY broken.
    # Dense reward + curriculum can converge slowly; we don't want to kill a
    # run that would have succeeded at 120k just because it didn't hit a 80k
    # milestone. Major checkpoints only.
    (30_000,  "apple_variance_cm", 10.0, "AppleVar > 10cm (sanity: apple moves)"),
    (280_000, "success_rate",      0.02, "Success > 2% end of full phase (relaxed +30k for deep redesign — continuous reward needs more critic adaptation)"),
    (450_000, "success_rate",      0.05, "Success > 5% end of partial phase (soft)"),
    (700_000, "success_rate",      0.05, "Success > 5% end of training (very soft)"),
]

# Curriculum schedule for pre_grasp difficulty:
CURRICULUM_SCHEDULE = [
    (0,       "full"),     # 0-350k: apple held, just place (extended again for deep redesign)
    (350_000, "partial"),  # 350-450k: apple near gripper, must close+lift
    (450_000, "none"),     # 450-700k: full task (grasp+lift+place)
]

# Curriculum on success threshold (3D, used in 1B+ phases).
THRESHOLD_SCHEDULE = [
    (0,       0.20),  # 0-50k: 20cm sphere (very lenient)
    (50_000,  0.15),  # 50-150k: 15cm
    (150_000, 0.10),  # 150k+: real 10cm threshold
]

# Reward-phase schedule:
# - 1A (0-100k):    XY-only success — just approach the bowl horizontally
# - 1B (100-200k):  4D (xyz + is_low) — must descend to bowl level
# - 1C (200k+):     6D (+ gripper_open + settled) — must RELEASE apple in bowl
# The 1A→1B transition was tested OK before. 1C only activates after 1B is
# stable (200k = end of pre_grasp full phase, plenty of time for placement).
PHASE_SCHEDULE = [
    (0,       "1A"),
    (100_000, "1B"),
    (200_000, "1C"),
]


def set_success_threshold(new_threshold):
    """Mutate the module-level SUCCESS_THRESHOLD global. compute_reward() picks
    up the new value at the next call (HER's relabeling included)."""
    global SUCCESS_THRESHOLD
    SUCCESS_THRESHOLD = new_threshold


def set_reward_phase(new_phase):
    """Mutate CURRENT_REWARD_PHASE. Switches reward computation between 1A
    (XY-only) and 1B+ (full 4D with HER air-relabel filter)."""
    global CURRENT_REWARD_PHASE
    CURRENT_REWARD_PHASE = new_phase


class DOFCurriculumCallback(BaseCallback):
    """Updates the ScaledBaseWrapper's base/torso scales at scheduled steps.
    Phase-by-phase opens up locomotion DOFs once manipulation is learned."""
    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.applied = None

    def _on_step(self):
        target = None
        for ts, scales in self.schedule:
            if self.num_timesteps >= ts:
                target = scales
        if target is not None and target != self.applied:
            print(f"\n{'='*70}\nDOF CURRICULUM @ {self.num_timesteps}: scales {target}\n{'='*70}", flush=True)
            try:
                self.training_env.env_method("set_dof_scales", target.get("xy"), target.get("rot"), target.get("torso"))
            except Exception:
                # Fallback: walk wrappers
                for env in self.training_env.envs:
                    inner = env
                    while hasattr(inner, "env"):
                        if hasattr(inner, "set_dof_scales"):
                            inner.set_dof_scales(**target)
                            break
                        inner = inner.env
            self.applied = target
        return True


class PhaseCallback(BaseCallback):
    """Updates the global CURRENT_REWARD_PHASE at scheduled timesteps.
    Also boosts ent_coef when entering 1C (lever D) — forces SAC to explore
    more aggressively, breaking deterministic local optima like 'hover'."""
    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_phase = None

    def _on_step(self):
        target = None
        for ts, phase in self.schedule:
            if self.num_timesteps >= ts:
                target = phase
        if target is not None and target != self.current_phase:
            print(f"\n{'='*70}\nREWARD PHASE ADVANCE @ {self.num_timesteps} steps: {target}\n{'='*70}", flush=True)
            set_reward_phase(target)
            # Lever D: boost ent_coef on 1C to break out of local optima
            if target == "1C":
                try:
                    import torch
                    if hasattr(self.model, "log_ent_coef"):
                        with torch.no_grad():
                            new_log = float(np.log(0.05))   # back to 0.05 — reward fix is the real lever, not exploration
                            self.model.log_ent_coef.data.fill_(new_log)
                        print(f"  → ent_coef set to 0.05 (log={new_log:.3f}) for 1C exploration", flush=True)
                except Exception as ex:
                    print(f"  → couldn't boost ent_coef: {ex}", flush=True)
            self.current_phase = target
        return True


class CurriculumCallback(BaseCallback):
    """Updates env.curriculum_pre_grasp at scheduled timesteps."""
    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_mode = None

    def _on_step(self):
        target = None
        for ts, mode in self.schedule:
            if self.num_timesteps >= ts:
                target = mode
        if target is not None and target != self.current_mode:
            print(f"\n{'='*70}\nCURRICULUM ADVANCE @ {self.num_timesteps} steps: pre_grasp_mode = {target}\n{'='*70}", flush=True)
            value = target if target != "none" else None
            try:
                self.training_env.env_method("set_curriculum_pre_grasp", value)
            except Exception:
                # Fallback: set attribute directly
                try:
                    for env in self.training_env.envs:
                        inner = env
                        while hasattr(inner, "env"):
                            inner = inner.env
                        inner.curriculum_pre_grasp = value
                except Exception as ex:
                    print(f"  WARN: couldn't update curriculum: {ex}")
            self.current_mode = target
        return True


class ThresholdCallback(BaseCallback):
    """Updates the global SUCCESS_THRESHOLD at scheduled timesteps. Easier
    threshold early gives HER more positive transitions to bootstrap, then
    tightens to the real 10cm target.
    """
    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_threshold = None

    def _on_step(self):
        target = None
        for ts, thr in self.schedule:
            if self.num_timesteps >= ts:
                target = thr
        if target is not None and target != self.current_threshold:
            print(f"\n{'='*70}\nTHRESHOLD ADVANCE @ {self.num_timesteps} steps: success_threshold = {target}m\n{'='*70}", flush=True)
            set_success_threshold(target)
            self.current_threshold = target
        return True


class MetricsCallback(BaseCallback):
    """Track success_rate, min_dist, and DIAGNOSTIC metrics:
    - apple_variance: variance of apple position during episode (catches "no movement" failure)
    - apple_max_lift: max lift above counter (catches "no lift" failure)

    Also enforces EARLY_STOP_CRITERIA — kills training if a milestone fails.
    """
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.writer = None
        self.episode_min_dist = []
        self.episode_variance = []   # diag: did apple move?
        self.episode_max_lift = []   # diag: was apple ever lifted?
        self.episode_max_drop = []   # diag: was apple lowered toward bowl?
        self.episode_z_range = []    # diag: total z-range covered (release indicator)
        self.episode_final_z = []    # diag: apple z at end-of-episode (settled in bowl?)
        self.episode_final_gripper = []  # diag: gripper open at end-of-episode (released?)
        self.episode_open_dist = []      # diag: avg d_xy WHEN gripper is open (cm). Tells us WHERE the policy opens.
        self.episode_release_dist = []   # diag: d_xy at the FIRST closed→open transition per episode (cm)
        self.current_min_dist = float("inf")
        self.current_apple_positions = []
        self.current_apple_zs = []   # all apple_z during episode
        self.current_gripper_opens = []  # all gripper_open during episode
        self.current_open_dxys = []  # d_xy values for steps where gripper is open
        self.current_release_dxy = None  # d_xy at first release event
        self.current_max_lift = 0.0
        self.current_max_drop = 0.0
        self.current_apple_z_initial = None
        self._prev_grip_open = 0.0  # for release-event detection
        self._stop = False
        self._milestones_checked = set()

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.csv_file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["timestep", "ep_rew_mean", "ep_len_mean", "success_rate",
                        "min_dist_avg_cm", "apple_variance_cm", "apple_max_lift_cm",
                        "apple_max_drop_cm", "z_range_cm", "final_apple_z_m",
                        "final_gripper_open", "open_dist_cm", "release_dist_cm",
                        "n_episodes"],
        )
        self.writer.writeheader()
        # When resuming from a checkpoint, mark all milestones whose timestep is
        # already past as "checked" — they were validated during the original
        # run, no need to re-check (and the metrics buffer is empty at resume,
        # so re-checking would always fail).
        try:
            n = int(self.num_timesteps)
        except Exception:
            n = 0
        for ts, *_ in EARLY_STOP_CRITERIA:
            if ts < n:
                self._milestones_checked.add(ts)
                if self.verbose >= 1:
                    print(f"  → milestone {ts} skipped (already passed before resume)", flush=True)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if len(infos) > 0:
            info = infos[0]
            if "place_dist" in info:
                self.current_min_dist = min(self.current_min_dist, info["place_dist"])
            if "apple_pos" in info:
                self.current_apple_positions.append(info["apple_pos"])
                apple_z = info["apple_pos"][2]
                self.current_apple_zs.append(apple_z)
                if "gripper_open" in info:
                    self.current_gripper_opens.append(info["gripper_open"])
                # Initialize apple_z_initial on first step of episode
                if self.current_apple_z_initial is None:
                    self.current_apple_z_initial = apple_z
                # max lift above counter (z=0.95)
                self.current_max_lift = max(self.current_max_lift, apple_z - 0.95)
                # max drop below initial position (positive = descended toward bowl)
                drop = self.current_apple_z_initial - apple_z
                if drop > self.current_max_drop:
                    self.current_max_drop = drop
            # OpenD/ReleD diagnostic: track WHERE the gripper opens (XY distance to bowl)
            if "d_xy" in info and "gripper_open" in info:
                grip = float(info["gripper_open"])
                dxy = float(info["d_xy"])
                if grip > 0.5:
                    self.current_open_dxys.append(dxy)
                # Detect first closed→open transition (the "release" event)
                if grip > 0.5 and self._prev_grip_open <= 0.5 and self.current_release_dxy is None:
                    self.current_release_dxy = dxy
                self._prev_grip_open = grip
            if len(dones) > 0 and dones[0]:
                self.episode_min_dist.append(self.current_min_dist)
                self.episode_max_lift.append(self.current_max_lift)
                self.episode_max_drop.append(self.current_max_drop)
                # z_range (apple z amplitude during episode — release indicator)
                if self.current_apple_zs:
                    z_range = max(self.current_apple_zs) - min(self.current_apple_zs)
                    self.episode_z_range.append(z_range)
                    # final apple_z (where the apple ended up — settled in bowl?)
                    self.episode_final_z.append(self.current_apple_zs[-1])
                # final gripper_open (was the gripper open at episode end? = released?)
                if self.current_gripper_opens:
                    self.episode_final_gripper.append(self.current_gripper_opens[-1])
                # Compute apple variance (diagnostic: did it move?)
                if len(self.current_apple_positions) > 1:
                    arr = np.array(self.current_apple_positions)
                    span = arr.max(axis=0) - arr.min(axis=0)
                    variance = float(np.linalg.norm(span))  # 3D span
                    self.episode_variance.append(variance)
                # OpenD/ReleD: commit per-episode aggregates
                if self.current_open_dxys:
                    self.episode_open_dist.append(float(np.mean(self.current_open_dxys)))
                if self.current_release_dxy is not None:
                    self.episode_release_dist.append(self.current_release_dxy)
                self.current_min_dist = float("inf")
                self.current_apple_positions = []
                self.current_apple_zs = []
                self.current_gripper_opens = []
                self.current_open_dxys = []
                self.current_release_dxy = None
                self._prev_grip_open = 0.0
                self.current_max_lift = 0.0
                self.current_max_drop = 0.0
                self.current_apple_z_initial = None
                # Roll buffers
                for buf in [self.episode_min_dist, self.episode_max_lift, self.episode_variance,
                            self.episode_max_drop, self.episode_z_range, self.episode_final_z,
                            self.episode_final_gripper, self.episode_open_dist,
                            self.episode_release_dist]:
                    if len(buf) > 100:
                        buf[:] = buf[-100:]
        # Stop training if early-stop flag was raised by _on_rollout_end
        return not self._stop

    def _on_rollout_end(self):
        if not self.model.ep_info_buffer:
            return
        ep_rews = [ep["r"] for ep in self.model.ep_info_buffer]
        ep_lens = [ep["l"] for ep in self.model.ep_info_buffer]
        successes = [ep.get("is_success", 0) for ep in self.model.ep_info_buffer]
        min_dist_cm = float(np.mean(self.episode_min_dist)) * 100 if self.episode_min_dist else 99900.0
        variance_cm = float(np.mean(self.episode_variance)) * 100 if self.episode_variance else 0.0
        max_lift_cm = float(np.mean(self.episode_max_lift)) * 100 if self.episode_max_lift else 0.0
        max_drop_cm = float(np.mean(self.episode_max_drop)) * 100 if self.episode_max_drop else 0.0
        z_range_cm = float(np.mean(self.episode_z_range)) * 100 if self.episode_z_range else 0.0
        final_z_m = float(np.mean(self.episode_final_z)) if self.episode_final_z else 0.0
        final_gripper = float(np.mean(self.episode_final_gripper)) if self.episode_final_gripper else 0.0
        # OpenD/ReleD diagnostic (cm). -1 means metric not yet available (no open events).
        open_d_cm = float(np.mean(self.episode_open_dist)) * 100 if self.episode_open_dist else -1.0
        rele_d_cm = float(np.mean(self.episode_release_dist)) * 100 if self.episode_release_dist else -1.0
        self.writer.writerow({
            "timestep": self.num_timesteps,
            "ep_rew_mean": float(np.mean(ep_rews)),
            "ep_len_mean": float(np.mean(ep_lens)),
            "success_rate": float(np.mean(successes)),
            "min_dist_avg_cm": min_dist_cm,
            "apple_variance_cm": variance_cm,
            "apple_max_lift_cm": max_lift_cm,
            "apple_max_drop_cm": max_drop_cm,
            "z_range_cm": z_range_cm,
            "final_apple_z_m": final_z_m,
            "final_gripper_open": final_gripper,
            "open_dist_cm": open_d_cm,
            "release_dist_cm": rele_d_cm,
            "n_episodes": len(ep_rews),
        })
        self.csv_file.flush()
        if self.verbose >= 1:
            print(f"Steps: {self.num_timesteps:>7}  Reward: {np.mean(ep_rews):+8.2f}  "
                  f"Success: {np.mean(successes):.1%}  MinDist: {min_dist_cm:.1f}cm  "
                  f"AppleVar: {variance_cm:.1f}cm  Lift: {max_lift_cm:.1f}cm  Drop: {max_drop_cm:.1f}cm  "
                  f"Zrange: {z_range_cm:.1f}cm  Zfinal: {final_z_m:.2f}m  GripF: {final_gripper:.2f}  "
                  f"OpenD: {open_d_cm:.1f}cm  ReleD: {rele_d_cm:.1f}cm  "
                  f"n_eps: {len(ep_rews)}")

        # Early-stopping check: any milestone we've passed without checking?
        current = {"success_rate": float(np.mean(successes)),
                   "apple_variance_cm": variance_cm,
                   "min_dist_avg_cm": min_dist_cm,
                   "apple_max_lift_cm": max_lift_cm}
        for ts, metric, threshold, label in EARLY_STOP_CRITERIA:
            if ts in self._milestones_checked or self.num_timesteps < ts:
                continue
            self._milestones_checked.add(ts)
            value = current.get(metric, 0.0)
            if value < threshold:
                print(f"\n{'='*70}\nEARLY STOP at {self.num_timesteps} steps")
                print(f"  Milestone {ts}: required '{label}' but got {metric}={value:.3f}")
                print(f"  Stopping training.\n{'='*70}")
                self._stop = True
            else:
                print(f"  ✓ Milestone {ts} passed ({label}: got {metric}={value:.3f})")

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


def make_env(args, rank, monitor_root):
    def _init():
        robots = "PandaOmron"
        cc = load_composite_controller_config(controller=None, robot=robots)
        env = MyPnPCounterToCab(
            robots=robots, controller_configs=cc,
            use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
            reward_shaping=False, control_freq=20, ignore_done=False,
            seed=args.seed + rank, horizon=args.horizon,
        )
        # CRITICAL: enable pre_grasp curriculum so apple is in robot's hand
        # at episode start. This solves the chicken-and-egg of HER:
        # apple must MOVE during episodes for HER's relabeling to be useful.
        # With pre_grasp=full, even random arm motions move the apple.
        env.curriculum_pre_grasp = args.pre_grasp_mode if args.pre_grasp_mode != "none" else None
        env.reset()
        env = GymWrapper(env, keys=None)
        env = GripperOverrideWrapper(env)  # bypass broken gripper controller
        env = ScaledBaseWrapper(env)  # base/torso scaled (curriculum), full 12-dim
        env = BowlGoalWrapper(env)
        log_dir = os.path.join(monitor_root, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"), info_keywords=("is_success",))
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--total_timesteps", type=int, default=200000)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--ent_coef", type=str, default="auto")

    # HER specific
    parser.add_argument("--n_sampled_goal", type=int, default=4)
    parser.add_argument("--her_strategy", type=str, default="future",
                        choices=["future", "final", "episode"])

    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")

    # Pre-grasp curriculum: bootstrap HER signal by ensuring apple moves
    parser.add_argument("--pre_grasp_mode", type=str, default="full",
                        choices=["full", "partial", "none"],
                        help="full=apple in closed gripper, partial=near gripper, none=natural")

    # Resume: load policy weights from a checkpoint and continue learning.
    # The replay buffer is NOT restored (we don't save it on disk), so HER
    # rebuilds its buffer from scratch — but the policy's learned weights are
    # preserved, which is the main asset of the run.
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .zip checkpoint to resume policy weights from")
    parser.add_argument("--resume_steps", type=int, default=0,
                        help="Already-completed steps (used for early-stop milestone tracking)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sac_her_bowl_{timestamp}"
    run_dir = os.path.join(args.model_dir, run_name)
    log_root = os.path.join(run_dir, "logs")
    tb_dir = os.path.join(log_root, "tensorboard")
    monitor_root = os.path.join(log_root, "monitor")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(monitor_root, exist_ok=True)

    # HER requires single env (no SubprocVecEnv with HER in SB3)
    env = DummyVecEnv([make_env(args, 0, monitor_root)])

    ent_coef_value = args.ent_coef
    try:
        ent_coef_value = float(args.ent_coef)
    except ValueError:
        pass

    if args.resume_from:
        print(f"=> Resuming policy from {args.resume_from}", flush=True)
        model = SAC.load(args.resume_from, env=env, device=args.device,
                         tensorboard_log=tb_dir, verbose=1)
        # Restore the timestep counter so milestone checks fire at correct steps
        if args.resume_steps > 0:
            model.num_timesteps = args.resume_steps
            model._num_timesteps_at_start = args.resume_steps
        # Replay buffer is EMPTY after SAC.load (we don't save it). Force a
        # fresh warmup so HER doesn't try to sample an empty buffer.
        model.learning_starts = (args.resume_steps if args.resume_steps > 0 else 0) + max(args.learning_starts, 1000)
        print(f"=> Effective learning_starts after resume: {model.learning_starts} (warmup before training)", flush=True)
        # OVERRIDE action_noise on resume — moderate noise on gripper.
        # σ=0.5 (was 1.0): high enough to give ~7-10% positive samples for
        # exploration (combined with policy bias -0.755 + std 0.225 + noise 0.5).
        # NOT σ=1.0 because that destabilized placement skill (apple released
        # everywhere). The reward fix (XY-gated bonus) does the heavy lifting
        # now — noise just needs to enable some exploration.
        n_actions = env.action_space.shape[-1]
        sigma_per_dim = 0.1 * np.ones(n_actions, dtype=np.float32)
        if n_actions > 6:
            sigma_per_dim[6] = 0.5   # moderate noise — paired with anti-hack reward
        model.action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=sigma_per_dim,
        )
        print(f"=> action_noise overridden: sigma={sigma_per_dim.tolist()} (gripper σ=0.5)", flush=True)
    else:
        # Action noise (lever E): adds Gaussian noise to actions on top of SAC's
        # stochastic policy. Helps prevent the policy from getting stuck in a
        # deterministic local optimum (especially "hovering" without releasing).
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model = SAC(
            policy="MultiInputPolicy", env=env,
            learning_rate=args.learning_rate, buffer_size=args.buffer_size,
            batch_size=args.batch_size, tau=args.tau, gamma=args.gamma,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts, ent_coef=ent_coef_value,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": args.n_sampled_goal,
                "goal_selection_strategy": args.her_strategy,
            },
            action_noise=action_noise,
            verbose=1, seed=args.seed, tensorboard_log=tb_dir, device=args.device,
        )

    print(f"\n{'='*60}\nSAC + HER on Bowl Task\n{'='*60}")
    print(f"Run name:        {run_name}")
    print(f"Total steps:     {args.total_timesteps:,}")
    print(f"HER strategy:    {args.her_strategy}, n_sampled_goal={args.n_sampled_goal}")
    print(f"Sparse reward:   0 if dist<{SUCCESS_THRESHOLD}, else -1")
    print(f"Target:          bowl center + 10cm height\n")

    metrics_cb = MetricsCallback(os.path.join(log_root, "metrics.csv"), verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="sac_her", save_replay_buffer=False,
    )
    curriculum_cb = CurriculumCallback(CURRICULUM_SCHEDULE, verbose=1)
    threshold_cb = ThresholdCallback(THRESHOLD_SCHEDULE, verbose=1)
    phase_cb = PhaseCallback(PHASE_SCHEDULE, verbose=1)
    dof_cb = DOFCurriculumCallback(DOF_SCHEDULE, verbose=1)
    print(f"Curriculum (pre_grasp): {CURRICULUM_SCHEDULE}")
    print(f"Curriculum (threshold): {THRESHOLD_SCHEDULE}")
    print(f"Reward-phase schedule:  {PHASE_SCHEDULE}")
    print(f"DOF schedule (base/torso scaling): {DOF_SCHEDULE}")
    print(f"Goal: 5D [x, y, z, is_low, gripper_open]\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps, progress_bar=False,
            callback=CallbackList([metrics_cb, checkpoint_cb, curriculum_cb, threshold_cb, phase_cb, dof_cb]),
            reset_num_timesteps=(args.resume_from is None),
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving current model...")
    finally:
        save_path = os.path.join(run_dir, "sac_her_final")
        model.save(save_path)
        env.close()
        print(f"\nFinal model: {save_path}.zip")
        print(f"Logs:        {log_root}")


if __name__ == "__main__":
    main()
