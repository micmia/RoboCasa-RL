"""
Custom environment: pick up an apple from the counter and place it into a bowl.

Both the apple (obj) and the bowl (container) start on the counter.
Success is defined as the apple being inside the bowl with the gripper released.

Usage:
    from env import MyPnPAppleToBowl

    env = MyPnPAppleToBowl(
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        reward_shaping=True,
    )
"""

import sys
import os

robocasa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robocasa")
if robocasa_path in sys.path:
    sys.path.remove(robocasa_path)
sys.path.insert(0, robocasa_path)

robosuite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robosuite")
if robosuite_path in sys.path:
    sys.path.remove(robosuite_path)
sys.path.insert(0, robosuite_path)

import robocasa

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
import robocasa.utils.object_utils as OU
import numpy as np


class MyPnPAppleToBowl(PickPlaceCounterToCabinet):
    """
    Pick up an apple from the counter and place it into a bowl on the counter.

    Inherits from PickPlaceCounterToCabinet to reuse the kitchen + counter fixture
    setup, but overrides objects, success check, and reward.
    """

    def __init__(self, *args, **kwargs):
        if "layout_ids" not in kwargs:
            kwargs["layout_ids"] = [1]
        if "style_ids" not in kwargs:
            kwargs["style_ids"] = [1]

        self.custom_seed = kwargs.get("seed", 0)

        self.curriculum_stage = 0
        self.curriculum_difficulty = 0.0
        self.curriculum_obj_size_xy = None
        self.curriculum_obj_rot_range = None

        # Track per-episode reward state for shaped reward
        self._prev_gripper_to_apple_dist = None
        self._prev_apple_to_bowl_dist = None

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Curriculum helpers (mirrors custom_pnp_counter_to_cab.py)
    # ------------------------------------------------------------------

    def set_curriculum(self, stage=None, difficulty=None, obj_size_xy=None, obj_rot_range=None):
        if stage is not None:
            self.curriculum_stage = int(stage)
        if difficulty is not None:
            self.curriculum_difficulty = float(np.clip(float(difficulty), 0.0, 1.0))
        if obj_size_xy is not None:
            x, y = obj_size_xy
            self.curriculum_obj_size_xy = (float(x), float(y))
        if obj_rot_range is not None:
            a, b = obj_rot_range
            self.curriculum_obj_rot_range = (float(a), float(b))

    def _get_curriculum_obj_params(self):
        size_xy = (0.50, 0.30)
        rot_range = (-np.pi / 4, np.pi / 4)
        if self.curriculum_obj_size_xy is not None:
            size_xy = self.curriculum_obj_size_xy
        if self.curriculum_obj_rot_range is not None:
            rot_range = self.curriculum_obj_rot_range
        size_xy = (max(0.0, float(size_xy[0])), max(0.0, float(size_xy[1])))
        rot_range = (float(rot_range[0]), float(rot_range[1]))
        return size_xy, rot_range

    # ------------------------------------------------------------------
    # Episode meta
    # ------------------------------------------------------------------

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the apple and place it into the bowl."
        return ep_meta

    # ------------------------------------------------------------------
    # Scene setup — keep the cabinet door closed (we don't use it)
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Call grandparent to skip the cab.open_door() call in PickPlaceCounterToCabinet
        from robocasa.environments.kitchen.kitchen import Kitchen
        Kitchen._setup_scene(self)

    # ------------------------------------------------------------------
    # Object configs: apple (obj) + bowl (container) on the counter
    # ------------------------------------------------------------------

    def _get_obj_cfgs(self):
        cfgs = []
        base_path = os.path.join(robocasa.models.assets_root, "objects", "objaverse")
        obj_size_xy, obj_rot_range = self._get_curriculum_obj_params()

        apple_path = os.path.join(base_path, "apple", "apple_1", "model.xml")
        cfgs.append(
            dict(
                name="obj",
                obj_groups=apple_path,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=obj_size_xy,
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.05),
                    rotation=obj_rot_range,
                ),
            )
        )

        bowl_path = os.path.join(base_path, "bowl", "bowl_1", "model.xml")
        cfgs.append(
            dict(
                name="container",
                obj_groups=bowl_path,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.50, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.10),
                    rotation=(-0.10, 0.10),
                ),
            )
        )

        return cfgs

    # ------------------------------------------------------------------
    # Fixture placement — deterministic per seed
    # ------------------------------------------------------------------

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        sampler = super()._get_placement_initializer(cfg_list, z_offset)
        is_fixture_placement = cfg_list and cfg_list[0].get("type") == "fixture"
        if is_fixture_placement:
            seed_val = getattr(self, "custom_seed", 0) or 0
            fixed_rng = np.random.default_rng(seed=seed_val)
            sampler.rng = fixed_rng
            if hasattr(sampler, "samplers"):
                for sub in sampler.samplers.values():
                    sub.rng = fixed_rng
        return sampler

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self):
        apple_in_bowl = OU.check_obj_in_receptacle(self, "obj", "container")
        # Use 0.08m threshold (vs default 0.25m): bowl is on the counter so less clearance is needed.
        gripper_far = OU.gripper_obj_far(self, obj_name="obj", th=0.05)
        return apple_in_bowl and gripper_far

