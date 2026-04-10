"""
Custom PnPCounterToCab environment with modified reward function.

This class inherits from the original RoboCasa PnPCounterToCab environment
and allows you to customize the reward function without modifying the
original robocasa or skrl packages.

Usage:
    from env import MyPnPCounterToCab
    
    env = MyPnPCounterToCab(
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
# Ensure local RoboCasa package path has priority over workspace root.
# Otherwise Python may treat top-level `robocasa/` as a namespace package,
# which breaks inspect.getfile(robocasa) inside RoboCasa utilities.
robocasa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robocasa")
if robocasa_path in sys.path:
    sys.path.remove(robocasa_path)
sys.path.insert(0, robocasa_path)

# Ensure local robosuite package is imported as a regular package as well.
robosuite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robosuite")
if robosuite_path in sys.path:
    sys.path.remove(robosuite_path)
sys.path.insert(0, robosuite_path)

import robocasa

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
import robocasa.utils.object_utils as OU
import numpy as np


class MyPnPCounterToCab(PickPlaceCounterToCabinet):
    """
    PnPCounterToCab environment with modified reward function.
    
    This class inherits from the original PickPlaceCounterToCabinet and overrides
    the reward() method to implement a custom reward function.
    
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom environment.
        All arguments are passed to the parent PickPlaceCounterToCabinet class.
        
        This environment fixes the kitchen layout to a single configuration
        while allowing object positions to vary.
        """
        # Fix the kitchen layout to layout 0, style 0 (you can change these values)
        # This prevents the kitchen configuration from changing between episodes
        if 'layout_ids' not in kwargs:
            kwargs['layout_ids'] = [1]  # Use layout 1
        if 'style_ids' not in kwargs:
            kwargs['style_ids'] = [1]   # Use style 1
            
        # Capture the seed if provided
        self.custom_seed = kwargs.get('seed', 0)
        
        super().__init__(*args, **kwargs)
        
    
    def _get_obj_cfgs(self):
        """
        Override to set specific objects:
        - Sample object (obj): always apple_1
        - Distractor objects: always bowl_1
        """
        
        cfgs = []
        
        # Get the base path for robocasa objects
        base_path = os.path.join(robocasa.models.assets_root, "objects", "objaverse")
        # print("base_path", base_path)
        # Sample object: always apple_1 (using full path to model.xml)
        apple_1_path = os.path.join(base_path, "apple", "apple_1", "model.xml")
        # print("apple_1_path", apple_1_path)
        cfgs.append(
            dict(
                name="obj",
                obj_groups=apple_1_path,  # Force apple_1 as the sample object
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        # Distractor on counter: always bowl_1 (using full path to model.xml)
        bowl_1_path = os.path.join(base_path, "bowl", "bowl_1", "model.xml")
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=bowl_1_path,  # Force bowl_1 as distractor
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )

        return cfgs
        
    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        """
        Override to enforce deterministic placement for fixtures (appliances),
        while allowing random placement for objects.
        """
        sampler = super()._get_placement_initializer(cfg_list, z_offset)
        
        # Check if this sampler is for fixtures (appliances)
        # Fixture configs usually have type="fixture"
        is_fixture_placement = False
        if cfg_list and len(cfg_list) > 0:
            if cfg_list[0].get("type") == "fixture":
                is_fixture_placement = True
        
        if is_fixture_placement:
            # Use the environment seed for fixture placement to ensure deterministic behavior
            # appliances will stay in place throughout the run (assuming constant seed for env)
            # but will change if you change the training run seed.
            
            # Retrieve the seed captured in __init__
            seed_val = getattr(self, "custom_seed", 0)
            if seed_val is None:
                seed_val = 0
                
            fixed_rng = np.random.default_rng(seed=seed_val)
            
            # Set the RNG for the main sampler
            sampler.rng = fixed_rng
            
            # Set the RNG for all sub-samplers
            if hasattr(sampler, "samplers"):
                for sub_sampler in sampler.samplers.values():
                    sub_sampler.rng = fixed_rng
                    
        return sampler
        
    def reward(self, action=None):
        r = 0

        return r
 

# Example of how to register this environment with robosuite if needed
def register_custom_env():
    """
    Register the custom environment with robosuite.
    This allows you to use robosuite.make("MyPnPCounterToCab", ...)
    """
    import robosuite
    from robosuite.environments.base import register_env
    
    register_env(MyPnPCounterToCab)
