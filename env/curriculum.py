"""
Curriculum learning utilities for RoboCasa atomic tasks.

This wrapper is intentionally a robosuite-style Wrapper so it can sit *under*
robosuite's GymWrapper (which expects robosuite reset / step signatures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from robosuite.wrappers import Wrapper


@dataclass(frozen=True)
class CurriculumStage:
    # Robot base spawn deviations (Kitchen env attributes)
    robot_pos_x: float
    robot_pos_y: float
    robot_rot: float

    # Object sampling region + rotation (consumed by MyPnPCounterToCab via set_curriculum)
    obj_size_xy: tuple[float, float]
    obj_rot_range: tuple[float, float]


DEFAULT_STAGES: tuple[CurriculumStage, ...] = (
    # Stage 0: very easy (robot close, tiny object region, fixed rotation)
    CurriculumStage(
        robot_pos_x=0.02,
        robot_pos_y=0.01,
        robot_rot=0.0,
        obj_size_xy=(0.08, 0.08),
        obj_rot_range=(0.0, 0.0),
    ),
    # Stage 1: medium (wider region, small orientation noise)
    CurriculumStage(
        robot_pos_x=0.08,
        robot_pos_y=0.03,
        robot_rot=0.0,
        obj_size_xy=(0.30, 0.20),
        obj_rot_range=(-0.30, 0.30),
    ),
    # Stage 2: hard (full region + full yaw randomization, allow base yaw deviation)
    CurriculumStage(
        robot_pos_x=0.15,
        robot_pos_y=0.05,
        robot_rot=0.25,
        obj_size_xy=(0.60, 0.30),
        obj_rot_range=(-np.pi, np.pi),
    ),
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


class CurriculumWrapper(Wrapper):
    """
    Applies curriculum parameters before every underlying env reset.

    Exposes `set_stage` / `set_difficulty` so SB3 callbacks can update it through VecEnv env_method.
    """

    def __init__(
        self,
        env,
        stages: Sequence[CurriculumStage] | None = None,
        initial_stage: int = 0,
    ):
        super().__init__(env)
        self._stages: tuple[CurriculumStage, ...] = tuple(stages) if stages is not None else DEFAULT_STAGES
        if len(self._stages) < 1:
            raise ValueError("CurriculumWrapper requires at least one stage")

        self._stage: int = int(initial_stage)
        self._stage = int(np.clip(self._stage, 0, len(self._stages) - 1))
        self._difficulty: float = self._stage_to_difficulty(self._stage)

    def _stage_to_difficulty(self, stage: int) -> float:
        if len(self._stages) == 1:
            return 0.0
        return float(stage) / float(len(self._stages) - 1)

    def get_stage(self) -> int:
        return int(self._stage)

    def get_difficulty(self) -> float:
        return float(self._difficulty)

    def set_stage(self, stage: int) -> None:
        stage = int(stage)
        self._stage = int(np.clip(stage, 0, len(self._stages) - 1))
        self._difficulty = self._stage_to_difficulty(self._stage)

    def set_difficulty(self, difficulty: float) -> None:
        self._difficulty = _clamp(float(difficulty), 0.0, 1.0)
        # Keep stage as-is; difficulty is used only as an informational scalar for now.

    def reset(self):
        self._apply_curriculum()
        return super().reset()

    def _apply_curriculum(self) -> None:
        stage_cfg = self._stages[self._stage]

        # 1) Robot base spawn deviations (Kitchen env attributes)
        for attr, val in (
            ("robot_spawn_deviation_pos_x", stage_cfg.robot_pos_x),
            ("robot_spawn_deviation_pos_y", stage_cfg.robot_pos_y),
            ("robot_spawn_deviation_rot", stage_cfg.robot_rot),
        ):
            if hasattr(self.env, attr):
                setattr(self.env, attr, float(val))

        # 2) Object placement knobs (env-specific hook)
        if hasattr(self.env, "set_curriculum") and callable(getattr(self.env, "set_curriculum")):
            try:
                self.env.set_curriculum(
                    stage=self._stage,
                    difficulty=self._difficulty,
                    obj_size_xy=stage_cfg.obj_size_xy,
                    obj_rot_range=stage_cfg.obj_rot_range,
                )
            except TypeError:
                # Backward-compatible if env implements a different signature
                self.env.set_curriculum(stage=self._stage, difficulty=self._difficulty)

