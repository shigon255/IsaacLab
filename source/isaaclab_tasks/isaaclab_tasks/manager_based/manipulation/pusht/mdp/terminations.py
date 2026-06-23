# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import t_overlap


def success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
    threshold: float = 0.95,
) -> torch.Tensor:
    return t_overlap(env, object_cfg=object_cfg, goal_xy=goal_xy, goal_yaw=goal_yaw) >= threshold


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    xy_limit: float = 0.55,
    min_height: float = -0.02,
) -> torch.Tensor:
    block: RigidObject = env.scene[object_cfg.name]
    block_pos = block.data.root_pos_w - env.scene.env_origins
    xy_out = torch.any(torch.abs(block_pos[:, :2]) > xy_limit, dim=-1)
    z_out = block_pos[:, 2] < min_height
    return xy_out | z_out
