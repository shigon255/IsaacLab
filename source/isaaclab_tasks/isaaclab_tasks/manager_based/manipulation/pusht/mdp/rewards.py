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
from isaaclab.utils import math as math_utils

from .observations import t_overlap


def overlap_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
) -> torch.Tensor:
    return t_overlap(env, object_cfg=object_cfg, goal_xy=goal_xy, goal_yaw=goal_yaw)


def pose_distance_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
) -> torch.Tensor:
    block: RigidObject = env.scene[object_cfg.name]
    block_xy = block.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal = torch.tensor(goal_xy, device=env.device)
    xy_error = torch.linalg.norm(block_xy - goal, dim=-1)
    block_yaw = math_utils.euler_xyz_from_quat(block.data.root_quat_w)[2]
    yaw_error = torch.abs(math_utils.wrap_to_pi(block_yaw - goal_yaw))
    return xy_error + 0.05 * yaw_error


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.norm(env.action_manager.action, dim=-1)


def success_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
    threshold: float = 0.95,
) -> torch.Tensor:
    return (t_overlap(env, object_cfg=object_cfg, goal_xy=goal_xy, goal_yaw=goal_yaw) >= threshold).float()
