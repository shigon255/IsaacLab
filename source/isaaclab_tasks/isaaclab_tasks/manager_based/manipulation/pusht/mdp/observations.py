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


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    return math_utils.euler_xyz_from_quat(quat)[2]


def _t_sample_points(
    env: ManagerBasedRLEnv,
    bar_size: tuple[float, float] = (0.24, 0.08),
    stem_size: tuple[float, float] = (0.08, 0.24),
    bar_offset: tuple[float, float] = (0.0, 0.08),
    stem_offset: tuple[float, float] = (0.0, -0.04),
    sample_spacing: float = 0.012,
) -> torch.Tensor:
    cache_key = "_pusht_t_sample_points"
    if hasattr(env, cache_key):
        return getattr(env, cache_key)

    half_x = max(bar_size[0], stem_size[0]) * 0.5
    min_y = stem_offset[1] - stem_size[1] * 0.5
    max_y = bar_offset[1] + bar_size[1] * 0.5
    xs = torch.arange(-half_x, half_x + sample_spacing, sample_spacing, device=env.device)
    ys = torch.arange(min_y, max_y + sample_spacing, sample_spacing, device=env.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    in_bar = (torch.abs(pts[:, 0] - bar_offset[0]) <= bar_size[0] * 0.5) & (
        torch.abs(pts[:, 1] - bar_offset[1]) <= bar_size[1] * 0.5
    )
    in_stem = (torch.abs(pts[:, 0] - stem_offset[0]) <= stem_size[0] * 0.5) & (
        torch.abs(pts[:, 1] - stem_offset[1]) <= stem_size[1] * 0.5
    )
    pts = pts[in_bar | in_stem]
    setattr(env, cache_key, pts)
    return pts


def _points_in_t(
    pts: torch.Tensor,
    bar_size: tuple[float, float],
    stem_size: tuple[float, float],
    bar_offset: tuple[float, float],
    stem_offset: tuple[float, float],
) -> torch.Tensor:
    in_bar = (torch.abs(pts[..., 0] - bar_offset[0]) <= bar_size[0] * 0.5) & (
        torch.abs(pts[..., 1] - bar_offset[1]) <= bar_size[1] * 0.5
    )
    in_stem = (torch.abs(pts[..., 0] - stem_offset[0]) <= stem_size[0] * 0.5) & (
        torch.abs(pts[..., 1] - stem_offset[1]) <= stem_size[1] * 0.5
    )
    return in_bar | in_stem


def t_overlap(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
    bar_size: tuple[float, float] = (0.24, 0.08),
    stem_size: tuple[float, float] = (0.08, 0.24),
    bar_offset: tuple[float, float] = (0.0, 0.08),
    stem_offset: tuple[float, float] = (0.0, -0.04),
) -> torch.Tensor:
    """Approximate T-on-T overlap by sampling points in the block T shape."""

    block: RigidObject = env.scene[object_cfg.name]
    pts = _t_sample_points(env, bar_size, stem_size, bar_offset, stem_offset)
    block_xy = block.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    block_yaw = _yaw_from_quat(block.data.root_quat_w)

    cos_b = torch.cos(block_yaw).unsqueeze(-1)
    sin_b = torch.sin(block_yaw).unsqueeze(-1)
    local_x = pts[:, 0].unsqueeze(0)
    local_y = pts[:, 1].unsqueeze(0)
    world_x = block_xy[:, 0:1] + cos_b * local_x - sin_b * local_y
    world_y = block_xy[:, 1:2] + sin_b * local_x + cos_b * local_y

    goal = torch.tensor(goal_xy, device=env.device)
    dx = world_x - goal[0]
    dy = world_y - goal[1]
    cos_g = math.cos(-goal_yaw)
    sin_g = math.sin(-goal_yaw)
    goal_local = torch.stack((cos_g * dx - sin_g * dy, sin_g * dx + cos_g * dy), dim=-1)
    inside = _points_in_t(goal_local, bar_size, stem_size, bar_offset, stem_offset)
    return inside.float().mean(dim=1)


def state_obs(
    env: ManagerBasedRLEnv,
    pusher_cfg: SceneEntityCfg = SceneEntityCfg("pusher"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
) -> torch.Tensor:
    """Low-dimensional Push-T state used alongside image observations."""

    pusher: RigidObject = env.scene[pusher_cfg.name]
    block: RigidObject = env.scene[object_cfg.name]

    pusher_xy = pusher.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    pusher_vel_xy = pusher.data.root_lin_vel_w[:, :2]
    block_xy = block.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    block_yaw = _yaw_from_quat(block.data.root_quat_w)
    block_yaw_sincos = torch.stack((torch.sin(block_yaw), torch.cos(block_yaw)), dim=-1)
    block_vel_xy = block.data.root_lin_vel_w[:, :2]
    block_yaw_vel = block.data.root_ang_vel_w[:, 2:3]
    goal = torch.tensor(goal_xy, device=env.device).repeat(env.num_envs, 1)
    goal_yaw_tensor = torch.full((env.num_envs,), goal_yaw, device=env.device)
    goal_yaw_sincos = torch.stack((torch.sin(goal_yaw_tensor), torch.cos(goal_yaw_tensor)), dim=-1)
    overlap = t_overlap(env, object_cfg=object_cfg, goal_xy=goal_xy, goal_yaw=goal_yaw).unsqueeze(-1)

    return torch.cat(
        (
            pusher_xy,
            pusher_vel_xy,
            block_xy,
            block_yaw_sincos,
            block_vel_xy,
            block_yaw_vel,
            goal,
            goal_yaw_sincos,
            overlap,
        ),
        dim=-1,
    )
