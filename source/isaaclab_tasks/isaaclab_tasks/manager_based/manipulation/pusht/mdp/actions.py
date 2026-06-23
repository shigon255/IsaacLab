# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PlanarPusherAction(ActionTerm):
    """Move a kinematic pusher in the tabletop XY plane from an SE(2) velocity command."""

    cfg: "PlanarPusherActionCfg"
    _asset: RigidObject

    def __init__(self, cfg: "PlanarPusherActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._root_pose = torch.zeros(env.num_envs, 7, device=self.device)
        self._root_velocity = torch.zeros(env.num_envs, 6, device=self.device)

        self._workspace_low = torch.tensor(cfg.workspace[0], device=self.device)
        self._workspace_high = torch.tensor(cfg.workspace[1], device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions[:, :3]
        self._processed_actions[:] = torch.clamp(self._raw_actions, -1.0, 1.0)
        self._processed_actions[:, :2] *= self.cfg.velocity_scale
        self._processed_actions[:, 2] *= self.cfg.yaw_rate_scale

    def apply_actions(self):
        root_state = self._asset.data.root_state_w
        self._root_pose[:] = root_state[:, :7]
        self._root_pose[:, 0:2] += self._processed_actions[:, 0:2] * self._env.physics_dt
        self._root_pose[:, 0:2] = torch.max(torch.min(self._root_pose[:, 0:2], self._workspace_high), self._workspace_low)
        self._root_pose[:, 2] = self.cfg.z_height
        self._root_pose[:, 3:7] = root_state[:, 3:7]

        self._root_velocity.zero_()
        self._root_velocity[:, 0:2] = self._processed_actions[:, 0:2]

        self._asset.write_root_pose_to_sim(self._root_pose)
        self._asset.write_root_velocity_to_sim(self._root_velocity)


@configclass
class PlanarPusherActionCfg(ActionTermCfg):
    """Configuration for planar pusher control."""

    class_type: type[ActionTerm] = PlanarPusherAction
    asset_name: str = MISSING
    velocity_scale: float = 0.35
    yaw_rate_scale: float = 0.0
    z_height: float = 0.025
    workspace: tuple[tuple[float, float], tuple[float, float]] = ((-0.45, -0.45), (0.45, 0.45))
