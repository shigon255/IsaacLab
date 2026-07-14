# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math as _math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
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


class FrankaEEPusherAction(ActionTerm):
    """Drive the Franka arm end-effector in the table XY plane to push the T-block.

    Maintains a target EE world-XY position that is integrated from SE(2) keyboard commands.
    The keyboard command is rotated by ``control_yaw_offset`` radians so that pressing
    "forward" moves the EE toward the top of the angled viewport camera.

    On each step the target XY + fixed ``ee_z_height`` is expressed in the robot base frame
    and sent to a differential IK controller (position-only, orientation unconstrained)
    which computes arm joint targets.  The Franka's physical fingers push the T-block directly.
    """

    cfg: "FrankaEEPusherActionCfg"
    _asset: Articulation  # the Franka robot

    def __init__(self, cfg: "FrankaEEPusherActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Integrated EE target in world-frame XY (initialised in reset before first use)
        self._target_xy = torch.zeros(env.num_envs, 2, device=self.device)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._workspace_low = torch.tensor(cfg.workspace[0], device=self.device)
        self._workspace_high = torch.tensor(cfg.workspace[1], device=self.device)

        # camera-alignment rotation coefficients (can be updated live via set_control_yaw)
        theta = cfg.control_yaw_offset
        self._rot_cos = _math.cos(theta)
        self._rot_sin = _math.sin(theta)

        # runtime speed multiplier (updated via set_speed_scale)
        self._speed_scale: float = 1.0

        # resolve arm joints
        self._arm_joint_ids, _ = self._asset.find_joints(cfg.arm_joint_names)
        if self._asset.num_joints == len(self._arm_joint_ids):
            self._arm_joint_ids = slice(None)

        # resolve EE body
        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body match for ee_body_name='{cfg.ee_body_name}', "
                f"got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]

        # Jacobian indexing — fixed-base arms offset body idx by 1
        if self._asset.is_fixed_base:
            self._jac_body_idx = self._ee_body_idx - 1
            self._jac_joint_ids = self._arm_joint_ids
        else:
            self._jac_body_idx = self._ee_body_idx
            self._jac_joint_ids = (
                [i + 6 for i in self._arm_joint_ids]
                if isinstance(self._arm_joint_ids, list)
                else self._arm_joint_ids
            )

        # position-only IK by default (orientation unconstrained, avoiding singularity issues);
        # cfg.lock_orientation switches to "pose" (position AND orientation constrained) instead.
        self._ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose" if cfg.lock_orientation else "position",
                use_relative_mode=False,
                ik_method="dls",
            ),
            num_envs=env.num_envs,
            device=self.device,
        )
        self._lock_orientation = cfg.lock_orientation
        self._home_quat_b: torch.Tensor | None = None

        # Z-axis target (per-env, mutable — integrated from the 3rd action channel)
        self._target_z = torch.full((env.num_envs,), cfg.ee_z_height, device=self.device)
        self._z_ws_low = cfg.z_workspace[0]
        self._z_ws_high = cfg.z_workspace[1]

        # planar-only flag — when False, action_dim=2 and Z is held fixed at ee_z_height
        self._enable_z: bool = cfg.enable_z

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 3 if self._enable_z else 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Runtime control API (called by the teleop script each frame)
    # ------------------------------------------------------------------

    def set_control_yaw(self, theta: float) -> None:
        """Update the camera-alignment yaw offset live (radians).

        The teleop script calls this every step with the live viewport camera
        azimuth so that pressing "forward" always moves the EE toward the top
        of the current screen view, regardless of how the camera was moved.
        """
        self._rot_cos = _math.cos(theta)
        self._rot_sin = _math.sin(theta)

    def set_speed_scale(self, s: float) -> None:
        """Set the runtime speed multiplier (clamped to [0.05, 8.0]).

        Multiplied into ``velocity_scale`` at action-processing time so the
        EE moves faster or slower without touching the base cfg.
        """
        self._speed_scale = max(0.05, min(8.0, float(s)))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:, :2] = actions[:, :2]
        # rotate keyboard command into camera-aligned world directions
        vx = self._raw_actions[:, 0]
        vy = self._raw_actions[:, 1]
        self._processed_actions[:, 0] = vx * self._rot_cos - vy * self._rot_sin
        self._processed_actions[:, 1] = vx * self._rot_sin + vy * self._rot_cos
        # clamp and scale XY (speed_scale multiplied after clamping so it gives a real range)
        self._processed_actions[:, :2] = torch.clamp(self._processed_actions[:, :2], -1.0, 1.0)
        self._processed_actions[:, :2] *= self.cfg.velocity_scale * self._speed_scale

        # Integrate target EE position in world XY and clamp to workspace
        self._target_xy += self._processed_actions[:, :2] * self._env.physics_dt
        self._target_xy[:] = torch.max(
            torch.min(self._target_xy, self._workspace_high), self._workspace_low
        )

        if self._enable_z:
            # 3rd channel: vertical (Z) velocity — Z/X keys move EE up/down
            self._raw_actions[:, 2] = actions[:, 2]
            self._processed_actions[:, 2] = torch.clamp(self._raw_actions[:, 2], -1.0, 1.0)
            self._processed_actions[:, 2] *= self.cfg.z_velocity_scale * self._speed_scale
            # Integrate Z target and clamp to z workspace
            self._target_z += self._processed_actions[:, 2] * self._env.physics_dt
            self._target_z.clamp_(self._z_ws_low, self._z_ws_high)

    def apply_actions(self):
        # Build 3-D target position in world frame
        target_pos_w = torch.zeros(self._env.num_envs, 3, device=self.device)
        target_pos_w[:, :2] = self._target_xy
        target_pos_w[:, 2] = self._target_z

        # Express target and current EE in the robot base frame
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        target_pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w)
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        # Set absolute position (or position+orientation, if locked) command and compute joint
        # targets via differential IK.
        if self._lock_orientation:
            command = torch.cat([target_pos_b, self._home_quat_b], dim=-1)
        else:
            command = target_pos_b
        self._ik_controller.set_command(command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        jac_w = self._asset.root_physx_view.get_jacobians()[:, self._jac_body_idx, :, self._jac_joint_ids]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot_matrix, jac_b[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot_matrix, jac_b[:, 3:, :])

        joint_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        if ee_quat_b.norm() != 0:
            joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jac_b, joint_pos)
            self._asset.set_joint_position_target(joint_pos_des, self._arm_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Reset target to the default env-local XY + each env's world origin
        origins = self._env.scene.env_origins[env_ids, :2]
        default_local = torch.tensor(self.cfg.default_target_xy, device=self.device)
        self._target_xy[env_ids] = origins + default_local

        # Reset Z to the configured default height
        self._target_z[env_ids] = self.cfg.ee_z_height

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)

        if self._lock_orientation:
            # Capture the EE's current base-frame orientation once per reset, held fixed
            # thereafter — the Isaac equivalent of Genesis's ArmController._home_quat. Mirrors
            # apply_actions()'s own subtract_frame_transforms() call exactly (same four
            # arguments: root pos/quat, then the body's own world pos/quat) rather than reusing
            # root_pos_w as a stand-in for the body position.
            root_pos_w = self._asset.data.root_pos_w[env_ids]
            root_quat_w = self._asset.data.root_quat_w[env_ids]
            ee_pos_w = self._asset.data.body_pos_w[env_ids, self._ee_body_idx]
            ee_quat_w = self._asset.data.body_quat_w[env_ids, self._ee_body_idx]
            _, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            if self._home_quat_b is None:
                self._home_quat_b = torch.zeros(self._env.num_envs, 4, device=self.device)
            self._home_quat_b[env_ids] = ee_quat_b


@configclass
class FrankaEEPusherActionCfg(ActionTermCfg):
    """Configuration for the Franka end-effector planar pusher."""

    class_type: type[ActionTerm] = FrankaEEPusherAction

    asset_name: str = "robot"
    """Name of the Franka Articulation in the scene."""

    arm_joint_names: list[str] = MISSING
    """Regex list selecting the arm joints to control (e.g. ['panda_joint.*'])."""

    ee_body_name: str = "panda_hand"
    """Name of the end-effector body on the Franka articulation."""

    ee_z_height: float = 0.15
    """Fixed world-frame Z the Franka EE tracks (metres above table surface)."""

    velocity_scale: float = 0.35
    """Scaling applied to the xy velocity after rotation (m/s per normalised unit)."""

    yaw_rate_scale: float = 0.0
    """Unused — kept for API compatibility. The 3rd action channel is repurposed as vz."""

    z_velocity_scale: float = 0.35
    """Scaling applied to the Z (vertical) velocity channel (m/s per normalised unit)."""

    z_workspace: tuple[float, float] = (0.05, 0.40)
    """(min_z, max_z) clamp for the EE height target (world frame, metres)."""

    workspace: tuple[tuple[float, float], tuple[float, float]] = ((-0.38, -0.25), (0.20, 0.25))
    """(min_xy, max_xy) workspace bounds in world XY."""

    control_yaw_offset: float = 0.0
    """Initial rotation (rad) applied to the raw (vx, vy) command before integration.

    Leave at 0.0 for training (the policy emits world-frame commands).
    The teleop script overrides this live each frame via ``set_control_yaw()``
    so that pressing "forward" always moves the EE toward the top of the
    current viewport camera view, regardless of how the camera was moved.
    """

    default_target_xy: tuple[float, float] = (-0.25, 0.0)
    """Initial EE target position in env-local XY on each episode reset."""

    enable_z: bool = True
    """If True (default) the 3rd action channel controls EE height (Z).
    If False the term is planar-only: ``action_dim`` becomes 2, the 3rd
    action channel is ignored, and the EE height stays pinned at
    ``ee_z_height`` for the entire episode.  The single-arm Push-T scene
    leaves this at the default True."""

    lock_orientation: bool = False
    """If True, fully constrain the EE's orientation to whatever it was at the last reset()
    (via the IK controller's "pose" command type instead of "position") — eliminates IK
    null-space orientation drift during pure position-tracking motion, at the cost of the EE
    never being able to rotate at all. False (default) reproduces the original behavior exactly:
    orientation left fully unconstrained for IK."""
