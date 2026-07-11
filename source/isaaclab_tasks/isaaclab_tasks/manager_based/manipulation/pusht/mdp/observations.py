# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

# ---------------------------------------------------------------------------
# T-block outline (local frame) — 8 ordered vertices of the T polygon.
# Derived from PushTShapeCfg defaults:
#   bar  : center (0, 0.08), half-extents (0.12, 0.04)
#   stem : center (0,-0.04), half-extents (0.04, 0.12)
# ---------------------------------------------------------------------------
_T_VERTS_LOCAL = [
    (-0.12,  0.12),  # bar  top-left
    ( 0.12,  0.12),  # bar  top-right
    ( 0.12,  0.04),  # bar  bottom-right
    ( 0.04,  0.04),  # inner corner right
    ( 0.04, -0.16),  # stem bottom-right
    (-0.04, -0.16),  # stem bottom-left
    (-0.04,  0.04),  # inner corner left
    (-0.12,  0.04),  # bar  bottom-left
]


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


def bimanual_ee_state_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    left_ee_body_name: str = "gripper_link",
    right_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
    right_ee_body_name: str = "gripper_link",
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
) -> torch.Tensor:
    """Low-dimensional bimanual Push-T state with both EE positions.

    Supports two separate arm articulations (e.g. two ViperX / ALOHA arms).
    ``robot_cfg`` is the left arm, ``right_robot_cfg`` is the right arm.

    Returns a concatenation of:
    ``[left_ee_xy, left_ee_vel_xy, right_ee_xy, right_ee_vel_xy,
       block_xy, block_yaw_sincos, block_vel_xy, block_yaw_vel,
       goal_xy, goal_yaw_sincos, overlap]``
    """

    def _get_ee(robot: Articulation, cfg_name: str, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = f"_pusht_ee_body_idx_{cfg_name}_{body_name}"
        if not hasattr(env, cache_key):
            body_ids, _ = robot.find_bodies(body_name)
            setattr(env, cache_key, body_ids[0])
        idx = getattr(env, cache_key)
        xy = robot.data.body_pos_w[:, idx, :2] - env.scene.env_origins[:, :2]
        vel_xy = robot.data.body_lin_vel_w[:, idx, :2]
        return xy, vel_xy

    left_robot: Articulation = env.scene[robot_cfg.name]
    right_robot: Articulation = env.scene[right_robot_cfg.name]
    block: RigidObject = env.scene[object_cfg.name]

    left_ee_xy, left_ee_vel_xy = _get_ee(left_robot, robot_cfg.name, left_ee_body_name)
    right_ee_xy, right_ee_vel_xy = _get_ee(right_robot, right_robot_cfg.name, right_ee_body_name)

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
            left_ee_xy,
            left_ee_vel_xy,
            right_ee_xy,
            right_ee_vel_xy,
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


# ---------------------------------------------------------------------------
# Shared outline helpers (used by both edge_image obs and the live overlay)
# ---------------------------------------------------------------------------


def _get_outline_local(device: str | torch.device) -> torch.Tensor:
    """Return the cached (8, 2) T outline tensor in the block's local frame."""
    attr = f"_pusht_outline_local_{device}"
    if not hasattr(_get_outline_local, attr):
        pts = torch.tensor(_T_VERTS_LOCAL, dtype=torch.float32, device=device)
        setattr(_get_outline_local, attr, pts)
    return getattr(_get_outline_local, attr)


def t_outline_world(
    pos_w_xy: torch.Tensor,
    yaw: torch.Tensor,
    z: torch.Tensor | float,
    outline_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform the T outline polygon from block-local frame to world frame.

    Args:
        pos_w_xy: World XY of the T-block center(s). Shape ``(N, 2)``.
        yaw: Yaw angle of the block(s) in radians. Shape ``(N,)``.
        z: World Z for all outline vertices. Scalar or shape ``(N,)``.
        outline_local: ``(8, 2)`` local vertices; fetched from cache if ``None``.

    Returns:
        World-frame vertices. Shape ``(N, 8, 3)``.
    """
    if outline_local is None:
        outline_local = _get_outline_local(pos_w_xy.device)  # (8, 2)
    cos_y = torch.cos(yaw).unsqueeze(-1)  # (N, 1)
    sin_y = torch.sin(yaw).unsqueeze(-1)
    lx = outline_local[:, 0]  # (8,)
    ly = outline_local[:, 1]
    wx = pos_w_xy[:, 0:1] + cos_y * lx - sin_y * ly  # (N, 8)
    wy = pos_w_xy[:, 1:2] + sin_y * lx + cos_y * ly  # (N, 8)
    if isinstance(z, (int, float)):
        wz = torch.full_like(wx, float(z))
    else:
        wz = z.unsqueeze(-1).expand(-1, outline_local.shape[0])
    return torch.stack([wx, wy, wz], dim=-1)  # (N, 8, 3)


def edge_image(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("fixed_cam"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
    height: int = 128,
    width: int = 128,
) -> torch.Tensor:
    """Synthetic edge image derived from the explicit 3D scene state.

    Layers drawn (back to front):

    * **Table boundary** — gray rectangle (table top edge, 1 m × 1 m), clamped
      to the frame so the border always renders even when the projected corners
      sit at the very edge of the FOV.
    * **Goal T outline** — dim green polyline (static target pose).
    * **T-block outline** — white polyline (current block pose).
    * **Left arm contour** — cyan outline traced from the instance-id segmentation
      mask; exact pixel-level silhouette of the whole ``RobotLeft`` arm.
    * **Right arm contour** — orange outline, same method for ``RobotRight``.

    The arm contours require ``instance_id_segmentation_fast`` to be listed in
    ``CameraCfg.data_types`` with ``colorize_instance_id_segmentation=False``.
    If the segmentation data is not yet available (e.g. during the
    ``_prepare_terms`` probe call before the first render), the arm layer is
    skipped silently and the image still contains the geometric layers.

    Returns a ``uint8`` tensor of shape ``(num_envs, H, W, 3)``, matching the
    format produced by ``mdp.image(..., normalize=False)``.

    .. note::
        Uses a per-env cv2 rasterisation loop — not intended for large-scale
        parallel RL training.  Optimised for ``num_envs=1`` teleoperation and
        demonstration recording.
    """
    import cv2

    block: RigidObject = env.scene[object_cfg.name]
    cam = env.scene[camera_cfg.name]
    num_envs = env.num_envs
    device = env.device

    outline_local = _get_outline_local(device)  # (8, 2)

    # ---- T-block outline in world frame ----------------------------------------
    block_pos_w = block.data.root_pos_w              # (N, 3)
    block_yaw = _yaw_from_quat(block.data.root_quat_w)
    block_z = block_pos_w[:, 2] + 0.02              # top face = center_z + BLOCK_HEIGHT/2
    block_pts_w = t_outline_world(block_pos_w[:, :2], block_yaw, block_z, outline_local)  # (N, 8, 3)

    # ---- Static scene elements (goal + table) — cached after first call ---------
    if not hasattr(env, "_pusht_edge_static_pts"):
        origins_xy = env.scene.env_origins[:, :2]              # (N, 2)
        origins_z = env.scene.env_origins[:, 2]                # (N,)

        # Goal T outline (env-relative goal_xy → world)
        goal_pos_xy = origins_xy + torch.tensor(goal_xy, dtype=torch.float32, device=device)
        goal_yaw_t = torch.full((num_envs,), goal_yaw, device=device)
        goal_pts = t_outline_world(goal_pos_xy, goal_yaw_t, 0.004, outline_local)  # (N, 8, 3)

        # Table boundary — 4 corners of the 1 m × 1 m top face, in world coords
        corners_env = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32, device=device,
        )  # (4, 2)
        table_xy = origins_xy.unsqueeze(1) + corners_env.unsqueeze(0)   # (N, 4, 2)
        table_z = origins_z.unsqueeze(1).expand(-1, 4)                  # (N, 4)
        table_pts = torch.cat([table_xy, table_z.unsqueeze(-1)], dim=-1)  # (N, 4, 3)

        env._pusht_edge_static_pts = {"goal": goal_pts, "table": table_pts}  # type: ignore[attr-defined]

    static = env._pusht_edge_static_pts  # type: ignore[attr-defined]
    goal_pts_w: torch.Tensor = static["goal"]
    table_pts_w: torch.Tensor = static["table"]

    # ---- Camera intrinsics and world pose (per env) ----------------------------
    K = cam.data.intrinsic_matrices    # (N, 3, 3)
    cam_pos_w = cam.data.pos_w         # (N, 3)
    cam_quat_ros = cam.data.quat_w_ros  # (N, 4)

    def _project(pts_w: torch.Tensor) -> torch.Tensor:
        """Project ``(N, P, 3)`` world points to ``(N, P, 2)`` pixel coords (u, v).

        ``project_points`` squeezes dim-0 when N=1; the unsqueeze guard restores
        the batch dimension so callers always receive a 3-D tensor.
        """
        npts = pts_w.shape[1]
        p_rel = pts_w - cam_pos_w.unsqueeze(1)
        p_cam = math_utils.quat_apply_inverse(
            cam_quat_ros.unsqueeze(1).expand(-1, npts, -1),
            p_rel,
        )
        uvd = math_utils.project_points(p_cam, K)
        if uvd.dim() == 2:  # project_points squeezed the N=1 batch dim
            uvd = uvd.unsqueeze(0)
        return uvd[..., :2]

    block_uv = _project(block_pts_w)   # (N, 8, 2)
    goal_uv = _project(goal_pts_w)     # (N, 8, 2)
    table_uv = _project(table_pts_w)   # (N, 4, 2)

    # ---- Rasterize per env (back-to-front) -------------------------------------
    # Camera data is uninitialized during _prepare_terms (first probe call), so
    # projection may yield NaN/Inf.  Replace with 0 so cv2 draws nothing but the
    # output shape is correct.
    def _safe_np(t: torch.Tensor) -> np.ndarray:
        return np.nan_to_num(t.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    block_np = _safe_np(block_uv)
    goal_np = _safe_np(goal_uv)
    table_np = _safe_np(table_uv)

    # ---- Arm segmentation data (available only after first render) --------------
    seg_key = "instance_id_segmentation_fast"
    has_seg = (
        seg_key in cam.data.output
        and cam.data.output[seg_key] is not None
    )

    frames = []
    for e in range(num_envs):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Table boundary: bright gray, clamped so the border always renders even
        # when the true projected corners sit right at the frame edge.
        t_raw = table_np[e].round().astype(np.int32)      # (4, 2)
        t_raw[:, 0] = np.clip(t_raw[:, 0], 0, width - 1)
        t_raw[:, 1] = np.clip(t_raw[:, 1], 0, height - 1)
        cv2.polylines(img, [t_raw.reshape(-1, 1, 2)], isClosed=True, color=(120, 120, 120), thickness=2)

        # Goal T: dim green
        g_pts = goal_np[e].round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [g_pts], isClosed=True, color=(0, 180, 0), thickness=1)

        # T-block: white (slightly thicker so it reads over the goal)
        b_pts = block_np[e].round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [b_pts], isClosed=True, color=(255, 255, 255), thickness=2)

        # Arm silhouettes: contours extracted from instance-id segmentation.
        # idToLabels keys are string-ints; values are prim-path strings such as
        # "/World/envs/env_0/RobotLeft/shoulder_link".  Filter by arm name prefix.
        if has_seg:
            try:
                info = (cam.data.info[e] or {}).get(seg_key, {})
                id_to_label = info.get("idToLabels", {})
                seg_hw = cam.data.output[seg_key][e, ..., 0]  # (H, W)
                seg_i64 = seg_hw.to(torch.int64)
                for substr, color in [("RobotLeft", (0, 220, 255)), ("RobotRight", (255, 140, 0))]:
                    arm_ids = [int(k) for k, v in id_to_label.items() if substr in str(v)]
                    if arm_ids:
                        ids_t = torch.tensor(arm_ids, dtype=torch.int64, device=seg_i64.device)
                        mask = torch.isin(seg_i64, ids_t).cpu().numpy().astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, contours, -1, color, thickness=1)
            except Exception:
                pass  # silently skip if seg data not yet available

        frames.append(img)

    out = np.stack(frames, axis=0)  # (N, H, W, 3)
    return torch.tensor(out, dtype=torch.uint8, device=device)


def robot_ee_state_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
    object_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
    goal_xy: tuple[float, float] = (0.16, 0.02),
    goal_yaw: float = math.pi / 4.0,
) -> torch.Tensor:
    """Low-dimensional Push-T state using the robot EE position instead of a separate pusher object."""

    robot: Articulation = env.scene[robot_cfg.name]
    block: RigidObject = env.scene[object_cfg.name]

    # Cache EE body index to avoid repeated find_bodies calls
    cache_key = f"_pusht_ee_body_idx_{robot_cfg.name}_{ee_body_name}"
    if not hasattr(env, cache_key):
        body_ids, _ = robot.find_bodies(ee_body_name)
        setattr(env, cache_key, body_ids[0])
    ee_idx = getattr(env, cache_key)

    ee_xy = robot.data.body_pos_w[:, ee_idx, :2] - env.scene.env_origins[:, :2]
    ee_vel_xy = robot.data.body_lin_vel_w[:, ee_idx, :2]

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
            ee_xy,
            ee_vel_xy,
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
