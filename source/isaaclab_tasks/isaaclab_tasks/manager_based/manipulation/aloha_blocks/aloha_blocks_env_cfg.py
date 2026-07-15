# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ALOHA colored-blocks environment: reproduces the shape of phys-vidsim's
Genesis-side simulation/aloha_blocks/scene.py::BlocksScene (two ViperX 300s arms + 4
colored blocks) as an Isaac Lab manager-based env, for isaac-lab-multi-scene (#21)'s M1.

Robot placement: IDENTICAL to ../pusht/pusht_bimanual_env_cfg.py's two-VX300S rig
(robot_left at (-0.55,0,0) yaw=0, robot_right at (0.55,0,0) yaw=180deg, same table) --
no new asset provisioning needed, this scene reuses Push-T's own already-proven VX300S
placement verbatim (isaac-lab-multi-scene #21's plan.md flagged this as the reason M1
was chosen first: "aloha_blocks reuses the same VX300S asset as Push-T -- no new USD
provisioning gap").

Objects: 4 plain procedural CuboidCfg blocks (block_0..block_3, matching Genesis's own
simulation/aloha_blocks/cube.py::BLOCK_NAMES), 0.025m half-size, resting on the table
top (world Z=0, same convention as Push-T's t_block), arranged in a row at Y=0 (table
center) spanning X=[-0.15, 0.15] -- unlike Genesis's own block row (Y=0.50, ALOHA's own
off-center table convention), Isaac's table is centered at the origin, so blocks sit at
Y=0 instead; this is the same "physically distinct rig, not a pixel-identical twin"
precedent scenes/pusht.py's own module docstring already established.

Gripper: Genesis's BlocksScene.arm_specs() sets `functional_grip=True` on BOTH arms
(stacking needs real grasping, unlike Push-T's push-only design) -- so, unlike Push-T,
this env's ActionsCfg gives each arm a real gripper BinaryJointPositionActionCfg term
(targeting VX300S's own `left_finger`/`right_finger` joints, open/close values taken
from pusht_bimanual_env_cfg.py's own robot_left/robot_right init_state, which uses
(0.035,-0.035) as the resting/open pose) declared immediately after that arm's pusher
term, same pattern franka_stack_env_cfg.py established for its own single gripper.
Close values (0.0/0.0, fingers meeting at center) are a reasoned default, NOT confirmed
live -- verify grasping actually closes on a block via a real teleop session before
trusting it (see isaac_backend/scenes/aloha_blocks.py's matching note).

Camera: fixed_cam mirrors ../pusht/pusht_bimanual_env_cfg.py's CameraCfg EXACTLY (same
top-down rot=(0,1,0,0) static default, position directly above the table center like
Push-T's) -- the same "don't hand-derive an oblique quaternion, use the teleop
viewpoint-orbit's per-scene default instead" precedent franka_stack_env_cfg.py set;
isaac_backend/scenes/aloha_blocks.py's own DEFAULT_CAM_EYE/DEFAULT_CAM_LOOKAT (reusing
Push-T's proven oblique default verbatim, since this scene's object layout is
similarly-scaled/centered) supplies the actual oblique teleop/replay default.

No wrist-yaw control (matching franka_stack_env_cfg.py's own deferral) -- Genesis's own
BlocksScene.arm_specs() doesn't set enable_yaw either, so this isn't a scope gap versus
the actual Genesis reference, just consistent with it.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import mdp as base_mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.aloha import VX300S_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg

# ---- Shared constants -- mirror simulation/aloha_blocks/cube.py's own values exactly
# (kept in sync manually; see simulation/isaac_backend/scenes/aloha_blocks.py's matching
# comment -- there's no Genesis-free import path to that module since it pulls in
# `genesis` at module level). ----
_BLOCK_NAMES: tuple[str, ...] = ("block_0", "block_1", "block_2", "block_3")
_BLOCK_HALF_SIZE = 0.025
_BLOCK_MASS = 0.05
_BLOCK_RGBA: dict[str, tuple[float, float, float]] = {
    "block_0": (0.85, 0.15, 0.15), "block_1": (0.15, 0.70, 0.20),
    "block_2": (0.20, 0.35, 0.85), "block_3": (0.90, 0.80, 0.15),
}
# Isaac's OWN layout (Y=0, table-centered) -- see module docstring for why this
# differs from Genesis's own Y=0.50 (ALOHA's off-center table convention).
_BLOCK_INIT_XS: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
_BLOCK_INIT_Y = 0.0
_BLOCK_INIT_Z = _BLOCK_HALF_SIZE
_BLOCK_INIT_POS: dict[str, tuple[float, float, float]] = {
    name: (x, _BLOCK_INIT_Y, _BLOCK_INIT_Z) for name, x in zip(_BLOCK_NAMES, _BLOCK_INIT_XS)
}
_BLOCK_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity
_BLOCK_PRIM_NAME: dict[str, str] = {
    "block_0": "Block0", "block_1": "Block1", "block_2": "Block2", "block_3": "Block3",
}

# Quaternion for 180-degree yaw around Z -- matches pusht_bimanual_env_cfg.py's own
# _ROT_180_Z exactly (robot_right's placement).
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

# VX300S's own resting-pose finger values (pusht_bimanual_env_cfg.py's robot_left/
# robot_right init_state joint_pos) -- reused here as the "open" gripper command.
_FINGER_OPEN = {"left_finger": 0.035, "right_finger": -0.035}
_FINGER_CLOSE = {"left_finger": 0.0, "right_finger": 0.0}


def _block_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{_BLOCK_PRIM_NAME[name]}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BLOCK_INIT_POS[name], rot=_BLOCK_INIT_ROT),
        spawn=sim_utils.CuboidCfg(
            size=(_BLOCK_HALF_SIZE * 2, _BLOCK_HALF_SIZE * 2, _BLOCK_HALF_SIZE * 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=_BLOCK_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_BLOCK_RGBA[name], roughness=0.6),
        ),
    )


@configclass
class AlohaBlocksSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual colored-blocks scene using two ViperX 300s (ALOHA follower)
    arms -- same placement as PushTBimanualSceneCfg (../pusht/pusht_bimanual_env_cfg.py)."""

    robot_left: ArticulationCfg = VX300S_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotLeft",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": 0.0, "shoulder": -0.96, "elbow": 1.16,
                "forearm_roll": 0.0, "wrist_angle": -0.3, "wrist_rotate": 0.0,
                **_FINGER_OPEN,
            },
        ),
    )

    robot_right: ArticulationCfg = VX300S_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotRight",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=_ROT_180_Z,
            joint_pos={
                "waist": 0.0, "shoulder": -0.96, "elbow": 1.16,
                "forearm_roll": 0.0, "wrist_angle": -0.3, "wrist_rotate": 0.0,
                **_FINGER_OPEN,
            },
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.42, 0.38), roughness=0.8),
        ),
    )

    block_0 = _block_cfg("block_0")
    block_1 = _block_cfg("block_1")
    block_2 = _block_cfg("block_2")
    block_3 = _block_cfg("block_3")

    fixed_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FixedCamera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=512,
        width=512,
        data_types=["rgb", "instance_id_segmentation_fast", "distance_to_image_plane", "normals"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.90), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.08)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Per-arm EE-velocity pusher (reused as-is from Push-T, enable_z=True for full
    XYZ + lock_orientation=True for a fixed top-down grasp approach, matching Genesis's
    own BlocksScene not setting enable_yaw) + a real binary gripper per arm (unlike
    Push-T -- see module docstring)."""

    pusher_left = FrankaEEPusherActionCfg(
        asset_name="robot_left",
        arm_joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        ee_body_name="gripper_link",
        # 0.25 matches every sibling scene's own reset-hover convention (franka_stack/
        # ur5e_pickplace/kuka_nutassembly all use ee_z_height=0.25) -- well above the
        # 0.05m-tall blocks. This used to be _BLOCK_HALF_SIZE (0.025) with
        # default_target_xy=(-0.15, 0.0), which is EXACTLY block_0's own spawn position
        # (see _BLOCK_INIT_POS above): reset() snaps the IK target straight there
        # (actions.py's FrankaEEPusherAction.reset()), so the arm drove its gripper
        # directly into block_0 on every env reset before any teleop input. Confirmed
        # live via real teleop -- fixed by hovering above and offsetting sideways,
        # clear of the whole block row (X=[-0.15, 0.15]), same as franka_stack's own
        # cube-clearing default.
        ee_z_height=0.25,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.005, 0.40),
        workspace=((-0.42, -0.25), (0.12, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(-0.35, 0.0),
        enable_z=True,
        lock_orientation=True,
    )
    gripper_left = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["left_finger", "right_finger"],
        open_command_expr=_FINGER_OPEN,
        close_command_expr=_FINGER_CLOSE,
    )

    pusher_right = FrankaEEPusherActionCfg(
        asset_name="robot_right",
        arm_joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        ee_body_name="gripper_link",
        # See pusher_left's comment above -- same bug (default_target_xy=(0.15, 0.0) was
        # EXACTLY block_3's spawn position), same fix.
        ee_z_height=0.25,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.005, 0.40),
        workspace=((-0.12, -0.25), (0.42, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(0.35, 0.0),
        enable_z=True,
        lock_orientation=True,
    )
    gripper_right = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["left_finger", "right_finger"],
        open_command_expr=_FINGER_OPEN,
        close_command_expr=_FINGER_CLOSE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        fixed_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("fixed_cam"), "data_type": "rgb", "normalize": False},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_left = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_left")},
    )
    reset_robot_right = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_right")},
    )
    reset_block_0 = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_0")},
    )
    reset_block_1 = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_1")},
    )
    reset_block_2 = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_2")},
    )
    reset_block_3 = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_3")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class AlohaBlocksEnvCfg(ManagerBasedRLEnvCfg):
    scene: AlohaBlocksSceneCfg = AlohaBlocksSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # Matches phys-vidsim's Genesis-side cadence exactly -- see
        # ../pusht/pusht_bimanual_env_cfg.py's identical override for the full derivation.
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
