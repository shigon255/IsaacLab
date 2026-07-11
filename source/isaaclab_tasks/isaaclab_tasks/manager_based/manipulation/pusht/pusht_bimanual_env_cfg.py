# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual Push-T environment: two ALOHA/ViperX-300s arms facing each other, planar motion, top-down view.

Robot placement geometry
------------------------
Two ViperX 300s arms (the ALOHA follower arms) are placed on opposite sides of the table:

  robot_left  : base at (-0.55, 0.0, 0.0), yaw = 0   → faces +X, appears screen-left
  robot_right : base at (+0.55, 0.0, 0.0), yaw = 180° → faces -X, appears screen-right

Camera nudge in -Y resolves: screen-right = +X, screen-up = +Y.
WASD/IJKL key layout aligns with this orientation.

Joint names (after MJCF→USD sanitisation, '/' → '_'):
  Arm joints : waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
  EE body    : gripper_link
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.aloha import VX300S_HIGH_PD_CFG

from . import mdp

# ---- Shared constants (same as single-arm scene) ----
from .pusht_env_cfg import TABLE_TOP_Z, BLOCK_HEIGHT, OBJECT_Z, GOAL_XY, GOAL_YAW

# Quaternion for 180° yaw around Z: (w=0, x=0, y=0, z=1)
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)


@configclass
class PushTBimanualSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Push-T scene using two ViperX 300s (ALOHA follower) arms.

    robot_left  : base at (-0.55, 0, 0), yaw=0  → faces +X, appears screen-left.
    robot_right : base at (+0.55, 0, 0), yaw=180° → faces -X, appears screen-right.
    """

    # ---- Left ViperX arm (ALOHA) ----
    robot_left: ArticulationCfg = VX300S_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotLeft",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # yaw = 0, faces +X
            joint_pos={
                "waist": 0.0,
                "shoulder": -0.96,
                "elbow": 1.16,
                "forearm_roll": 0.0,
                "wrist_angle": -0.3,
                "wrist_rotate": 0.0,
                "left_finger": 0.035,
                "right_finger": -0.035,
            },
        ),
    )

    # ---- Right ViperX arm (ALOHA) ----
    robot_right: ArticulationCfg = VX300S_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotRight",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=_ROT_180_Z,  # yaw = 180°, faces -X
            joint_pos={
                "waist": 0.0,
                "shoulder": -0.96,
                "elbow": 1.16,
                "forearm_roll": 0.0,
                "wrist_angle": -0.3,
                "wrist_rotate": 0.0,
                "left_finger": 0.035,
                "right_finger": -0.035,
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

    t_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TBlock",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.0, OBJECT_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=mdp.PushTShapeCfg(
            func=mdp.spawn_t_shape,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.15,
                angular_damping=0.15,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.20, 0.08), roughness=0.55),
        ),
    )

    goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(GOAL_XY[0], GOAL_XY[1], TABLE_TOP_Z + 0.004),
            rot=(0.92388, 0.0, 0.0, 0.38268),
        ),
        spawn=mdp.PushTShapeCfg(
            func=mdp.spawn_t_shape,
            bar_size=(0.24, 0.08, 0.008),
            stem_size=(0.08, 0.24, 0.008),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.85, 0.22), opacity=0.55, roughness=0.7),
        ),
    )

    fixed_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FixedCamera",
        update_period=0.0,
        height=128,
        width=128,
        # distance_to_image_plane (not distance_to_camera): perspective/plane-Z depth,
        # matching Genesis's depth semantics (linear camera-space Z, not Euclidean range) —
        # required for the isaac_backend condition bridge (phys-vidsim's FrameContext.depth).
        data_types=["rgb", "instance_id_segmentation_fast", "distance_to_image_plane"],
        # colorize=False → raw uint32 prim-id image used by edge_image for arm contours.
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        # Top-down: ROS convention, rot=(0,1,0,0) = 180° around X → lens points -world-Z.
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
    """Planar (2-DoF) action for each arm.  Action vector = [vx_L, vy_L, vx_R, vy_R]."""

    # enable_z=False: planar-only; EE height fixed at ee_z_height.
    # subtract_frame_transforms handles the 180°-rotated right arm automatically.
    pusher_left = mdp.FrankaEEPusherActionCfg(
        asset_name="robot_left",
        arm_joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        ee_body_name="gripper_link",
        # gripper_link centre at lower quarter of T-block: TABLE_TOP_Z + BLOCK_HEIGHT * 0.25 = 0.01
        ee_z_height=TABLE_TOP_Z + BLOCK_HEIGHT * 0.25,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.005, 0.40),
        workspace=((-0.42, -0.25), (0.12, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(-0.25, 0.0),
        enable_z=True,
    )

    pusher_right = mdp.FrankaEEPusherActionCfg(
        asset_name="robot_right",
        arm_joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        ee_body_name="gripper_link",
        # gripper_link centre at lower quarter of T-block: TABLE_TOP_Z + BLOCK_HEIGHT * 0.25 = 0.01
        ee_z_height=TABLE_TOP_Z + BLOCK_HEIGHT * 0.25,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.005, 0.40),
        workspace=((-0.12, -0.25), (0.42, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(0.25, 0.0),
        enable_z=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        fixed_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("fixed_cam"), "data_type": "rgb", "normalize": False},
        )
        edge_cam = ObsTerm(
            func=mdp.edge_image,
            params={
                "object_cfg": SceneEntityCfg("t_block"),
                "camera_cfg": SceneEntityCfg("fixed_cam"),
                "goal_xy": GOAL_XY,
                "goal_yaw": GOAL_YAW,
            },
        )
        state = ObsTerm(
            func=mdp.bimanual_ee_state_obs,
            params={
                "robot_cfg": SceneEntityCfg("robot_left"),
                "left_ee_body_name": "gripper_link",
                "right_robot_cfg": SceneEntityCfg("robot_right"),
                "right_ee_body_name": "gripper_link",
                "object_cfg": SceneEntityCfg("t_block"),
                "goal_xy": GOAL_XY,
                "goal_yaw": GOAL_YAW,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_left = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left"),
        },
    )
    reset_robot_right = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_right"),
        },
    )
    reset_block = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.12, 0.08), "y": (-0.16, 0.16), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("t_block"),
        },
    )


@configclass
class RewardsCfg:
    overlap = RewTerm(
        func=mdp.overlap_reward,
        weight=4.0,
        params={"object_cfg": SceneEntityCfg("t_block"), "goal_xy": GOAL_XY, "goal_yaw": GOAL_YAW},
    )
    pose_distance = RewTerm(
        func=mdp.pose_distance_penalty,
        weight=-1.0,
        params={"object_cfg": SceneEntityCfg("t_block"), "goal_xy": GOAL_XY, "goal_yaw": GOAL_YAW},
    )
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.01)
    success = RewTerm(
        func=mdp.success_bonus,
        weight=10.0,
        params={"object_cfg": SceneEntityCfg("t_block"), "goal_xy": GOAL_XY, "goal_yaw": GOAL_YAW, "threshold": 0.95},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds,
        params={"object_cfg": SceneEntityCfg("t_block"), "xy_limit": 0.55, "min_height": -0.02},
    )
    success = DoneTerm(
        func=mdp.success,
        params={"object_cfg": SceneEntityCfg("t_block"), "goal_xy": GOAL_XY, "goal_yaw": GOAL_YAW, "threshold": 0.95},
    )


@configclass
class PushTBimanualEnvCfg(ManagerBasedRLEnvCfg):
    scene: PushTBimanualSceneCfg = PushTBimanualSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.num_rerenders_on_reset = 2
        self.image_obs_list = ["fixed_cam"]
        # Nudge eye in -Y: screen-right = +X, screen-up = +Y.
        # Left arm (-X) appears screen-left; right arm (+X) appears screen-right.
        self.viewer = ViewerCfg(
            eye=(0.0, -0.001, 1.8),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se2KeyboardCfg(
                    v_x_sensitivity=0.6,
                    v_y_sensitivity=0.6,
                    omega_z_sensitivity=0.0,
                    sim_device=self.sim.device,
                )
            }
        )
