# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Push-T environment (phys-vidsim robodojo-scene-collection #18, M2) --
RoboDojo-derived analogue of ../pusht/pusht_bimanual_env_cfg.py, with X5_HIGH_PD_CFG
(isaaclab_assets.robots.x5) replacing ALOHA's VX300S_HIGH_PD_CFG.

Robot placement (world XY, empirically derived on the Genesis side --
simulation/robodojo_pusht/scene.py's module docstring -- and reused here unchanged
since the underlying robot geometry is identical, same source URDF):

  robot_left  : base at (-0.35, 0.0, 0.0), yaw = 180 deg -> reach converges toward +X
  robot_right : base at (+0.35, 0.0, 0.0), yaw = 0        -> reach converges toward -X

Both arms' unrotated home reach points toward -X in their own local frame (confirmed
empirically via simulation/sim_common/robodojo_x5.py's smoke test) -- NOT
mirror-symmetric under identical mounting the way ALOHA's own vx300s_left/right.xml
fragments already are, hence robot_left needing the extra 180 deg yaw and robot_right
needing none (asymmetric, unlike Push-T's ALOHA rig).

Unlike Push-T's push-only ALOHA arms, X5 has a real prismatic gripper (joint7/joint8) --
this env wires it via IsaacLab core's BinaryJointPositionActionCfg (same mechanism
../franka_stack/franka_stack_env_cfg.py uses for the Panda), open/close bounds matching
X5A.urdf's own [0, 0.044] stroke range exactly.

T-block object reuses ../pusht/mdp's own PushTShapeCfg/spawn_t_shape (the SAME
procedural T both aloha_pusht's Isaac scene and Genesis's aloha_pusht/robodojo_pusht
scenes use) rather than inventing new geometry -- keeps the T-block pixel/vertex-
identical across every scene on this backend.

Joint names (X5A.urdf, unprefixed -- each side is its own separate Articulation prim,
unlike Genesis's single composed-URDF entity; see simulation/sim_common/robodojo_x5.py's
docstring for why Genesis needed the composition trick and Isaac doesn't):
  Arm joints   : joint1, joint2, joint3, joint4, joint5, joint6
  Finger joints: joint7, joint8
  EE body      : link6 (no dedicated gripper_link -- fingers attach directly to link6,
                 same convention the Genesis side uses)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as base_mdp
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
from isaaclab_assets.robots.x5 import X5_HIGH_PD_CFG

from ..pusht import mdp
from ..pusht.mdp.actions import FrankaEEPusherActionCfg
from ..pusht.pusht_env_cfg import BLOCK_HEIGHT, GOAL_XY, GOAL_YAW, OBJECT_Z, TABLE_TOP_Z

# 180 deg yaw around Z, (w,x,y,z) -- same encoding pusht_bimanual_env_cfg.py's _ROT_180_Z uses.
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_FINGER_JOINT_NAMES = ["joint7", "joint8"]
_EE_BODY_NAME = "link6"


@configclass
class RobodojoPushTSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Push-T scene using two ARX X5 arms."""

    robot_left: ArticulationCfg = X5_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotLeft",
        init_state=X5_HIGH_PD_CFG.init_state.replace(
            pos=(-0.35, 0.0, 0.0),
            rot=_ROT_180_Z,
        ),
    )

    robot_right: ArticulationCfg = X5_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotRight",
        init_state=X5_HIGH_PD_CFG.init_state.replace(
            pos=(0.35, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.15, OBJECT_Z), rot=(1.0, 0.0, 0.0, 0.0)),
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
        # See ../pusht/pusht_bimanual_env_cfg.py's identical field for the full
        # rationale -- required for isaac_backend's runtime camera repositioning to work.
        update_latest_camera_pose=True,
        # 512x512, matching Genesis's convention (see ../pusht/pusht_bimanual_env_cfg.py).
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
    """Arm EE-velocity pusher (position+orientation-locked differential IK, full XYZ) +
    binary gripper per side -- X5 has a real prismatic gripper, unlike Push-T's
    push-only ALOHA arms, so this reuses franka_stack's gripper-wiring pattern rather
    than pusht_bimanual's push-only ActionsCfg."""

    pusher_left = FrankaEEPusherActionCfg(
        asset_name="robot_left",
        arm_joint_names=_ARM_JOINT_NAMES,
        ee_body_name=_EE_BODY_NAME,
        ee_z_height=OBJECT_Z,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.01, 0.40),
        workspace=((-0.35, -0.25), (0.15, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(-0.15, 0.15),
        enable_z=True,
        lock_orientation=True,
    )

    gripper_left = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=_FINGER_JOINT_NAMES,
        open_command_expr={"joint7": 0.044, "joint8": 0.044},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )

    pusher_right = FrankaEEPusherActionCfg(
        asset_name="robot_right",
        arm_joint_names=_ARM_JOINT_NAMES,
        ee_body_name=_EE_BODY_NAME,
        ee_z_height=OBJECT_Z,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.01, 0.40),
        workspace=((-0.15, -0.25), (0.35, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(0.15, 0.15),
        enable_z=True,
        lock_orientation=True,
    )

    gripper_right = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=_FINGER_JOINT_NAMES,
        open_command_expr={"joint7": 0.044, "joint8": 0.044},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
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
    reset_block = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("t_block")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds,
        params={"object_cfg": SceneEntityCfg("t_block"), "xy_limit": 0.55, "min_height": -0.02},
    )
    success = DoneTerm(
        func=mdp.success,
        params={"object_cfg": SceneEntityCfg("t_block"), "goal_xy": GOAL_XY, "goal_yaw": GOAL_YAW, "threshold": 0.95},
    )


@configclass
class RobodojoPushTEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoPushTSceneCfg = RobodojoPushTSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
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
        self.num_rerenders_on_reset = 2
        self.image_obs_list = ["fixed_cam"]
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
