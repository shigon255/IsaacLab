# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp


TABLE_TOP_Z = 0.0
BLOCK_HEIGHT = 0.04
OBJECT_Z = TABLE_TOP_Z + BLOCK_HEIGHT * 0.5 + 0.002
GOAL_XY = (0.16, 0.02)
GOAL_YAW = math.pi / 4.0


@configclass
class PushTSceneCfg(InteractiveSceneCfg):
    """Tabletop Push-T scene."""

    # ---- Franka Panda arm ----
    # High-PD variant recommended for differential-IK task-space control.
    # Mounted just behind the table edge; fingers start closed.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.60, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.0,  # closed gripper
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
                # linear/angular damping left unset (None -> PhysX's own 0.0 default),
                # matching every other rigid object in this repo and the Genesis-side
                # tblock.xml (explicit damping="0"). A stray 0.15/0.15 override here was
                # the only non-zero damping in the whole cross-backend scene set, with no
                # recorded rationale (phys-vidsim physics-time-calibration #33 audit).
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(GOAL_XY[0], GOAL_XY[1], TABLE_TOP_Z + 0.004), rot=(0.92388, 0.0, 0.0, 0.38268)),
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
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.78), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
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
    # control_yaw_offset=0.0: for training the policy emits world-frame commands (no rotation).
    # For teleoperation, teleop_pusht.py reads the live viewport camera azimuth each frame
    # and calls pusher_term.set_control_yaw(azimuth) so "forward" always points up-screen
    # regardless of how the camera was moved (mouse or keyboard).
    pusher = mdp.FrankaEEPusherActionCfg(
        asset_name="robot",
        arm_joint_names=["panda_joint.*"],
        ee_body_name="panda_hand",
        ee_z_height=0.15,
        velocity_scale=0.035,
        yaw_rate_scale=0.0,
        z_velocity_scale=0.035,
        z_workspace=(0.05, 0.40),
        workspace=((-0.38, -0.25), (0.20, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(-0.25, 0.0),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        fixed_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("fixed_cam"), "data_type": "rgb", "normalize": False},
        )
        state = ObsTerm(
            func=mdp.robot_ee_state_obs,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_body_name": "panda_hand",
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
    # Reset the Franka arm to its default joint configuration each episode.
    # position_range=(1.0, 1.0) means "scale default by 1" = exact default; velocity = 0.
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
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
class PushTEnvCfg(ManagerBasedRLEnvCfg):
    scene: PushTSceneCfg = PushTSceneCfg(num_envs=16, env_spacing=1.2, replicate_physics=True)
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
        # Angled 3rd-person viewport: wide coverage of the whole table.
        # The eye/lookat are in the coordinate frame of env_0.
        # control_yaw_offset in ActionsCfg is set to atan2(0.9, 1.3) so that pressing
        # "forward" moves the pusher toward the top of this screen.
        self.viewer = ViewerCfg(
            eye=(-1.3, -0.9, 1.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se2KeyboardCfg(
                    v_x_sensitivity=0.6,
                    v_y_sensitivity=0.6,
                    omega_z_sensitivity=0.6,  # Z/X keys → EE up/down
                    sim_device=self.sim.device,
                )
            }
        )
