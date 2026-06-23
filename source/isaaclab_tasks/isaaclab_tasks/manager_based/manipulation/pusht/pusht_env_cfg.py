# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp


TABLE_TOP_Z = 0.0
BLOCK_HEIGHT = 0.04
OBJECT_Z = TABLE_TOP_Z + BLOCK_HEIGHT * 0.5 + 0.002
GOAL_XY = (0.16, 0.02)
GOAL_YAW = math.pi / 4.0


@configclass
class PushTSceneCfg(InteractiveSceneCfg):
    """Tabletop Push-T scene."""

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

    pusher = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pusher",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.0, OBJECT_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=BLOCK_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.20, 0.95), roughness=0.45),
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
    pusher = mdp.PlanarPusherActionCfg(
        asset_name="pusher",
        velocity_scale=0.35,
        yaw_rate_scale=0.0,
        z_height=OBJECT_Z,
        workspace=((-0.45, -0.45), (0.45, 0.45)),
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
            func=mdp.state_obs,
            params={
                "pusher_cfg": SceneEntityCfg("pusher"),
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
    reset_pusher = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.35, -0.25), "y": (-0.15, 0.15), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pusher"),
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
