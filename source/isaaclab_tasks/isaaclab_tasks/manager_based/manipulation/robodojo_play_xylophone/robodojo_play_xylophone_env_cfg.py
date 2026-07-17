# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Play-Xylophone environment (phys-vidsim robodojo-scene-expansion #29,
M2) -- RoboDojo's `play_Xylophone` task. Reuses ../robodojo_pusht/robodojo_pusht_env_cfg.py's
exact robot/table/camera setup unchanged (same dual-X5 rig, same mount poses), matching
every other robodojo_* env cfg in this fork. The final WAN-hard scene in this pass: a large
static xylophone (Geometry, ~42cm-wide, by far the biggest footprint of any object in this
experiment) + a mallet + mallet_stand (Rigid).

Object assets staged into this repo's gitignored `_robodojo_object_usd/`. RoboDojo's export
pipeline ships PhysicsRigidBodyAPI + PhysicsCollisionAPI/PhysicsMeshCollisionAPI baked into
every object.usdz (confirmed live via `GetAppliedSchemas()`, not assumed) -- no
MeshConverter physics-baking step needed, same plain UsdFileCfg(usd_path=...) as every
other robodojo_* object.

`xylophone` is the THIRD `Geometry`-typed (fixed, non-free-jointed) real mesh in this fork
(after `play_tic_tac_toe`'s checkerboard and `hang_mugs`' cup_holder) -- mounted as
`AssetBaseCfg`+`UsdFileCfg`, NOT `RigidObjectCfg`. It is base-anchored (z=0 sits on the
table), same convention as the checkerboard.

Object initial positions match the Genesis side exactly
(`simulation/robodojo_play_xylophone/scene.py`'s `MALLET_STAND_POS`/`MALLET_POS`/
`XYLOPHONE_POS`) -- see that module's docstring for the hand-verified clearance layout
(the xylophone's unusually large footprint required real clearance checking, not reuse of
a prior scene's positions).

`mallet`/`mallet_stand`'s `mass_props` (2026-07-17, phys-vidsim `physics-time-calibration`
#33 deliverable 4) are real-world target masses, matching `simulation/sim_common/
physics_defaults.py`'s `ROBODOJO_TARGET_MASS_KG` exactly. Spawned via `UsdFileWithMassCfg`
(`..pusht.mdp.spawners`), not a plain `sim_utils.UsdFileCfg` (see that spawner's docstring
for why: RoboDojo's `object.usdz` has no pre-authored `MassAPI`). Fixed `xylophone` doesn't
need one.

No custom success/out-of-bounds termination -- construction + condition-rendering scope
only, matching every other robodojo_* env cfg in this fork.
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.x5 import X5_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg
from ..pusht.mdp.spawners import UsdFileWithMassCfg

# 180 deg yaw around Z, (w,x,y,z) -- same as robodojo_pusht_env_cfg.py.
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_FINGER_JOINT_NAMES = ["joint7", "joint8"]
_EE_BODY_NAME = "link6"

_REPO_ROOT_USD_DIR = "_robodojo_object_usd"
_XYLOPHONE_STAGED_DIR = "xylophone_00000"
_XYLOPHONE_POS = (0.0, -0.35, 0.0)

# Matches simulation/robodojo_play_xylophone/scene.py's MALLET_STAND_POS/MALLET_POS exactly.
_MALLET_STAND_POS = (0.28, -0.15, 0.0276)
_MALLET_POS = (0.28, -0.42, 0.033)
_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

# Real-world target mass (kg) -- matches phys-vidsim's sim_common/physics_defaults.py
# ROBODOJO_TARGET_MASS_KG exactly.
_MALLET_STAND_MASS_KG = 0.05
_MALLET_MASS_KG = 0.03


def _usd_path(staged_dir: str) -> str:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    return str(repo_root / _REPO_ROOT_USD_DIR / staged_dir / "object.usdz")


@configclass
class RobodojoPlayXylophoneSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Play-Xylophone scene using two ARX X5 arms + a static xylophone +
    a mallet + mallet_stand."""

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

    # Static Geometry-typed xylophone -- fixed AssetBaseCfg + UsdFileCfg, like the table
    # itself, NOT RigidObjectCfg (see module docstring).
    xylophone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Xylophone",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_XYLOPHONE_POS),
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path(_XYLOPHONE_STAGED_DIR)),
    )

    mallet_stand = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MalletStand",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_MALLET_STAND_POS, rot=_INIT_ROT),
        spawn=UsdFileWithMassCfg(
            usd_path=_usd_path("mallet_stand_00000"),
            mass_props=sim_utils.MassPropertiesCfg(mass=_MALLET_STAND_MASS_KG),
        ),
    )

    mallet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mallet",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_MALLET_POS, rot=_INIT_ROT),
        spawn=UsdFileWithMassCfg(
            usd_path=_usd_path("mallet_00000"),
            mass_props=sim_utils.MassPropertiesCfg(mass=_MALLET_MASS_KG),
        ),
    )

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
    """Identical wiring to robodojo_pusht_env_cfg.py's ActionsCfg -- same rig, same
    EE-velocity pusher + binary gripper per side."""

    pusher_left = FrankaEEPusherActionCfg(
        asset_name="robot_left",
        arm_joint_names=_ARM_JOINT_NAMES,
        ee_body_name=_EE_BODY_NAME,
        ee_z_height=0.03,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
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
        ee_z_height=0.03,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
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
    reset_mallet_stand = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("mallet_stand")},
    )
    reset_mallet = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("mallet")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RobodojoPlayXylophoneEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoPlayXylophoneSceneCfg = RobodojoPlayXylophoneSceneCfg(
        num_envs=1, env_spacing=1.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # Matches robodojo_pusht_env_cfg.py's cadence exactly.
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
