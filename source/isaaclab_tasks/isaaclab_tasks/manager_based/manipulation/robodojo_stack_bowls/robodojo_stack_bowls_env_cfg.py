# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Stack-Bowls environment (phys-vidsim robodojo-scene-collection #18,
M3) -- RoboDojo's `stack_bowls` task. Reuses ../robodojo_pusht/robodojo_pusht_env_cfg.py's
exact robot/table/camera setup unchanged (same dual-X5 rig, same mount poses); only the
object differs -- 3 RoboDojo bowl instances instead of the procedural T-block.

Bowl asset: `Assets/Object/RoboDojo/Rigid/bowl/00001/object.usdz`, staged into this repo's
gitignored `_robodojo_bowl_usd/` via `scripts/setup_robodojo_bowl_usd.sh` (phys-vidsim),
mirroring `_x5_usd/`'s/`_robosuite_usd/`'s pattern. UNLIKE the robosuite objects
(`../ur5e_pickplace/ur5e_pickplace_env_cfg.py`'s `_object_cfg`), this asset needs no
MeshConverter physics-baking step first -- confirmed live via a `pxr.Usd.Stage.Traverse()`
that RoboDojo's own `object.usdz` already ships `PhysicsRigidBodyAPI` (on `/root`) and
`PhysicsCollisionAPI`+`PhysicsMeshCollisionAPI` (on `/root/collision/model`) baked in, so
it's referenced directly via a plain `UsdFileCfg(usd_path=...)`.

`mass_props=sim_utils.MassPropertiesCfg(mass=_BOWL_MASS_KG)` (2026-07-17, phys-vidsim
`physics-time-calibration` #33 deliverable 4): the source asset has no explicit mass
attribute, so it previously fell back to PhysX's density-based default (same as RoboDojo's
own runtime) -- an unexamined default, not a real bowl's weight. `modify_mass_properties`
(the function `mass_props` drives, see `isaaclab.sim.schemas`) is `apply_nested`-decorated
and operates on the fully-resolved stage at spawn time, so it correctly overrides the mass
on the referenced content's already-baked `RigidBodyAPI` prim -- unlike the robosuite
MeshConverter case above, this isn't creating a physics schema from scratch, just setting a
value on one that already exists. `_BOWL_MASS_KG` matches the Genesis-side target exactly
(`simulation/sim_common/physics_defaults.py`'s `ROBODOJO_TARGET_MASS_KG["bowl"]`, phys-vidsim
repo) so both backends agree on the object's mass even though their density/collision-volume
values differ (Genesis: convex hull; Isaac: exact mesh).

Bowl initial positions match the Genesis side exactly (`simulation/robodojo_stack_bowls/
scene.py`'s `BOWL_INIT_POS`) -- spread across the table at y=-0.10, x=[-0.18, 0, 0.18],
z=0.03 (not stacked initially, since stacking is the task).

No custom success/out-of-bounds termination -- like the Genesis side, this milestone's
scope is construction + condition rendering, not exact task-success detection; `time_out`
alone is sufficient (task_success() on the Genesis/repo side is computed post-hoc from
recorded state, not a live sim termination, so Isaac doesn't need the equivalent either).
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

# 180 deg yaw around Z, (w,x,y,z) -- same as robodojo_pusht_env_cfg.py.
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_FINGER_JOINT_NAMES = ["joint7", "joint8"]
_EE_BODY_NAME = "link6"

# Matches simulation/robodojo_stack_bowls/scene.py's BOWL_INIT_POS exactly.
_BOWL_NAMES = ("bowl0", "bowl1", "bowl2")
_BOWL_PRIM_NAME = {"bowl0": "Bowl0", "bowl1": "Bowl1", "bowl2": "Bowl2"}
_BOWL_INIT_POS: dict[str, tuple[float, float, float]] = {
    "bowl0": (-0.18, -0.10, 0.03),
    "bowl1": (0.0, -0.10, 0.03),
    "bowl2": (0.18, -0.10, 0.03),
}
_BOWL_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
_REPO_ROOT_USD_DIR = "_robodojo_bowl_usd"

# Real-world target mass (kg) -- matches phys-vidsim's sim_common/physics_defaults.py
# ROBODOJO_TARGET_MASS_KG["bowl"] exactly (a ceramic bowl, typical 250-400g).
_BOWL_MASS_KG = 0.30


def _bowl_cfg(name: str) -> RigidObjectCfg:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    usd_path = str(repo_root / _REPO_ROOT_USD_DIR / "bowl_00001" / "object.usdz")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{_BOWL_PRIM_NAME[name]}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BOWL_INIT_POS[name], rot=_BOWL_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            mass_props=sim_utils.MassPropertiesCfg(mass=_BOWL_MASS_KG),
        ),
    )


@configclass
class RobodojoStackBowlsSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Stack-Bowls scene using two ARX X5 arms + 3 real bowl meshes."""

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

    bowl0 = _bowl_cfg("bowl0")
    bowl1 = _bowl_cfg("bowl1")
    bowl2 = _bowl_cfg("bowl2")

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
    reset_bowl0 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("bowl0")},
    )
    reset_bowl1 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("bowl1")},
    )
    reset_bowl2 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("bowl2")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RobodojoStackBowlsEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoStackBowlsSceneCfg = RobodojoStackBowlsSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
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
