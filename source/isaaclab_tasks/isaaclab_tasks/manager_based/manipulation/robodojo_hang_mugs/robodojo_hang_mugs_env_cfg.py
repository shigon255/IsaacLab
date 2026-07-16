# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Hang-Mugs environment (phys-vidsim robodojo-scene-expansion #29, M2) --
RoboDojo's `hang_mugs` task. Reuses ../robodojo_pusht/robodojo_pusht_env_cfg.py's exact
robot/table/camera setup unchanged (same dual-X5 rig, same mount poses), matching every
other robodojo_* env cfg in this fork. The scene motivating this pass's Genesis-side
convex-decomposition collision proxy -- Isaac needs no such workaround: RoboDojo's export
pipeline ships exact `PhysicsMeshCollisionAPI` collision baked into every object.usdz
(confirmed live via `GetAppliedSchemas()` for both `mug`/`cup_holder`, not assumed from the
bowl's precedent), so a concave mug's handle/interior is physically exact here, unlike the
Genesis side's decomposed-hull approximation.

`cup_holder` is the SECOND `Geometry`-typed (fixed, non-free-jointed) real mesh in this
fork (after `play_tic_tac_toe`'s checkerboard) -- mounted as `AssetBaseCfg`+`UsdFileCfg`,
NOT `RigidObjectCfg`. Unlike the checkerboard (base-anchored, `z=0`), `cup_holder` is
CENTER-anchored (confirmed via `pxr.UsdGeom.Mesh` bbox, not assumed uniform across every
`Geometry`-typed asset) -- its own placement uses `z = half_z` to sit on the table surface,
matching the Genesis-side scene module's identical distinction.

Object assets staged into this repo's gitignored `_robodojo_object_usd/`. Object initial
positions match the Genesis side exactly
(`simulation/robodojo_hang_mugs/scene.py`'s `MUG_INIT_POS`/`CUP_HOLDER_POS`).

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

# 180 deg yaw around Z, (w,x,y,z) -- same as robodojo_pusht_env_cfg.py.
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_FINGER_JOINT_NAMES = ["joint7", "joint8"]
_EE_BODY_NAME = "link6"

_REPO_ROOT_USD_DIR = "_robodojo_object_usd"
_MUG_STAGED_DIR = "mug_00000"
_HOLDER_STAGED_DIR = "cup_holder_00000"
_HOLDER_POS = (0.0, 0.025, 0.174)

# Matches simulation/robodojo_hang_mugs/scene.py's MUG_INIT_POS exactly.
_MUG_POS: dict[str, tuple[float, float, float]] = {
    "mug0": (-0.30, -0.15, 0.057),
    "mug1": (0.00,  -0.15, 0.057),
    "mug2": (0.30,  -0.15, 0.057),
}
_MUG_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def _usd_path(staged_dir: str) -> str:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    return str(repo_root / _REPO_ROOT_USD_DIR / staged_dir / "object.usdz")


def _mug_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_MUG_POS[name], rot=_MUG_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path(_MUG_STAGED_DIR)),
    )


@configclass
class RobodojoHangMugsSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Hang-Mugs scene using two ARX X5 arms + a static cup_holder rack +
    3 mug instances (one distinct real mesh, exact PhysX mesh collision)."""

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

    # Static Geometry-typed cup holder -- fixed AssetBaseCfg + UsdFileCfg, like the table
    # itself, NOT RigidObjectCfg (see module docstring).
    holder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Holder",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_HOLDER_POS),
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path(_HOLDER_STAGED_DIR)),
    )

    mug0 = _mug_cfg("mug0")
    mug1 = _mug_cfg("mug1")
    mug2 = _mug_cfg("mug2")

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
        ee_z_height=0.03,
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
    reset_mug0 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("mug0")},
    )
    reset_mug1 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("mug1")},
    )
    reset_mug2 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("mug2")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RobodojoHangMugsEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoHangMugsSceneCfg = RobodojoHangMugsSceneCfg(
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
