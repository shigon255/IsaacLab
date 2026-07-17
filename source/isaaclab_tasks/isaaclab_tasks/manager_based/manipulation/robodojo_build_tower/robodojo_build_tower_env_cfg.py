# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Build-Tower environment (phys-vidsim robodojo-scene-expansion #29, M2) --
RoboDojo's `build_tower` task. Reuses ../robodojo_pusht/robodojo_pusht_env_cfg.py's exact
robot/table/camera setup unchanged (same dual-X5 rig, same mount poses), matching every
other robodojo_* env cfg in this fork; only the objects differ -- 8 block instances drawn
from 5 distinct source meshes (`build_tower.yml`'s own placement->index mapping: index 2 is
reused across 4 placements, every other index once), a SECOND test of the multi-distinct-
mesh pipeline (after `robodojo_pack_objects_into_box`, #29 M1) that additionally exercises
mesh reuse (N>1 body instances sharing ONE staged usdz) plus per-instance rotation.

Object assets staged into this repo's gitignored `_robodojo_object_usd/` (phys-vidsim's
scripts/setup_robodojo_object_usd.sh). RoboDojo's export pipeline ships
PhysicsRigidBodyAPI + PhysicsCollisionAPI baked into every object.usdz (confirmed for the
bowl in #18, holds for every RoboDojo asset since they share one export pipeline) -- no
MeshConverter physics-baking step needed, same plain UsdFileCfg(usd_path=...) as every
other robodojo_* object.

`mass_props=sim_utils.MassPropertiesCfg(density=_BLOCK_DENSITY_KG_M3)` (2026-07-17,
phys-vidsim `physics-time-calibration` #33 deliverable 4): unlike the other robodojo_* env
cfgs (which set an explicit real-world `mass=` per object), these 5 distinct block meshes are
flooring/tile-sample assets reused as generic stacking blocks with no individual real-world
identity -- so a uniform DENSITY (not a per-object mass) is the right calibration axis, matching
`simulation/sim_common/physics_defaults.py`'s `GENERIC_BLOCK_DENSITY_KG_M3` exactly. PhysX
computes each block's actual mass from this density x its own exact-mesh volume. Spawned via
`UsdFileWithMassCfg` (`..pusht.mdp.spawners`), not a plain `sim_utils.UsdFileCfg` -- see that
spawner's docstring for why (RoboDojo's `object.usdz` has no pre-authored `MassAPI`, so
`UsdFileCfg`'s own `mass_props` handling silently no-ops). The resulting per-block masses
won't numerically match Genesis's (which applies the same density to a convex-HULL volume,
not the exact mesh), but both backends now apply the same *density*, closing the
previously-undocumented gap where neither backend examined this at all.

Object initial positions/rotations match the Genesis side exactly
(`simulation/robodojo_build_tower/scene.py`'s `BLOCK_INIT_POS`/`BLOCK_INIT_QUAT`) -- see
that module's docstring for the layout derivation (two side clusters + two planks, hand-
verified clear of both arm mounts).

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

# Matches simulation/robodojo_build_tower/scene.py's BLOCK_INDEX/BLOCK_INIT_POS/
# BLOCK_INIT_QUAT exactly -- (staged block index dir, prim name, init pos, init rot).
_BLOCK_SPECS: dict[str, tuple[str, str, tuple[float, float, float], tuple[float, float, float, float]]] = {
    "block0": ("block_00000", "Block0", (0.00, -0.45, 0.040), (1.0, 0.0, 0.0, 0.0)),
    "block1": ("block_00002", "Block1", (-0.28, -0.15, 0.049), (0.9659258, 0.0, 0.0, 0.258819)),
    "block2": ("block_00002", "Block2", (-0.28, -0.30, 0.049), (1.0, 0.0, 0.0, 0.0)),
    "block3": ("block_00001", "Block3", (0.28, -0.30, 0.037), (0.9848078, 0.0, 0.0, -0.1736482)),
    "block4": ("block_00003", "Block4", (0.28, -0.45, 0.053), (0.9914449, 0.0, 0.0, 0.1305262)),
    "block5": ("block_00002", "Block5", (-0.28, -0.45, 0.049), (1.0, 0.0, 0.0, 0.0)),
    "block6": ("block_00002", "Block6", (0.28, -0.15, 0.049), (1.0, 0.0, 0.0, 0.0)),
    "block7": ("block_00004", "Block7", (0.00, -0.15, 0.040), (1.0, 0.0, 0.0, 0.0)),
}
_REPO_ROOT_USD_DIR = "_robodojo_object_usd"

# Generic block density (kg/m^3) -- matches phys-vidsim's sim_common/physics_defaults.py
# GENERIC_BLOCK_DENSITY_KG_M3 exactly (a light solid-wood-like value; these blocks have no
# individual real-world identity, see module docstring).
_BLOCK_DENSITY_KG_M3 = 600.0


def _block_cfg(name: str) -> RigidObjectCfg:
    from pathlib import Path

    staged_dir, prim_name, pos, rot = _BLOCK_SPECS[name]
    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    usd_path = str(repo_root / _REPO_ROOT_USD_DIR / staged_dir / "object.usdz")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=UsdFileWithMassCfg(
            usd_path=usd_path,
            mass_props=sim_utils.MassPropertiesCfg(density=_BLOCK_DENSITY_KG_M3),
        ),
    )


@configclass
class RobodojoBuildTowerSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Build-Tower scene using two ARX X5 arms + 8 block instances drawn
    from 5 distinct real block meshes."""

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

    block0 = _block_cfg("block0")
    block1 = _block_cfg("block1")
    block2 = _block_cfg("block2")
    block3 = _block_cfg("block3")
    block4 = _block_cfg("block4")
    block5 = _block_cfg("block5")
    block6 = _block_cfg("block6")
    block7 = _block_cfg("block7")

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
    reset_block0 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block0")},
    )
    reset_block1 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block1")},
    )
    reset_block2 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block2")},
    )
    reset_block3 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block3")},
    )
    reset_block4 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block4")},
    )
    reset_block5 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block5")},
    )
    reset_block6 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block6")},
    )
    reset_block7 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block7")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RobodojoBuildTowerEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoBuildTowerSceneCfg = RobodojoBuildTowerSceneCfg(
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
