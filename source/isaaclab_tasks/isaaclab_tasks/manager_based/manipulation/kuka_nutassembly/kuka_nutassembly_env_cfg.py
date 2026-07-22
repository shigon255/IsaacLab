# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm KUKA push-together environment: reproduces the shape of phys-vidsim's
Genesis-side simulation/kuka_nutassembly/scene.py::KukaNutAssemblyScene (KUKA iiwa14 +
round-nut/square-nut, no gripper, push-together) as an Isaac Lab manager-based env,
for isaac-lab-multi-scene (#21)'s M3.

Robot: KUKA_IIWA14_HIGH_PD_CFG (isaaclab_assets.robots.kuka, new this pass -- see that
module's own docstring). No gripper (matches Genesis's own no-gripper ArmSpec). No
wrist-yaw control (same pre-existing, documented Isaac-side gap as
ur5e_pickplace_env_cfg.py -- FrankaEEPusherAction has no yaw channel at all; see that
file's module docstring for the full writeup, not repeated here).

Objects: round-nut/square-nut are procedural CuboidCfg BOUNDING-BOX approximations,
NOT converted meshes -- a DELIBERATE, documented simplification, not a deferred bug.
robosuite's own nut fragments are torus-ring/square-frame shapes built from 5-9
individual <geom type="box"> primitives each (no single mesh file at all -- see
simulation/sim_common/robosuite_assets.py's own object fragments), and MJCF-converting
that multi-geom structure hit a SEPARATE, still-unresolved importer bug this pass
(`RuntimeError: Used null prim` / `_IsValidPathForCreatingPrim ... Path must be an
absolute path: <>` -- distinct from the meshed-object MeshConverter bug
ur5e_pickplace's cereal/can/milk/bread hit; naming the previously-unnamed object-root
<body> didn't fix it, not root-caused further this pass). Rather than keep chasing
that importer bug for a shape that's fundamentally just "two objects pushed into
contact" (a CT/contact scenario, not a shape-recognition one -- see Genesis's own
scene.py module docstring), a single bounding-box CuboidCfg per nut (computed from
each fragment's own geom list -- round-nut half-extent (0.07, 0.055, 0.01), square-nut
(0.065, 0.045, 0.01), both rounded up slightly from the exact AABB) is functionally
equivalent for push-together contact detection while sidestepping the bug entirely,
using the SAME already-proven procedural-CuboidCfg + RigidObjectCfg pattern
franka_stack_env_cfg.py's cubes and aloha_blocks_env_cfg.py's blocks already use (no
mesh conversion, no physics-baking gotcha -- that pattern's spawn-time rigid_props DO
apply correctly for a procedural spawner, unlike the referenced-USD-file case). Masses
are reasoned-not-measured (round-nut 0.15kg, square-nut 0.12kg -- large industrial-prop
nut-assembly-task pieces, ~10-11cm span, thin flat steel/brass plate; see module
docstring for why a real per-nut datasheet doesn't exist: robosuite's fragments specify
a material density against actual (non-box) geometry, not a usable mass for a box
approximation). Matches `physics_defaults.ROBOSUITE_TARGET_MASS_KG["round-nut"/
"square-nut"]` on the phys-vidsim side exactly (sim-verification-campaign #45, F1) --
the original 0.02kg/0.015kg values were never targeted at a real-world mass either and
measured far too light on BOTH backends (unlike cereal/can/milk/bread, which were
already fine on Isaac; only Genesis needed a fix for those).

Camera: fixed_cam mirrors every other scene's CameraCfg exactly (same top-down
rot=(0,1,0,0) static default) -- same "tune the teleop viewpoint-orbit default
instead" precedent every prior scene in this repo established.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
from isaaclab_assets.robots.kuka import KUKA_IIWA14_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg

# ---- Shared constants -- mirror simulation/kuka_nutassembly/scene.py's own NUT_NAMES
# (kept in sync manually; there's no Genesis-free import path to that module). ----
_NUT_NAMES: tuple[str, ...] = ("round-nut", "square-nut")
_NUT_PRIM_NAME: dict[str, str] = {"round-nut": "RoundNut", "square-nut": "SquareNut"}
# Bounding-box half-extents computed from each fragment's own <geom> list (see module
# docstring) -- a documented approximation, not the real ring/frame shape.
_NUT_HALF_SIZE: dict[str, tuple[float, float, float]] = {
    "round-nut": (0.07, 0.055, 0.01),
    "square-nut": (0.065, 0.045, 0.01),
}
_NUT_MASS: dict[str, float] = {"round-nut": 0.15, "square-nut": 0.12}
_NUT_RGBA: dict[str, tuple[float, float, float]] = {
    # Loosely matches robosuite's own steel-scratched/brass-ambra material tone.
    "round-nut": (0.6, 0.6, 0.65), "square-nut": (0.65, 0.55, 0.25),
}
# Z copied from Genesis's own _NUT_INIT_POS (~5mm table clearance for a ~1cm
# half-height object); X/Y are Isaac's own layout (see module docstring pattern).
_NUT_INIT_POS: dict[str, tuple[float, float, float]] = {
    "round-nut": (0.35, -0.10, 0.015),
    "square-nut": (0.35, 0.10, 0.015),
}
_NUT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity


def _nut_cfg(name: str) -> RigidObjectCfg:
    half = _NUT_HALF_SIZE[name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{_NUT_PRIM_NAME[name]}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_NUT_INIT_POS[name], rot=_NUT_INIT_ROT),
        spawn=sim_utils.CuboidCfg(
            size=(half[0] * 2, half[1] * 2, half[2] * 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=_NUT_MASS[name]),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_NUT_RGBA[name], roughness=0.3, metallic=0.8),
        ),
    )


@configclass
class KukaNutAssemblySceneCfg(InteractiveSceneCfg):
    """Single KUKA iiwa14 + table + round-nut/square-nut + fixed_cam."""

    robot = KUKA_IIWA14_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.42, 0.38), roughness=0.8),
        ),
    )

    round_nut = _nut_cfg("round-nut")
    square_nut = _nut_cfg("square-nut")

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
        # This raw spawn-time offset (every sibling scene's own top-down convention)
        # is a rarely-exercised fallback ONLY -- collect_demos_isaac.py repositions
        # fixed_cam to the oblique scene.DEFAULT_CAM_EYE/DEFAULT_CAM_LOOKAT default
        # right after every env.reset() (see that module's own comment on
        # `_initial_eye`), and replay_render_isaac.py always prefers a demo's own
        # recorded cam_pos/cam_lookat, falling back to DEFAULT_CAM_EYE/LOOKAT only for
        # a demo whose metadata is missing them entirely -- this raw offset is never
        # actually rendered in normal operation. An earlier pass tuned THIS value
        # (height 0.9->1.8, focal_length 18->14) chasing what turned out to be a
        # --auto-start ordering bug (fixed in collect_demos_isaac.py, see its own
        # comment) that made auto-start-recorded demos capture this dead default
        # instead of the real oblique one -- reverted back to the sibling convention
        # once the actual bug (and the real viewpoint fix, in
        # scenes/kuka_nutassembly.py's DEFAULT_CAM_EYE) were found instead.
        offset=CameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.9), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
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
    """Single-arm EE-velocity pusher, no gripper -- matches Genesis's own no-gripper
    KUKA ArmSpec exactly."""

    arm = FrankaEEPusherActionCfg(
        asset_name="robot",
        arm_joint_names=[f"lbr_iiwa_joint_{i}" for i in range(1, 8)],
        ee_body_name="lbr_iiwa_link_7",
        ee_z_height=0.25,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
        z_workspace=(0.02, 0.5),
        workspace=((0.15, -0.35), (0.55, 0.35)),
        control_yaw_offset=0.0,
        default_target_xy=(0.35, 0.0),
        enable_z=True,
        # lock_orientation=True (every sibling scene's own default) was confirmed live
        # to be kinematically infeasible here: driving the arm toward round_nut's own
        # position (0.35,-0.10,~0.05) while holding the home pose's EE orientation fixed
        # pins 4 of the 7 joints exactly at their hard limits and diverges rather than
        # converging (a real diagnostic script, not assumed -- see
        # exps/current/isaac-lab-multi-scene/status.md). KUKA's home pose (matching
        # Genesis's own _KUKA_HOME_QPOS -- can't change independently, both backends'
        # teleop must start from the same joint config) is tuned for a forward/elevated
        # reach, not a close-in low one, and that specific orientation just isn't
        # compatible with reaching down near the base. Free-orientation (position-only)
        # IK sidesteps the conflict entirely -- re-verified with the same script:
        # converges to within ~1cm of the nut's XY with 6/7 joints comfortably clear of
        # their limits (worst case 0.13 rad margin, none pinned).
        lock_orientation=False,
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
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_round_nut = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("round_nut")},
    )
    reset_square_nut = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("square_nut")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class KukaNutAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    scene: KukaNutAssemblySceneCfg = KukaNutAssemblySceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
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
