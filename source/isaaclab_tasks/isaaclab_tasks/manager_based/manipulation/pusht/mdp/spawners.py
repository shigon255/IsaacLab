# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass


@configclass
class PushTShapeCfg(RigidObjectSpawnerCfg):
    """Spawner config for a T shape made from two rectangular cuboids."""

    func: Callable = None

    bar_size: tuple[float, float, float] = (0.24, 0.08, 0.04)
    stem_size: tuple[float, float, float] = (0.08, 0.24, 0.04)
    bar_offset: tuple[float, float, float] = (0.0, 0.08, 0.0)
    stem_offset: tuple[float, float, float] = (0.0, -0.04, 0.0)

    visual_material_path: str = "material"
    visual_material: materials.VisualMaterialCfg | None = None
    physics_material_path: str = "material"
    physics_material: materials.PhysicsMaterialCfg | None = None


def _spawn_box(
    prim_path: str,
    size: tuple[float, float, float],
    offset: tuple[float, float, float],
    cfg: PushTShapeCfg,
    stage: Usd.Stage,
    visual_material_path: str | None,
    physics_material_path: str | None,
):
    size_min = min(size)
    scale = tuple(dim / size_min for dim in size)
    create_prim(
        prim_path,
        "Cube",
        translation=offset,
        scale=scale,
        attributes={"size": size_min},
        stage=stage,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(prim_path, cfg.collision_props, stage=stage)
    if visual_material_path is not None:
        bind_visual_material(prim_path, visual_material_path, stage=stage)
    if physics_material_path is not None:
        bind_physics_material(prim_path, physics_material_path, stage=stage)


@clone
def spawn_t_shape(
    prim_path: str,
    cfg: PushTShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a compound T shape with collision geometry on child cuboids."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    visual_material_path = None
    if cfg.visual_material is not None:
        visual_material_path = (
            f"{prim_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(visual_material_path, cfg.visual_material)

    physics_material_path = None
    if cfg.physics_material is not None:
        physics_material_path = (
            f"{prim_path}/{cfg.physics_material_path}"
            if not cfg.physics_material_path.startswith("/")
            else cfg.physics_material_path
        )
        cfg.physics_material.func(physics_material_path, cfg.physics_material)

    _spawn_box(
        f"{prim_path}/bar",
        cfg.bar_size,
        cfg.bar_offset,
        cfg,
        stage,
        visual_material_path,
        physics_material_path,
    )
    _spawn_box(
        f"{prim_path}/stem",
        cfg.stem_size,
        cfg.stem_offset,
        cfg,
        stage,
        visual_material_path,
        physics_material_path,
    )

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)


PushTShapeCfg.func = spawn_t_shape


@configclass
class UsdFileWithMassCfg(UsdFileCfg):
    """`UsdFileCfg` variant that correctly applies `mass_props` on REFERENCED content whose
    `UsdPhysics.MassAPI` schema isn't already authored in the source file -- see
    `spawn_usd_file_with_mass` for why the base spawner's own `mass_props` handling silently
    no-ops for this case."""

    func: Callable = None


def spawn_usd_file_with_mass(
    prim_path: str,
    cfg: UsdFileWithMassCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD-file asset via `spawn_from_usd`, then explicitly APPLY (not just modify)
    its mass schema.

    `spawn_from_usd`'s own `mass_props` handling calls `schemas.modify_mass_properties`
    directly (`isaaclab/sim/spawners/from_files/from_files.py`), which requires
    `UsdPhysics.MassAPI` to already be applied to the target prim -- that function's own body
    is `if not UsdPhysics.MassAPI(rigid_prim): return False`. RoboDojo's `object.usdz` assets
    ship `PhysicsRigidBodyAPI`/`PhysicsCollisionAPI` baked in but NOT `MassAPI`, so
    `modify_mass_properties` silently fails on every prim in the referenced subtree (a
    `Could not perform 'modify_mass_properties' on any prims...` warning at spawn time --
    confirmed live 2026-07-17, phys-vidsim `physics-time-calibration` #33 deliverable 4; the
    `mass_props=` originally added directly to `UsdFileCfg` in every `robodojo_*_env_cfg.py`
    had NO effect). `schemas.define_mass_properties` (already used by `spawn_t_shape` above,
    for procedurally-authored prims) applies the schema first if missing, then sets it --
    exactly what referenced content with no pre-authored `MassAPI` needs instead. Runs
    harmlessly alongside `spawn_from_usd`'s own no-op attempt -- this function's explicit call
    afterward is what actually takes effect.
    """
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=get_current_stage())
    return prim


UsdFileWithMassCfg.func = spawn_usd_file_with_mass
