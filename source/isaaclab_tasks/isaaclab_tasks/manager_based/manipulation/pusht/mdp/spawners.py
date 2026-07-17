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


@clone
def spawn_usd_file_with_mass(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Drop-in replacement for `UsdFileCfg`'s default `func` (`spawn_from_usd`) that
    correctly APPLIES (not just modifies) the mass schema on REFERENCED content whose
    `UsdPhysics.MassAPI` isn't already authored in the source file. Use via
    `UsdFileCfg(..., mass_props=..., func=spawn_usd_file_with_mass)` -- `func` is a plain
    overridable field on the base `SpawnerCfg` (`isaaclab/sim/spawners/spawner_cfg.py`), no
    custom cfg subclass needed (a `UsdFileCfg` subclass redeclaring `func` as a new dataclass
    field was tried first and silently ended up `None` at instantiation -- `@configclass`
    field-shadowing across an already-defaulted parent field, not worth chasing further when
    passing `func=` explicitly at each call site works and is just as clear).

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

    The `@clone` decorator (matching `spawn_t_shape`'s own decoration above) is required, not
    optional -- confirmed live: without it, `AssetBase.__init__` calls this function directly
    with the UNRESOLVED regex prim path (e.g. `/World/envs/env_.*/Bowl0`), and
    `define_mass_properties` below raises `ValueError: Prim path '...' is not valid` since
    that's never a literal path on the stage. `@clone` resolves the regex to the concrete
    per-env path BEFORE calling this function; `spawn_from_usd` below then receives an
    already-concrete path, which its own (also `@clone`-decorated) wrapper passes through
    unchanged (`is_regex_expression` is false for an already-resolved path).
    """
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=get_current_stage())
    return prim
