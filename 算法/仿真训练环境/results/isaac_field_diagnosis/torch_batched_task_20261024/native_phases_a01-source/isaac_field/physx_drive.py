"""Native PhysX mimic constraints and explicit backend/contact identity.

All coupling is authored into a separate USD asset. No joint state is overwritten
to imitate a physical constraint, and the original geometry asset is retained.
"""
from __future__ import annotations

import math
from pathlib import Path
import uuid


def coupling_relations(spec, guides=None, *, couple_support_wheels=True):
    rows = []
    for side in ("left", "right"):
        leader = f"{side}_wheel_0_wheel_joint"
        if couple_support_wheels:
            rows.extend({"leader": leader, "follower": f"{side}_wheel_{i}_wheel_joint",
                         "ratio": 1.0, "kind": "support_wheel"} for i in range(1, 7))
        if guides is not None and guides.rotation_coupling == "same_side_track_speed":
            guide = next(wheel for wheel in guides.wheel_specs(spec) if wheel.side == side)
            rows.append({"leader": leader, "follower": guide.joint_name,
                         "ratio": spec.wheel_radius_m / guide.radius_m, "kind": "front_guide"})
    return rows


def ensure_physx_robot_asset(cache_dir, *, front_guide_spec=None, couple_support_wheels=True):
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics
    from .assets import ROBOT_ROOT_PATH, ROBOT_SPEC, ensure_robot_asset

    source = ensure_robot_asset(cache_dir, front_guide_spec=front_guide_spec)
    source = Path(source)
    target = source.with_name(f"{source.stem}_physx_r1_c{int(couple_support_wheels)}.usda")
    rows = coupling_relations(ROBOT_SPEC, front_guide_spec,
                              couple_support_wheels=couple_support_wheels)
    if not target.exists():
        # Export to a new layer rather than editing the cached source in place.
        temporary = target.with_name(f'.px-{uuid.uuid4().hex}.usda')
        # Keep the owning stage alive while exporting its root layer. Kit's
        # layer wrapper can otherwise outlive the temporary stage's weak ptr.
        source_stage = Usd.Stage.Open(str(source))
        if source_stage is None:
            raise RuntimeError(f'could not open source robot asset: {source}')
        if not source_stage.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f'could not export temporary PhysX asset: {temporary}')
        stage = Usd.Stage.Open(str(temporary))
        root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
        root.CreateAttribute("field:physicsBackend", Sdf.ValueTypeNames.String).Set("physx")
        for row in rows:
            follower = stage.GetPrimAtPath(f"{ROBOT_ROOT_PATH}/{row['follower']}")
            reference = Sdf.Path(f"{ROBOT_ROOT_PATH}/{row['leader']}")
            if not follower.IsValid() or not stage.GetPrimAtPath(reference).IsValid():
                raise RuntimeError(f"missing authored mimic joint: {row}")
            # A revolute joint has one DOF; rotX is the conventional mimic
            # instance token irrespective of its USD axis (NVIDIA demo).
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(follower, "rotX")
            mimic.CreateReferenceJointRel().SetTargets([reference])
            mimic.CreateGearingAttr().Set(-float(row["ratio"]))
            mimic.CreateOffsetAttr().Set(0.0)
            for joint_name in (row["leader"], row["follower"]):
                joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{ROBOT_ROOT_PATH}/{joint_name}"))
                joint.CreateLowerLimitAttr().Set(-1.0e10)
                joint.CreateUpperLimitAttr().Set(1.0e10)
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f'could not save complete PhysX asset: {temporary}')
        stage = None
        try:
            # Windows rename refuses an existing target; concurrent readers
            # therefore never see a half-authored asset or an overwritten one.
            temporary.rename(target)
        except FileExistsError:
            temporary.unlink()
    stage = Usd.Stage.Open(str(target))
    if stage.GetPrimAtPath(ROBOT_ROOT_PATH).GetAttribute("field:physicsBackend").Get() != "physx":
        raise RuntimeError(f"asset is not an identified PhysX variant: {target}")
    for row in rows:
        joint = stage.GetPrimAtPath(f"{ROBOT_ROOT_PATH}/{row['follower']}")
        mimic = PhysxSchema.PhysxMimicJointAPI.Get(joint, 'rotX')
        expected_reference = [Sdf.Path(f"{ROBOT_ROOT_PATH}/{row['leader']}")]
        if (not mimic or mimic.GetReferenceJointRel().GetTargets() != expected_reference
                or not math.isclose(float(mimic.GetGearingAttr().Get()), -row['ratio'], abs_tol=1e-7)):
            raise RuntimeError(f'incomplete PhysX asset; refusing cached variant: {target}: {row}')
    return target


def same_side_wheel_coupling_metadata(enabled, *, guide_relations_per_world=None):
    wheels = 12 if enabled else 0
    guides = int(guide_relations_per_world or 0)
    if guides not in (0, 2):
        raise ValueError("guide_relations_per_world must be zero or two")
    return {
        "enabled": bool(wheels or guides),
        "mechanism": "PhysxMimicJointAPI",
        "relation": "q_follower = ratio * q_same_side_wheel0",
        "relations_per_world": wheels + guides,
        "wheel_relations_per_world": wheels,
        "guide_relations_per_world": guides,
        "front_guide_relation": "q_guide = (R_main/R_guide)*q_same_side_wheel0" if guides else None,
        "leader_wheel_index_per_side": 0,
        "follower_wheel_indices_per_side": list(range(1, 7)) if enabled else [],
        "source_usd_changed": False,
        "physx_variant_authored": True,
        "execution": "retain per-wheel actuators, PI state and configured effort limits",
    }


def audit_physx_coupling(env):
    from pxr import PhysxSchema
    import isaaclab.sim as sim_utils
    from .assets import ROBOT_SPEC

    rows = coupling_relations(ROBOT_SPEC, env.front_guide_spec,
                              couple_support_wheels=env.cfg.couple_same_side_wheels)
    stage = sim_utils.get_current_stage()
    native_names = set(env.scene["robot"].joint_names)
    for world_path in env.scene.env_prim_paths:
        root = f"{world_path}/Robot"
        for row in rows:
            assert row["leader"] in native_names and row["follower"] in native_names
            joint = stage.GetPrimAtPath(f"{root}/{row['follower']}")
            mimic = PhysxSchema.PhysxMimicJointAPI.Get(joint, "rotX")
            if not mimic or [str(path) for path in mimic.GetReferenceJointRel().GetTargets()] != [f"{root}/{row['leader']}"]:
                raise RuntimeError(f"missing or cross-world PhysX mimic relation: {root}: {row}")
            if not math.isclose(float(mimic.GetGearingAttr().Get()), -row["ratio"], rel_tol=0, abs_tol=1e-7):
                raise RuntimeError(f"incorrect PhysX mimic ratio: {row}")
    return {"status": "passed", "mechanism": "PhysxMimicJointAPI",
            "worlds_checked": env.num_envs, "relations_per_world": len(rows),
            "wheel_relations_per_world": sum(row["kind"] == "support_wheel" for row in rows),
            "guide_relations_per_world": sum(row["kind"] == "front_guide" for row in rows),
            "total_mimic_rows": len(rows) * env.num_envs, "cross_world_pairs": 0,
            "audit_scope": "USD relations and imported native joint names; motion residuals tested separately"}


def build_physics_contact_metadata(env):
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics
    from .runtime import simulator_identity

    stage = sim_utils.get_current_stage()
    materials = {}
    for world in env.scene.env_prim_paths:
        for role in ("Field", "Robot"):
            prim_path = f"{world}/{role}/material"
            material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(prim_path))
            if not material:
                raise RuntimeError(f"missing explicit contact material: {prim_path}")
            values = {"static_friction": material.GetStaticFrictionAttr().Get(),
                      "dynamic_friction": material.GetDynamicFrictionAttr().Get(),
                      "restitution": material.GetRestitutionAttr().Get()}
            if any(value is None or not math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"invalid material values: {prim_path}: {values}")
            materials[prim_path] = values
    manager = env.sim.physics_manager
    # Configs may expose a lazy ResolvableString with callable metadata.
    manager_module = manager.__module__
    manager_name = manager.__name__
    if not manager_module.startswith('isaaclab_physx.'):
        raise RuntimeError(f'expected Isaac Sim PhysX manager, got {manager_module}.{manager_name}')
    return {**simulator_identity(), "backend": "PhysxCfg", "solver": "TGS",
            "physics_manager": f'{manager_module}.{manager_name}',
            "solver_position_iteration_range": [env.cfg.sim.physics.min_position_iteration_count,
                                                 env.cfg.sim.physics.max_position_iteration_count],
            "solver_velocity_iteration_range": [env.cfg.sim.physics.min_velocity_iteration_count,
                                                 env.cfg.sim.physics.max_velocity_iteration_count],
            "physics_dt_s": float(env.cfg.sim.dt), "decimation": int(env.cfg.decimation),
            "policy_dt_s": float(env.step_dt),
            "usd_authored_materials": materials,
            "contact_offset_m": 0.005, "rest_offset_m": 0.0,
            "calibrated_hardware": False}
