"""CPU-only builder, solver, and canonical-metadata preflight for drive v7."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import newton
import warp as wp

from isaac_field.newton_drive import (
    CONTACT_SHAPE_FIELDS,
    SameSideMimicBuilderHook,
    _canonical_materials_by_role,
    audit_same_side_mimic_model,
    same_side_wheel_coupling_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("production_v7_cpu_preflight.json")
NAMES = {side: [f"{side}_wheel_{i}_wheel_joint" for i in range(7)] for side in ("left", "right")}


def build(world_count: int):
    builder = newton.ModelBuilder(up_axis="Z")
    hook = SameSideMimicBuilderHook(NAMES, world_count)
    with wp.ScopedDevice("cpu"):
        for world in range(world_count):
            builder.begin_world()
            for side in ("left", "right"):
                for wheel in NAMES[side]:
                    label = f"/World/envs/env_{world}/Robot/base_link/{wheel}"
                    body = builder.add_link(
                        mass=1.0, com=wp.vec3(0.0, 0.0, 0.0),
                        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                        label=f"/World/envs/env_{world}/Robot/base_link/{wheel}_body",
                    )
                    joint = builder.add_joint_revolute(
                        -1, body, axis=(0.0, 1.0, 0.0), label=label,
                        damping=0.0, friction=0.0, armature=0.0,
                    )
                    builder.add_articulation([joint])
            if int(builder.current_world) != world:
                raise AssertionError("builder current_world did not match build world")
            hook(builder, world, np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0]))
            builder.end_world()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    audit = audit_same_side_mimic_model(hook, model)
    return {
        "model_world_count": int(model.world_count),
        "builder_callback_worlds": [row["world_index"] for row in hook.worlds],
        "model_audit": audit,
        "solver_host_neq": int(solver.mj_model.neq),
        "solver_warp_neq": int(solver.mjw_model.neq),
        "solver_warp_nworld": int(solver.mjw_data.nworld),
        "solver_warp_qpos_shape": list(solver.mjw_data.qpos.shape),
        "solver_warp_qvel_shape": list(solver.mjw_data.qvel.shape),
    }


def canonical_metadata_check():
    wheels = [f"{side}_wheel_{index}_wheel_link" for side in ("left", "right") for index in range(7)]
    def make(worlds, bad_kf=False):
        labels, arrays = [], {field: [] for field in CONTACT_SHAPE_FIELDS}
        for world in range(worlds):
            labels.append(f"/World/envs/env_{world}/Field/Ground")
            labels.extend(f"/World/envs/env_{world}/Robot/base_link/{name}/Collision/cylinder" for name in wheels)
            values = {"shape_material_mu": 0.8, "shape_material_kf": 1001.0 if bad_kf else 1000.0,
                      "shape_material_ke": 2500.0, "shape_material_kd": 100.0,
                      "shape_margin": 0.002, "shape_gap": 0.005}
            for field in CONTACT_SHAPE_FIELDS:
                arrays[field].append(values[field])
                arrays[field].extend([values[field]] * len(wheels))
        return _canonical_materials_by_role(labels, arrays, wheels)
    one, four = make(1), make(4)
    changed = make(4, bad_kf=True)
    if one != four or one == changed:
        raise AssertionError("material metadata must ignore env count and expose changed material values")
    mech1, mech4 = same_side_wheel_coupling_metadata(True), same_side_wheel_coupling_metadata(True)
    if mech1 != mech4:
        raise AssertionError("coupling metadata unexpectedly depends on env count")
    if same_side_wheel_coupling_metadata(False) == mech1:
        raise AssertionError("disabled coupling metadata must differ")
    return {"one_world_materials": one, "four_world_materials": four,
            "different_kf_detected": one != changed, "mechanism_env_count_independent": True}


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    one, two = build(1), build(2)
    if one["model_audit"]["total_mimic_rows"] != 12 or one["model_audit"]["total_equality_rows"] != 12:
        raise AssertionError("one-world model row counts are wrong")
    if two["model_audit"]["total_mimic_rows"] != 24 or two["model_audit"]["total_equality_rows"] != 24:
        raise AssertionError("two-world model row counts are wrong")
    if two["solver_host_neq"] != two["solver_warp_neq"] or two["solver_warp_nworld"] != 2:
        raise AssertionError("two-world solver topology or batch shape is wrong")
    if two["solver_warp_qpos_shape"][0] != 2 or two["solver_warp_qvel_shape"][0] != 2:
        raise AssertionError("MJWarp state is not batched across two worlds")
    result = {"schema": "drive-v7-mimic-cpu-preflight-v1", "newton_version": version("newton"),
              "device": "cpu", "one_world": one, "two_worlds": two,
              "metadata": canonical_metadata_check(), "passed": True}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
