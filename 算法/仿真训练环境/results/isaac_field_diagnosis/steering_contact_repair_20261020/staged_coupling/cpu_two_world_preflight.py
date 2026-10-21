"""CPU-only two-world Newton builder/SolverMuJoCo preflight for staged coupling."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import newton
import warp as wp

from multi_world_mimic import (
    EQUALITY_SOLIMP,
    EQUALITY_SOLREF,
    MultiWorldSameSideMimicHook,
    audit_builder_and_solver,
    mechanism_metadata,
)

STAGE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = STAGE_DIR / "cpu_two_world_preflight.json"
EXIT_PATH = STAGE_DIR / "cpu_two_world_preflight.exit.json"
WHEEL_NAMES = {
    side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
    for side in ("left", "right")
}


def _build_and_audit(world_count: int) -> dict:
    builder = newton.ModelBuilder(up_axis="Z")
    hook = MultiWorldSameSideMimicHook(WHEEL_NAMES, expected_world_count=world_count)
    with wp.ScopedDevice("cpu"):
        for world_index in range(world_count):
            builder.begin_world()
            for side in ("left", "right"):
                for wheel_name in WHEEL_NAMES[side]:
                    label = f"/World/envs/env_{world_index}/Robot/base_link/{wheel_name}"
                    body = builder.add_link(
                        mass=1.0,
                        com=wp.vec3(0.0, 0.0, 0.0),
                        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                        label=f"/World/envs/env_{world_index}/Robot/base_link/{side}_{wheel_name}_body",
                    )
                    joint = builder.add_joint_revolute(
                        -1,
                        body,
                        axis=(0.0, 1.0, 0.0),
                        label=label,
                        damping=0.0,
                        friction=0.0,
                        armature=0.0,
                    )
                    builder.add_articulation([joint])
            callback_world = int(builder.current_world)
            if callback_world != world_index:
                raise AssertionError(
                    f"Newton world numbering changed: loop={world_index}, current={callback_world}"
                )
            hook(builder, callback_world, np.zeros(3), np.asarray([0., 0., 0., 1.]))
            builder.end_world()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    audit = audit_builder_and_solver(hook, model, solver)
    def count_fields(owner):
        result = {"type": f"{type(owner).__module__}.{type(owner).__name__}"}
        for name in (
            "world_count", "nworld", "nq", "nv", "njnt", "nbody", "neq", "nconmax",
            "qpos", "qvel", "time", "efc_force", "efc_vel",
        ):
            if not hasattr(owner, name):
                continue
            value = getattr(owner, name)
            if isinstance(value, (int, float, str, bool)):
                result[name] = value
            elif hasattr(value, "shape"):
                result[name] = {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", "unknown"))}
        return result
    return {
        "status": "passed",
        "model_device": str(model.device),
        "builder_world_count": int(builder.world_count),
        "builder_current_world_after_end": int(builder.current_world),
        "builder_joint_count": len(builder.joint_label),
        "builder_joint_world_counts": {
            str(world): int(np.count_nonzero(np.asarray(builder.joint_world) == world))
            for world in range(world_count)
        },
        "newton_model_layout": count_fields(model),
        "solver_host_model_layout": count_fields(solver.mj_model),
        "solver_warp_model_layout": count_fields(solver.mjw_model),
        "solver_host_data_layout": count_fields(solver.mj_data),
        "solver_warp_data_layout": count_fields(solver.mjw_data),
        "solver_attributes": sorted(name for name in dir(solver) if not name.startswith("__")),
        "audit": audit,
    }


def main() -> int:
    if OUTPUT_PATH.exists() or EXIT_PATH.exists():
        raise FileExistsError("refusing to overwrite CPU preflight evidence")
    one = _build_and_audit(1)
    two = _build_and_audit(2)
    canonical_one = one["audit"]["canonical_mechanism_metadata"]
    canonical_two = two["audit"]["canonical_mechanism_metadata"]
    if canonical_one != canonical_two or canonical_one != mechanism_metadata():
        raise AssertionError("canonical mechanism metadata depends on the environment count")
    result = {
        "schema": "staged-same-side-mimic-multiworld-cpu-preflight-v1",
        "newton_version": version("newton"),
        "builder_public_fields": {
            "joint_world": hasattr(newton.ModelBuilder(), "joint_world"),
            "current_world": hasattr(newton.ModelBuilder(), "current_world"),
            "add_constraint_mimic": hasattr(newton.ModelBuilder(), "add_constraint_mimic"),
            "add_custom_values": hasattr(newton.ModelBuilder(), "add_custom_values"),
        },
        "one_world": one,
        "two_world": two,
        "canonical_metadata_equal_across_world_counts": True,
        "canonical_mechanism_metadata": canonical_one,
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    EXIT_PATH.write_text(json.dumps({"actual_exit_code": 0, "device": "cpu"}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        EXIT_PATH.write_text(json.dumps({"actual_exit_code": 1, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
