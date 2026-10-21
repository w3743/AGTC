from __future__ import annotations

import inspect
import json

import newton
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo


builder = ModelBuilder(up_axis="Z")
fields = {}
for name in ("joint_world", "current_world", "world_count", "add_constraint_mimic", "add_custom_values", "begin_world", "end_world", "add_body", "add_joint_revolute", "finalize"):
    value = getattr(builder, name, None)
    fields[name] = {
        "exists": hasattr(builder, name),
        "type": type(value).__name__,
        "value": value if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))) else None,
    }
    if callable(value):
        try:
            fields[name]["signature"] = str(inspect.signature(value))
        except (TypeError, ValueError):
            pass
print(json.dumps({"newton_file": newton.__file__, "version": getattr(newton, "__version__", None), "fields": fields}, sort_keys=True))
print(json.dumps({"solver_mujoco_signature": str(inspect.signature(SolverMuJoCo)), "solver_mujoco_init": str(inspect.signature(SolverMuJoCo.__init__))}, sort_keys=True))
