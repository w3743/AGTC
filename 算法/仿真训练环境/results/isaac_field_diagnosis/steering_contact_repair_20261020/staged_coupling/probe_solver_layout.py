from __future__ import annotations

import json
from pathlib import Path

from cpu_two_world_preflight import _build_and_audit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "solver_layout_probe.json"
EXIT = ROOT / "solver_layout_probe.exit.json"


if OUT.exists() or EXIT.exists():
    raise FileExistsError("refusing to overwrite solver layout probe")
one = _build_and_audit(1)
two = _build_and_audit(2)
result = {
    "schema": "staged-mimic-solver-layout-probe-v1",
    "device": "cpu",
    "one_world": {
        "newton_model": one["newton_model_layout"],
        "solver_host": one["solver_host_model_layout"],
        "solver_warp": one["solver_warp_model_layout"],
        "solver_attribute_names": one["solver_attributes"],
        "mimic_count": one["audit"]["mimic_constraints"],
        "equality_custom_rows": one["audit"]["mujoco_equality_custom_rows"],
        "solver_neq": one["audit"]["solver_host_neq"],
    },
    "two_worlds": {
        "newton_model": two["newton_model_layout"],
        "solver_host": two["solver_host_model_layout"],
        "solver_warp": two["solver_warp_model_layout"],
        "solver_attribute_names": two["solver_attributes"],
        "mimic_count": two["audit"]["mimic_constraints"],
        "equality_custom_rows": two["audit"]["mujoco_equality_custom_rows"],
        "solver_neq": two["audit"]["solver_host_neq"],
    },
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
EXIT.write_text('{"actual_exit_code":0,"device":"cpu"}\n', encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
