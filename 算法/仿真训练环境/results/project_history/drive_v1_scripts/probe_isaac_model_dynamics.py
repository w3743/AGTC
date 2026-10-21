"""Read public Newton mass, inertia, COM, and joint target metadata once."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import newton
from isaac_field.env import make_env
from isaac_field.task import Config

OUT = Path(__file__).resolve().parents[1] / "results/isaac_field_diagnosis/skid_response/model_dynamics.json"
env = make_env(cfg=Config(plants=0, terrain_enabled=False, validate_scene=False), mode="goto",
               num_envs=1, seed=80_003, auto_reset=False, distance_cost=0.0)
try:
    model = env.sim.physics_manager.get_model()
    def arr(*names):
        for name in names:
            try: return name, wp.to_torch(getattr(model, name)).detach().cpu().numpy().tolist()
            except (AttributeError, TypeError): pass
        return None, None
    bodies = list(model.body_label); joints = list(model.joint_label)
    base_ids = [i for i, label in enumerate(bodies) if str(label).endswith("/base_link")]
    values = {key: arr(*candidates) for key, candidates in {
        "mass": ("body_mass",), "com": ("body_com", "body_com_pos"),
        "inertia": ("body_inertia", "body_I"), "joint_type": ("joint_type",),
        "target_ke": ("joint_target_ke",), "target_kd": ("joint_target_kd",),
    }.items()}
    joint_types = np.asarray(values["joint_type"][1]).reshape(-1).tolist()
    target_ke = np.asarray(values["target_ke"][1]).reshape(-1).tolist()
    target_kd = np.asarray(values["target_kd"][1]).reshape(-1).tolist()
    dofs = {"FREE": 6, "BALL": 3, "REVOLUTE": 1, "PRISMATIC": 1, "FIXED": 0}
    joint_rows, offset = [], 0
    for i, (label, type_id) in enumerate(zip(joints, joint_types)):
        type_name = newton.JointType(int(type_id)).name
        count = dofs.get(type_name)
        joint_rows.append({"index": i, "label": str(label), "type_id": int(type_id), "type_name": type_name,
                           "dof_start": offset, "dof_count": count,
                           "target_ke": target_ke[offset:offset + count] if count is not None else None,
                           "target_kd": target_kd[offset:offset + count] if count is not None else None})
        if count is not None: offset += count
    out = {"body_labels": bodies, "base_link_ids": base_ids,
           "base_link": {key: (data[1][base_ids[0]] if data[1] and base_ids else None)
                         for key, data in values.items() if key in ("mass", "com", "inertia")},
           "model_field_names": {key: data[0] for key, data in values.items()},
           "joint_target_ke_dof_count": len(target_ke), "joint_target_kd_dof_count": len(target_kd),
           "joints": joint_rows}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)
finally:
    env.close()
