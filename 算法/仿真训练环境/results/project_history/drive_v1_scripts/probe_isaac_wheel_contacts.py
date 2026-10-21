"""Inspect wheel loads with Isaac Lab's native Newton contact sensor."""
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab.sensors import ContactSensorCfg
from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
from isaac_field.task import Config
from scripts.diagnose_isaac_skid_response import _numpy


def run():
    cfg = FieldEnvCfg(task=Config(plants=0, terrain_enabled=False,
                                 validate_scene=False, max_time=10.),
                      mode="goto", auto_reset=False, seed=80003)
    cfg.scene.num_envs = 1
    cfg.scene.wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
        update_period=0., history_length=1, debug_vis=False)
    env = IsaacFieldEnv(cfg)
    rows = []
    try:
        env.reset(seed=80003, options={"start": [50., 50., 0.], "goal": [52., 50.]})
        env.rover.write_joint_damping_to_sim_index(
            damping=torch.full((1, 14), 20., device=env.device), joint_ids=env._wheel_ids)
        sensor = env.scene["wheel_contacts"]
        model = env.sim.physics_manager.get_model()
        masses = _numpy(model.body_mass).tolist()
        for step in range(60):
            action = torch.tensor([[0., 0. if step < 20 else .7, 0., 0.]], device=env.device)
            env.step(action)
            forces = _numpy(sensor.data.net_forces_w).reshape(-1, 3)
            contacts = env.sim.physics_manager.get_contacts()
            count = int(_numpy(contacts.rigid_contact_count).reshape(-1)[0])
            rows.append({"step": step, "time": env.tasks[0].time,
                         "pose_v_w": env.tasks[0].robot.tolist(),
                         "net_wheel_forces_w": forces.tolist(),
                         "sum_force_z_n": float(forces[:, 2].sum()),
                         "contact_count": count,
                         "contact_capacity": int(contacts.rigid_contact_max)})
        report = {"schema": "isaac-field-wheel-contacts-v1", "body_masses_kg": masses,
                  "total_mass_kg": float(np.sum(masses)),
                  "sensor_bodies": list(sensor.body_names), "rows": rows,
                  "scope": "Native Newton contact sensor; no custom force reconstruction."}
        out = Path("results/isaac_field_diagnosis/skid_response/contact_probe.json")
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"total_mass_kg": report["total_mass_kg"],
                          "static_force_z_mean_n": float(np.mean([r["sum_force_z_n"] for r in rows[10:20]])),
                          "spin_force_z_mean_n": float(np.mean([r["sum_force_z_n"] for r in rows[-10:]])),
                          "max_contact_count": max(r["contact_count"] for r in rows),
                          "capacity": rows[-1]["contact_capacity"], "output": str(out)}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    run()
