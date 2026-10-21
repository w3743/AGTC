"""Short native Newton A/B/C ground and wheel-spin diagnostic; no training or FPS."""
from __future__ import annotations

import json
from pathlib import Path
import argparse
from datetime import datetime
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/isaac_field_diagnosis/ground_contact"
SEED, DT = 80_017, 0.05
PHASES = (("forward", (0.12, 0.0, 0.0, 0.0)), ("forward_zero", (0., 0., 0., 0.)),
          ("mix", (0.12, 0.05, 0.0, 0.0)), ("mix_zero", (0., 0., 0., 0.)))
WHEEL_JOINTS = ("left_wheel_3_wheel_joint", "right_wheel_3_wheel_joint")


def arr(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def make_visual_only_usd(source: Path, output: Path) -> Path:
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(source))
    prim = stage.GetPrimAtPath("/Terrain/Ground")
    if not prim.IsValid():
        raise RuntimeError(f"terrain ground prim missing in {source}")
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"could not export diagnostic terrain: {output}")
    return output


def inspect_model(env) -> tuple[dict[str, Any], dict[str, int]]:
    model = env.sim.physics_manager.get_model()
    labels = list(getattr(model, "shape_label", []) or [])
    kinds = arr(model.shape_type).reshape(-1)
    try:
        import newton
        shape_rows = [{"label": label, "type": newton.GeoType(int(k)).name}
                      for label, k in zip(labels, kinds)]
    except Exception:
        shape_rows = [{"label": label, "type": int(k)} for label, k in zip(labels, kinds)]
    jlabels = list(getattr(model, "joint_label", []) or [])
    starts = arr(model.joint_qd_start).reshape(-1).astype(int).tolist()
    native = {}
    for name in WHEEL_JOINTS:
        hits = [(i, s) for i, (lab, s) in enumerate(zip(jlabels, starts)) if lab.endswith("/" + name)]
        if hits:
            native[name] = {"joint_index": hits[0][0], "qd_offset": hits[0][1], "label": jlabels[hits[0][0]]}
    wanted = [r for r in shape_rows if any(x in r["label"] for x in
              ("Ground", "DiagnosticGround", "left_wheel_3", "right_wheel_3"))]
    return {"shape_count": len(shape_rows), "all_shape_labels_and_types": shape_rows,
            "selected_shape_labels_and_types": wanted,
            "native_joint_qd_offsets": native,
            "native_joint_order_sample": jlabels[:24]}, native


def quat_y(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)], dtype=float)


def quat_x(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)], dtype=float)


def sample(env, adapter, wheel_ids: list[int], target_ids: list[int], body_ids: list[int], native_offsets: dict[str, int],
           previous_q: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    from isaaclab_newton.physics import NewtonManager
    pose = arr(env.rover.data.root_link_pose_w)[0].astype(float)
    root_v = arr(env.rover.data.root_link_vel_w)[0].astype(float)
    root_omega = root_v[3:6]
    q = arr(env.rover.data.joint_pos)[0, wheel_ids].astype(float)
    qd = arr(env.rover.data.joint_vel)[0, wheel_ids].astype(float)
    fd_raw = None if previous_q is None else (q - previous_q) / DT
    fd_wrapped = None if previous_q is None else ((q - previous_q + np.pi) % (2*np.pi) - np.pi) / DT
    spins, axes = [], []
    body_omega = arr(env.rover.data.body_link_ang_vel_w)[0, body_ids].astype(float)
    body_q = arr(env.rover.data.body_link_quat_w)[0, body_ids].astype(float)
    for omega, quat in zip(body_omega, body_q):
        axis = quat_y(quat)
        axes.append(axis.tolist())
        spins.append(float(np.dot(omega - root_omega, axis)))
    target = arr(env.wheel_target_velocity)[0, target_ids].astype(float)
    state = NewtonManager.get_state_0()
    raw_qd = arr(state.joint_qd).reshape(-1)
    raw_native = [float(raw_qd[native_offsets[n]["qd_offset"]]) if n in native_offsets else None
                  for n in WHEEL_JOINTS]
    contacts = NewtonManager.get_contacts()
    pair_count = wheel_ground = None
    if contacts is not None:
        pair_count = int(arr(contacts.rigid_contact_count).reshape(-1)[0])
        s0, s1 = arr(contacts.rigid_contact_shape0).reshape(-1), arr(contacts.rigid_contact_shape1).reshape(-1)
        labels = list(env.sim.physics_manager.get_model().shape_label or [])
        ground_ids = {i for i, lab in enumerate(labels) if "ground" in lab.lower() and "robot" not in lab.lower()}
        wheel_ids_shape = {i for i, lab in enumerate(labels) if "_wheel_" in lab and "Robot" in lab}
        wheel_ground = sum(1 for a, b in zip(s0[:pair_count], s1[:pair_count])
                           if (int(a) in ground_ids and int(b) in wheel_ids_shape) or
                              (int(b) in ground_ids and int(a) in wheel_ids_shape))
    sensor = arr(env.scene["wheel_contacts"].data.net_forces_w)
    sensor = sensor[0] if sensor.ndim == 3 else sensor
    fsel = sensor[[3, 10]].astype(float)
    row = {"time_s": float(adapter.time), "root_xyz_m": pose[:3].tolist(),
           "root_quaternion_xyzw": pose[3:].tolist(),
           "root_linear_velocity_world_mps": root_v[:3].tolist(),
           "wheel_target_rad_s": target.tolist(), "public_joint_velocity_rad_s": qd.tolist(),
           "wheel_joint_position_rad": q.tolist(),
           "joint_position_finite_difference_raw_rad_s": None if fd_raw is None else fd_raw.tolist(),
           "joint_position_finite_difference_wrapped_rad_s": None if fd_wrapped is None else fd_wrapped.tolist(),
           "native_model_qd_at_joint_offsets_rad_s": raw_native,
           "wheel_body_spin_minus_base_projected_on_world_axle_rad_s": spins,
           "wheel_world_axle_y": axes, "wheel_contact_force_world_N": fsel.tolist(),
           "native_contact_pair_count": pair_count, "native_wheel_ground_pair_count": wheel_ground}
    return row, q


def run_case(case: str) -> dict[str, Any]:
    import isaac_field.env as env_module
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import FieldEnvCfg, IsaacFieldEnv, SingleFieldAdapter
    from isaac_field.task import Config
    from isaaclab import sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import ContactSensorCfg

    cfg_task = Config(size=100., plants=0, layout="rows", terrain_enabled=False,
                      validate_scene=False, max_time=30.)
    cfg = FieldEnvCfg(task=cfg_task, mode="goto", auto_reset=False, seed=SEED)
    cfg.scene.num_envs = 1
    cfg.scene.wheel_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
                                                update_period=0., history_length=1)
    cfg.drive_velocity_gain = 0.5 if case == "C" else 20.
    if case in ("B", "D"):
        from isaaclab.sim.spawners.from_files.from_files import spawn_ground_plane
        from pxr import UsdGeom
        def spawn_collision_only_plane(prim_path, cfg, **kwargs):
            prim = spawn_ground_plane(prim_path, cfg, **kwargs)
            stage = prim.GetStage()
            visual = stage.GetPrimAtPath(f"{prim_path}/Environment")
            if visual.IsValid():
                UsdGeom.Imageable(visual).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            return prim
        cfg.scene.diagnostic_ground = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/DiagnosticGround",
            spawn=sim_utils.GroundPlaneCfg(size=(100., 100.), physics_material=
                sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8),
                func=spawn_collision_only_plane))
    if case == "D":
        cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    if case in ("B", "D"):
        from dataclasses import asdict
        import hashlib
        fp = hashlib.sha256(json.dumps({"config": asdict(cfg_task), "seed": cfg.seed,
                                        "asset_revision": 8}, sort_keys=True).encode()).hexdigest()[:16]
        terrain_path = ROOT / "results/isaac_field/assets" / fp / "terrain.usda"
        if not terrain_path.exists():
            raise FileNotFoundError("no generated flat terrain USD to copy")
        field_usd = make_visual_only_usd(terrain_path, OUT_DIR / f"terrain_visual_only_{fp}.usda")
    else:
        field_usd = None
    original_asset_cfg = env_module.AssetBaseCfg
    def asset_cfg(prim_path, spawn=None, **kwargs):
        if field_usd is not None and prim_path == "{ENV_REGEX_NS}/Field":
            spawn = sim_utils.UsdFileCfg(usd_path=str(field_usd))
        return original_asset_cfg(prim_path=prim_path, spawn=spawn, **kwargs)
    env_module.AssetBaseCfg = asset_cfg
    try:
        env = IsaacFieldEnv(cfg)
    finally:
        env_module.AssetBaseCfg = original_asset_cfg
    adapter = SingleFieldAdapter(env)
    try:
        names = [f"{w.name}_wheel_joint" for w in ROBOT_SPEC.wheel_specs]
        wheel_ids = [env.rover.joint_names.index(n) for n in names]
        selected_ids = [wheel_ids[3], wheel_ids[10]]
        target_ids = [3, 10]
        body_ids = [env.rover.body_names.index(n.removesuffix("_wheel_joint") + "_wheel_link")
                    for n in WHEEL_JOINTS]
        model_info, offsets = inspect_model(env)
        result = {"case": case, "ground": "plane_plus_visual_only_flat_mesh" if case in ("B", "D") else "flat_triangle_mesh",
                  "drive_velocity_gain": cfg.drive_velocity_gain, "seed": SEED,
                  "native_model": model_info, "phases": {}, "status": "running"}
        adapter.reset(seed=SEED, options={"mode": "goto", "start": [50., 50., 0.], "goal": [90., 50.],
                                          "random_start": False})
        for _ in range(20):
            adapter.step(np.zeros(4, dtype=np.float32))
        previous_q = arr(env.rover.data.joint_pos)[0, selected_ids].astype(float)
        previous_p = arr(env.rover.data.root_link_pose_w)[0, :3].astype(float)
        for phase, command in PHASES:
            rows = []
            for step in range(40):
                adapter.step(np.asarray(command, dtype=np.float32))
                row, previous_q = sample(env, adapter, selected_ids, target_ids, body_ids,
                                         offsets, previous_q)
                row["phase_step"] = step + 1
                rows.append(row)
            last_p = np.asarray(rows[-1]["root_xyz_m"])
            speed = np.asarray([r["root_linear_velocity_world_mps"] for r in rows])
            joint = np.asarray([r["public_joint_velocity_rad_s"] for r in rows])
            spin = np.asarray([r["wheel_body_spin_minus_base_projected_on_world_axle_rad_s"] for r in rows])
            fd_raw = np.asarray([r["joint_position_finite_difference_raw_rad_s"] for r in rows
                                 if r["joint_position_finite_difference_raw_rad_s"] is not None])
            fd_wrapped = np.asarray([r["joint_position_finite_difference_wrapped_rad_s"] for r in rows
                                     if r["joint_position_finite_difference_wrapped_rad_s"] is not None])
            forward = np.asarray([quat_x(np.asarray(r["root_quaternion_xyzw"])) for r in rows])
            root_forward = np.sum(speed * forward, axis=1)
            phase_sum = {"steps": len(rows), "start_xyz_m": previous_p.tolist(), "end_xyz_m": last_p.tolist(),
                "displacement_xyz_m": (last_p - previous_p).tolist(),
                "max_abs_root_forward_velocity_mps": float(np.max(np.abs(root_forward))),
                "max_abs_public_wheel_surface_mps": float(np.max(np.abs(joint * ROBOT_SPEC.wheel_radius_m))),
                "max_abs_wheel_body_spin_surface_mps": float(np.max(np.abs(spin * ROBOT_SPEC.wheel_radius_m))),
                "max_abs_public_vs_body_spin_rad_s": float(np.max(np.abs(joint - spin))),
                "max_abs_raw_position_fd_vs_public_rad_s": float(np.max(np.abs(fd_raw - joint))) if len(fd_raw) else None,
                "max_abs_wrapped_position_fd_vs_public_rad_s": float(np.max(np.abs(fd_wrapped - joint))) if len(fd_wrapped) else None,
                "max_native_contact_pairs": max((r["native_contact_pair_count"] or 0) for r in rows),
                "max_wheel_ground_contact_pairs": max((r["native_wheel_ground_pair_count"] or 0) for r in rows),
                "max_abs_selected_wheel_contact_force_N": float(np.max(np.linalg.norm(
                    np.asarray([r["wheel_contact_force_world_N"] for r in rows]), axis=2)))}
            result["phases"][phase] = {"command": list(command), "summary": phase_sum, "samples": rows}
            previous_p = last_p
            print(json.dumps({"case": case, "phase": phase, "summary": phase_sum}), flush=True)
        result["status"] = "complete"
    finally:
        adapter.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("A", "B", "C", "D"), help="run one case only")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [args.case] if args.case else ["A", "B", "C", "D"]
    output = OUT_DIR / f"ground_contact_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    doc = {"schema": "isaac-field-newton-ground-contact-ab-v1", "setup": {
        "physics": "Isaac Lab Newton MJWarp, use_mujoco_contacts=False", "plants": 0, "seed": SEED,
        "control_dt_s": DT, "settle_zero_steps": 20, "phase_steps": 40,
        "phases": [{"name": n, "command": list(a)} for n, a in PHASES],
        "wheel_body_spin": "body_link_ang_vel_w minus root angular velocity, projected on wheel local-Y transformed to world",
        "wheel_samples": list(WHEEL_JOINTS), "target_wheel_indices": [3, 10],
        "finite_difference": "raw and wrapped joint_pos delta / 0.05 s",
        "case_definitions": {"A": "flat triangle mesh, drive gain 20", "B": "visual-only mesh plus GroundPlaneCfg, gain 20",
                             "C": "flat triangle mesh, drive gain 0.5",
                             "D": "same as B, gain 20, use_mujoco_contacts=True"}}, "cases": {}, "status": "running"}
    for case in cases:
        try:
            doc["cases"][case] = run_case(case)
            doc["status"] = f"complete:{case}"
        except Exception as exc:
            doc["cases"][case] = {"status": f"failed:{type(exc).__name__}:{exc}"}
            doc["status"] = f"failed:{case}"
            output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
