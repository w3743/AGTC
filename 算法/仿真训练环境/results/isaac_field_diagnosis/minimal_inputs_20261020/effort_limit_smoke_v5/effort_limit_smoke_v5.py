#!/usr/bin/env python3
"""Post-fix Newton actuator effort-limit smoke; not training, PI, or FPS."""
from __future__ import annotations
import hashlib, json, math, os, runpy, traceback
from pathlib import Path
import numpy as np
import torch
import warp as wp

@wp.kernel
def _record(applied: wp.array2d(dtype=wp.float32), ids: wp.array(dtype=wp.int32), tape: wp.array2d(dtype=wp.float32), counter: wp.array(dtype=wp.int32), capacity: int):
    slot = wp.atomic_add(counter, 0, 1)
    if slot < capacity:
        for j in range(14):
            tape[slot, j] = applied[0, ids[j]]

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _array_values(value):
    try:
        return np.asarray(value.numpy()).tolist()
    except Exception:
        try:
            return value.detach().cpu().numpy().tolist()
        except Exception:
            return None

def main():
    if os.environ.get("RUN_ISAAC_PHYSICS") != "1":
        raise SystemExit("Refusing Newton smoke without RUN_ISAAC_PHYSICS=1")
    out = Path(__file__).resolve().parent
    summary_path, samples_path = out / "effort_limit_v5_summary.json", out / "effort_limit_v5_substeps.jsonl"
    if summary_path.exists() or samples_path.exists():
        raise SystemExit(f"Refusing to overwrite outputs in {out}")
    helper = runpy.run_path("tests/test_env_integration.py")
    make_env, step = helper["_make_env"], helper["_step"]
    from isaaclab_newton.physics import NewtonManager
    official_newton = Path("/home/wangj/robotics-lab/venv/isaaclab-field/lib/python3.12/site-packages/newton")
    source_paths = (
        "isaac_field/env.py", "scripts/train_isaac_foundation_fresh.py", "tests/test_env_integration.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py",
        "external/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/newton/kernels.py",
        official_newton / "_src/actuators/clamping/clamping_max_effort.py",
        official_newton / "_src/solvers/mujoco/kernels.py",
        official_newton / "_src/solvers/mujoco/solver_mujoco.py",
    )
    env = None
    try:
        env = make_env(plants=0, max_time=20.0, terrain_enabled=False, seed=90231)
        dt, decimation = float(env.cfg.sim.dt), int(env.cfg.decimation)
        if not math.isclose(dt, 0.005, abs_tol=1e-12) or decimation != 10:
            raise RuntimeError(f"unexpected cadence: dt={dt}, decimation={decimation}")
        env.reset(seed=90231, options={"mode":"search", "start":[50.0,50.0,0.0], "goal":[90.0,50.0]})
        ids = env._wheel_ids.to(dtype=torch.int32).contiguous()
        ids_warp = wp.from_torch(ids)
        capacity = 2048
        tape, counter = wp.empty((capacity,14), dtype=wp.float32, device=env.device), wp.zeros(1, dtype=wp.int32, device=env.device)
        applied_warp = env.rover.actuators.applied_effort.warp
        def capture():
            wp.launch(_record, dim=1, inputs=[applied_warp, ids_warp, tape, counter, capacity], device=env.device)
        NewtonManager.register_post_actuator_callback(capture)
        original_apply = env._apply_action
        active = {"name":None, "ff":None}
        boundaries = []
        zero = torch.zeros_like(env.wheel_target_velocity)
        def patched_apply():
            if active["name"] is not None:
                previous = env.rover.actuators.applied_effort.torch[:,env._wheel_ids].detach().cpu().numpy()[0]
                boundaries.append({"case":active["name"], "previous_policy_boundary_applied_effort_nm":[float(x) for x in previous]})
            original_apply()
            env.rover.actuators.target_command.set_velocity_index(value=zero, joint_ids=env._wheel_ids)
            ff = active["ff"] if active["ff"] is not None else zero
            env.rover.actuators.target_command.set_effort_index(value=ff, joint_ids=env._wheel_ids)
        env._apply_action = patched_apply

        model = env.sim.physics_manager.get_model()
        actuator_rows = []
        for index, actuator in enumerate(getattr(model, "actuators", ())):
            clamps = []
            for clamp in (getattr(actuator, "clamping", None) or []):
                clamps.append({"type":type(clamp).__name__, "max_effort_nm":_array_values(getattr(clamp,"max_effort",None))})
            actuator_rows.append({"index":index, "type":type(actuator).__name__, "controller":type(getattr(actuator,"controller",None)).__name__, "clamping":clamps})
        joint_limits = np.asarray(env.rover.data.joint_effort_limits.torch.detach().cpu())[0]
        gimbal_ids = env._gimbal_ids.detach().cpu().numpy().astype(int)
        drive_evidence = {
            "interface": env.drive_config.get("interface"),
            "drive_config": env.drive_config,
            "native_actuator_count":len(actuator_rows),
            "native_actuators":actuator_rows,
            "drive_joint_effort_limits_nm":np.asarray(env.rover.data.joint_effort_limits.torch[:,env._wheel_ids].detach().cpu())[0].tolist(),
            "gimbal_joint_names":[env.rover.joint_names[i] for i in gimbal_ids],
            "gimbal_solver_joint_effort_limits_nm":joint_limits[gimbal_ids].tolist(),
            "gimbal_actuator_cfg":{
                "class":type(env.cfg.scene.robot.actuators["gimbal"]).__name__,
                "joint_effort_limit":float(env.cfg.scene.robot.actuators["gimbal"].joint_effort_limit),
                "actuator_effort_limit":env.cfg.scene.robot.actuators["gimbal"].actuator_effort_limit,
            },
        }
        side = env._wheel_track_ids
        left, right = side == 0, side == 1
        if int(left.sum()) != 7 or int(right.sum()) != 7:
            raise RuntimeError("expected seven wheel joints per side")
        patterns = (("same_positive",4.0,4.0),("same_negative",-4.0,-4.0),("opposed_left_positive",4.0,-4.0),("opposed_left_negative",-4.0,4.0))
        cases=[]
        zero_action=np.zeros(4,dtype=np.float32)
        with samples_path.open("w",encoding="utf-8",newline="\n") as samples:
            for name,left_nm,right_nm in patterns:
                env.reset(seed=90231,options={"mode":"search","start":[50.0,50.0,0.0],"goal":[90.0,50.0]})
                active["name"],active["ff"]=None,None
                step(env,zero_action,count=round(1.0/float(env.step_dt)))
                ff=torch.zeros_like(env.wheel_target_velocity)
                ff[:,left],ff[:,right]=left_nm,right_nm
                active["name"],active["ff"]=name,ff
                first=int(counter.numpy()[0])
                step(env,zero_action,count=2)
                last=int(counter.numpy()[0])
                if last-first != 2*decimation or last>capacity:
                    raise RuntimeError(f"substep coverage mismatch: wanted {2*decimation}, got {last-first}")
                rows=np.asarray(tape.numpy()[first:last],dtype=np.float64)
                lv=rows[:,left.detach().cpu().numpy().astype(bool)]
                rv=rows[:,right.detach().cpu().numpy().astype(bool)]
                max_abs=float(np.max(np.abs(rows)))
                within=bool(max_abs<=2.0+1e-5)
                case={"name":name,"requested_ff_nm":{"left":left_nm,"right":right_nm},"target_wheel_velocity_rad_s":0.0,"independent_reset":True,"settle_s":1.0,"policy_ticks":2,"physics_iterations":int(last-first),"left_applied_effort_min_max_nm":[float(lv.min()),float(lv.max())],"right_applied_effort_min_max_nm":[float(rv.min()),float(rv.max())],"max_abs_applied_effort_nm":max_abs,"all_physics_samples_within_2nm_plus_1e-5":within,"per_substep_applied_effort_nm":rows.tolist()}
                cases.append({k:v for k,v in case.items() if k!="per_substep_applied_effort_nm"})
                for j,row in enumerate(rows):
                    samples.write(json.dumps({"case":name,"substep_in_case":j,"physics_iteration_global":first+j,"dt_s":dt,"requested_ff_nm":{"left":left_nm,"right":right_nm},"wheel_names":env._wheel_names,"applied_effort_nm":[float(x) for x in row]},sort_keys=True,allow_nan=False)+"\n")
                samples.flush()
                print("V5 EFFORT CASE "+json.dumps(cases[-1],sort_keys=True),flush=True)
                active["name"],active["ff"]=None,None
        cap_actual=all(c["all_physics_samples_within_2nm_plus_1e-5"] for c in cases)
        max_actual=max(c["max_abs_applied_effort_nm"] for c in cases)
        result={
            "schema":"isaac-field-effort-limit-smoke-v2",
            "purpose":"post-fix native effort clamp validation; isolated physics command diagnostic, not training, PI, FPS, or motor rating",
            "execution_status":"completed",
            "cap_assertion_status":"pass" if cap_actual else "limit_not_enforced",
            "cap_assertion_tolerance_nm":1e-5,
            "max_abs_applied_effort_nm":max_actual,
            "runtime_actuator_and_gimbal_limits":drive_evidence,
            "environment":{"helper":"tests/test_env_integration.py::_make_env/_step","plants":0,"terrain_enabled":False,"seed":90231,"start_xy_yaw":[50.0,50.0,0.0],"goal_xy":[90.0,50.0],"physics_dt_s":dt,"policy_dt_s":float(env.step_dt),"decimation":decimation,"action_all_zero":True,"patterns":len(patterns),"test_policy_ticks_per_pattern":2,"physics_iterations_per_pattern":2*decimation},
            "cases":cases,
            "policy_boundary_previous_effort_reads":boundaries,
            "source_sha256":{str(path):_sha(path) for path in source_paths},
        }
        with summary_path.open("w",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write("\n")
        print("V5 EFFORT SMOKE "+json.dumps({"execution_status":result["execution_status"],"cap_assertion_status":result["cap_assertion_status"],"max_abs_applied_effort_nm":max_actual,"case_count":len(cases),"summary":str(summary_path),"samples":str(samples_path)},sort_keys=True),flush=True)
    except Exception as exc:
        failure={"schema":"isaac-field-effort-limit-smoke-v2","execution_status":"failed","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}
        with (Path(__file__).resolve().parent/"effort_limit_v5_failure.json").open("w",encoding="utf-8",newline="\n") as stream:
            json.dump(failure,stream,indent=2,sort_keys=True); stream.write("\n")
        raise
    finally:
        if env is not None: env.close()

if __name__ == "__main__": main()