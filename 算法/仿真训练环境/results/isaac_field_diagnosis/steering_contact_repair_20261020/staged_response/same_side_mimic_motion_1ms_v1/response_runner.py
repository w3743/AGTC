#!/usr/bin/env python3
"""Run the frozen eight-motion collector with or without the result-only mimic hook."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parents[5]
EXP = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
STAGED = EXP / "staged_response"
PROBE_PATH = STAGED / "staged_motion_response_probe.py"
HOOK_PATH = EXP / "same_side_mimic_v2/same_side_mimic_hook.py"
MIMIC_PROTOCOL = EXP / "same_side_mimic_v2/protocol.json"
RUN_ROOT = STAGED / "same_side_mimic_motion_1ms_v1/runs"
RUNNER_PATH = Path(__file__).resolve()
PAIR_NAMES = [
    {"side": side,
     "leader_name": f"{side}_wheel_0_wheel_joint",
     "follower_name": f"{side}_wheel_{follower}_wheel_joint"}
    for side in ("left", "right") for follower in range(1, 7)
]
PHYSICS_DT = 0.001
CONE = "elliptic"
IMPRATIO = 1.0
MU = 0.8
KF = 1000.0
KP = 0.04
KI = 0.1
EFFORT_CAP = 2.0
TARGET_WHEEL_SPEED = 9.25925925925926


@wp.kernel
def capture_pair_qd_residuals(
    state_qd: wp.array(dtype=wp.float32),
    pair_offsets: wp.array(dtype=wp.int32),
    tape: wp.array2d(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    capacity: int,
):
    slot = wp.atomic_add(counter, 0, 1)
    if slot < capacity:
        for pair in range(12):
            leader = pair_offsets[pair * 2]
            follower = pair_offsets[pair * 2 + 1]
            tape[slot, pair] = state_qd[follower] - state_qd[leader]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"collector anchor {label!r} matched {count} times; expected exactly once")
    return source.replace(old, new, 1)


def _host(value):
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def resolve_pair_layout(model) -> tuple[list[dict], list[int]]:
    labels = [str(value) for value in getattr(model, "joint_label", [])]
    q_starts = _host(getattr(model, "joint_q_start")).reshape(-1).astype(np.int64)
    qd_starts = _host(getattr(model, "joint_qd_start")).reshape(-1).astype(np.int64)
    if len(q_starts) < len(labels) or len(qd_starts) < len(labels):
        raise RuntimeError("native q/qd offsets do not cover all final model joints")
    layout, flat_qd_offsets = [], []
    for pair in PAIR_NAMES:
        row = dict(pair)
        for role in ("leader", "follower"):
            basename = pair[f"{role}_name"]
            hits = [index for index, label in enumerate(labels)
                    if label.rsplit("/", 1)[-1] == basename]
            if len(hits) != 1:
                raise RuntimeError(f"expected one native joint {basename!r}; found {len(hits)}")
            index = hits[0]
            q_offset, qd_offset = int(q_starts[index]), int(qd_starts[index])
            row[f"{role}_label"] = labels[index]
            row[f"{role}_q_offset"] = q_offset
            row[f"{role}_qd_offset"] = qd_offset
            if q_offset < 0 or qd_offset < 0:
                raise RuntimeError(f"invalid native offsets for {basename}: {q_offset}, {qd_offset}")
        flat_qd_offsets.extend((row["leader_qd_offset"], row["follower_qd_offset"]))
        layout.append(row)
    return layout, flat_qd_offsets


def read_policy_pair_state(model, state, layout: list[dict]) -> dict:
    q = _host(getattr(state, "joint_q")).reshape(-1)
    qd = _host(getattr(state, "joint_qd")).reshape(-1)
    relations = []
    for pair in layout:
        lq, fq = pair["leader_q_offset"], pair["follower_q_offset"]
        lqd, fqd = pair["leader_qd_offset"], pair["follower_qd_offset"]
        if max(lq, fq) >= len(q) or max(lqd, fqd) >= len(qd):
            raise RuntimeError("native state arrays are shorter than pre-resolved wheel offsets")
        relations.append({
            "side": pair["side"], "leader_name": pair["leader_name"],
            "follower_name": pair["follower_name"], "coef0": 0.0, "coef1": 1.0,
            "leader_q_rad": float(q[lq]), "follower_q_rad": float(q[fq]),
            "q_follower_minus_expected_rad": float(q[fq] - q[lq]),
            "leader_qd_rad_s": float(qd[lqd]), "follower_qd_rad_s": float(qd[fqd]),
            "qd_follower_minus_expected_rad_s": float(qd[fqd] - qd[lqd]),
        })
    return {
        "source": "NewtonManager.get_state_0() after policy step; offsets resolved once before motion",
        "constraint_semantics": "same-side reference q_follower = q_leader; residual is follower minus leader",
        "relation_count": len(relations), "relations": relations,
    }


def _solver_counts(solver) -> dict:
    result = {}
    for label, attr in (("host_neq", "mj_model"), ("mjwarp_neq", "mjw_model")):
        model = getattr(solver, attr, None)
        value = getattr(model, "neq", None) if model is not None else None
        try:
            result[label] = int(_host(value).reshape(-1)[0]) if value is not None else "NA"
        except Exception:
            result[label] = "NA"
    return result


def _load_modules():
    if not PROBE_PATH.is_file() or not HOOK_PATH.is_file():
        raise FileNotFoundError("staged collector or same-side public hook is missing")
    return (_load(PROBE_PATH, "staged_motion_response_probe_frozen"),
            _load(HOOK_PATH, "same_side_mimic_v2_public_hook"))


def prepare():
    probe, hook_module = _load_modules()
    source = inspect.getsource(probe.main)
    source = _replace_once(
        source,
        "        from isaaclab_newton.physics import NewtonManager\n",
        "        from isaaclab_newton.physics import NewtonManager\n"
        "        from isaaclab_newton.cloner import newton_builder_world_hook\n",
        "builder_hook_import",
    )
    source = _replace_once(
        source,
        "        env = IsaacFieldEnv(cfg)\n        physics = runtime_snapshot(cfg, env)\n",
        "        if MIMIC_MODE == \"mimic\":\n"
        "            mimic_hook = make_mimic_hook()\n"
        "            with newton_builder_world_hook(mimic_hook):\n"
        "                env = IsaacFieldEnv(cfg)\n"
        "            mimic_solver_audit = finalized_mimic_audit(\n"
        "                mimic_hook, NewtonManager.get_model(), NewtonManager._solver\n"
        "            )\n"
        "        else:\n"
        "            mimic_hook = None\n"
        "            env = IsaacFieldEnv(cfg)\n"
        "            mimic_solver_audit = {\"status\": \"no_hook\", **_solver_counts(NewtonManager._solver)}\n"
        "        physics = runtime_snapshot(cfg, env)\n",
        "conditional_mimic_builder_hook",
    )
    source = _replace_once(
        source,
        '        "output_paths": {"policy_trajectories": str(trajectories_path), "native_physics_efforts": str(efforts_path)},\n',
        '        "output_paths": {"policy_trajectories": str(trajectories_path), "native_physics_efforts": str(efforts_path),\n'
        '                         "native_same_side_qd_residuals": str(mimic_qd_path)},\n',
        "qd_raw_output_path",
    )
    source = _replace_once(
        source,
        '    efforts_path = out_dir / "native_physics_efforts.jsonl"\n',
        '    efforts_path = out_dir / "native_physics_efforts.jsonl"\n'
        '    mimic_qd_path = out_dir / "native_same_side_qd_residuals.jsonl"\n',
        "qd_raw_path",
    )
    source = _replace_once(
        source,
        '        summary["drive_config"] = env.drive_config\n\n'
        '        first_options = {"mode": "search", "start": START, "goal": GOAL}\n',
        '        summary["drive_config"] = env.drive_config\n'
        '        summary["same_side_treatment"] = {\n'
        '            "mode": MIMIC_MODE, "hook_enabled": mimic_hook is not None,\n'
        '            "solver_equality_audit": mimic_solver_audit,\n'
        '            "hook_sha256": sha_file(HOOK_PATH),\n'
        '            "hook_protocol_sha256": sha_file(MIMIC_PROTOCOL),\n'
        '            "runner_sha256": sha_file(RUNNER_PATH),\n'
        '            "collector_sha256": sha_file(PROBE_PATH),\n'
        '            "pair_layout": PAIR_NAMES,\n'
        '            "native_qd_capture": "pre-solver get_state_0() in post-actuator callback; Warp-only tape write",\n'
        '        }\n'
        '        summary["runtime_drive_fixed_validation"] = _assert_fixed_drive(env, summary["drive_config"])\n'
        '        physics["ground_wheel_mu_kf_runtime_validation"] = _validate_mu_kf(physics, MU, KF)\n'
        '        if physics["ground_wheel_mu_kf_runtime_validation"]["status"] != "pass":\n'
        '            raise RuntimeError(f"runtime ground+14-wheel mu/kf mismatch: {physics[\'ground_wheel_mu_kf_runtime_validation\']}")\n'
        '        if (physics["solver_config"].get("cone") != CONE\n'
        '                or not math.isclose(float(physics["solver_config"].get("impratio")), IMPRATIO, abs_tol=1e-9)\n'
        '                or physics["solver_config"].get("use_mujoco_contacts") is not False):\n'
        '            raise RuntimeError(f"runtime solver treatment mismatch: {physics[\'solver_config\']}")\n'
        '        summary["runtime_physics"] = physics\n\n'
        '        first_options = {"mode": "search", "start": START, "goal": GOAL}\n',
        "fixed_runtime_gates_and_mimic_metadata",
    )
    source = _replace_once(
        source,
        '        capacity = total_policy_steps * decimation\n'
        '        effort_tape = wp.empty((capacity, 14), dtype=wp.float32, device=env.device)\n',
        '        capacity = total_policy_steps * decimation\n'
        '        mimic_pair_layout, pair_qd_offsets = resolve_pair_layout(NewtonManager.get_model())\n'
        '        summary["same_side_treatment"]["resolved_native_pair_layout"] = mimic_pair_layout\n'
        '        pair_offsets_warp = wp.from_torch(torch.as_tensor(pair_qd_offsets, dtype=torch.int32,\n'
        '                                                               device=env.device).contiguous())\n'
        '        mimic_qd_tape = wp.empty((capacity, 12), dtype=wp.float32, device=env.device)\n'
        '        mimic_qd_counter = wp.zeros(1, dtype=wp.int32, device=env.device)\n'
        '        def capture_mimic_qd():\n'
        '            state = NewtonManager.get_state_0()\n'
        '            state_qd = getattr(state, "joint_qd", None)\n'
        '            if state_qd is None:\n'
        '                raise RuntimeError("Newton state_0 joint_qd unavailable in post-actuator callback")\n'
        '            wp.launch(capture_pair_qd_residuals, dim=1, device=env.device, inputs=[\n'
        '                state_qd, pair_offsets_warp, mimic_qd_tape, mimic_qd_counter, capacity])\n'
        '        NewtonManager.register_post_actuator_callback(capture_mimic_qd)\n'
        '        effort_tape = wp.empty((capacity, 14), dtype=wp.float32, device=env.device)\n',
        "physics_step_pair_qd_tape",
    )
    source = _replace_once(
        source,
        '        expected_slot = 0\n'
        '        native_effort_host = np.empty((capacity, 14), dtype=np.float64)\n'
        '        effort_tape_torch = wp.to_torch(effort_tape)\n'
        '        with trajectories_path.open("x", encoding="utf-8") as trajectory_stream, \\\n'
        '                efforts_path.open("x", encoding="utf-8") as effort_stream:\n',
        '        expected_slot = 0\n'
        '        native_effort_host = np.empty((capacity, 14), dtype=np.float64)\n'
        '        native_mimic_qd_host = np.empty((capacity, 12), dtype=np.float64)\n'
        '        effort_tape_torch = wp.to_torch(effort_tape)\n'
        '        mimic_qd_tape_torch = wp.to_torch(mimic_qd_tape)\n'
        '        with trajectories_path.open("x", encoding="utf-8") as trajectory_stream, \\\n'
        '                efforts_path.open("x", encoding="utf-8") as effort_stream, \\\n'
        '                mimic_qd_path.open("x", encoding="utf-8") as mimic_qd_stream:\n',
        "open_qd_raw_stream",
    )
    source = _replace_once(
        source,
        '                    before_reset = counter_value(counter, wp)\n'
        '                    env.reset(seed=SEED, options=first_options)\n'
        '                    after_reset = counter_value(counter, wp)\n'
        '                    if after_reset != before_reset:\n'
        '                        raise RuntimeError(f"physics callback ran during reset: {before_reset}->{after_reset}")\n',
        '                    before_reset = counter_value(counter, wp)\n'
        '                    before_qd_reset = counter_value(mimic_qd_counter, wp)\n'
        '                    env.reset(seed=SEED, options=first_options)\n'
        '                    after_reset = counter_value(counter, wp)\n'
        '                    after_qd_reset = counter_value(mimic_qd_counter, wp)\n'
        '                    if after_reset != before_reset or after_qd_reset != before_qd_reset:\n'
        '                        raise RuntimeError(f"physics callback ran during reset: effort {before_reset}->{after_reset}; qd {before_qd_reset}->{after_qd_reset}")\n',
        "reset_no_callback_gate",
    )
    source = _replace_once(
        source,
        '                        slot_before = counter_value(counter, wp)\n'
        '                        step_helper(env, np.asarray(action, dtype=np.float32))\n'
        '                        slot_after = counter_value(counter, wp)\n',
        '                        slot_before = counter_value(counter, wp)\n'
        '                        qd_slot_before = counter_value(mimic_qd_counter, wp)\n'
        '                        step_helper(env, np.asarray(action, dtype=np.float32))\n'
        '                        slot_after = counter_value(counter, wp)\n'
        '                        qd_slot_after = counter_value(mimic_qd_counter, wp)\n'
        '                        if qd_slot_after - qd_slot_before != decimation:\n'
        '                            raise RuntimeError(f"{case[\'name\']} {phase} step {phase_step}: mimic qd callback {qd_slot_before}->{qd_slot_after}, expected {decimation}")\n',
        "per_policy_qd_callback_gate",
    )
    source = _replace_once(
        source,
        '                        native_effort_host[slot_before:slot_after] = sampled_effort\n'
        '                        for substep, values in enumerate(sampled_effort, start=1):\n',
        '                        native_effort_host[slot_before:slot_after] = sampled_effort\n'
        '                        sampled_pair_qd = np.asarray(\n'
        '                            mimic_qd_tape_torch[qd_slot_before:qd_slot_after].detach().cpu().numpy(), dtype=np.float64)\n'
        '                        if sampled_pair_qd.shape != (decimation, 12) or not np.isfinite(sampled_pair_qd).all():\n'
        '                            raise RuntimeError(f"{case[\'name\']} {phase} step {phase_step}: invalid pair qd slice {sampled_pair_qd.shape}")\n'
        '                        native_mimic_qd_host[qd_slot_before:qd_slot_after] = sampled_pair_qd\n'
        '                        for substep, values in enumerate(sampled_pair_qd, start=1):\n'
        '                            qd_slot = qd_slot_before + substep - 1\n'
        '                            mimic_qd_stream.write(json.dumps({"case": case["name"], "phase": phase,\n'
        '                                "policy_step_in_phase": phase_step, "physics_substep_in_policy": substep,\n'
        '                                "physics_iteration_global": qd_slot + 1,\n'
        '                                "simulation_time_s_global": (qd_slot + 1) * args.physics_dt,\n'
        '                                "native_qd_pair_residual_rad_s_by_pair_order": values.tolist(),\n'
        '                                "finite": bool(np.isfinite(values).all())}, ensure_ascii=False, allow_nan=False) + "\\n")\n'
        '                        mimic_qd_stream.flush()\n'
        '                        for substep, values in enumerate(sampled_effort, start=1):\n',
        "qd_raw_rows_per_substep",
    )
    source = _replace_once(
        source,
        '                        row = state_row(env, case["name"], phase, phase_step, action,\n'
        '                                        local_elapsed, slot_before, slot_after)\n'
        '                        case_rows.append(row)\n',
        '                        row = state_row(env, case["name"], phase, phase_step, action,\n'
        '                                        local_elapsed, slot_before, slot_after)\n'
        '                        row["same_side_pair_policy_state_after_step"] = read_policy_pair_state(\n'
        '                            NewtonManager.get_model(), NewtonManager.get_state_0(), mimic_pair_layout)\n'
        '                        row["native_qd_pair_slot_range_zero_based"] = [qd_slot_before, qd_slot_after]\n'
        '                        case_rows.append(row)\n',
        "policy_step_pair_q_sample",
    )
    source = _replace_once(
        source,
        '        captured = counter_value(counter, wp)\n'
        '        if captured != capacity or expected_slot != capacity:\n',
        '        captured = counter_value(counter, wp)\n'
        '        captured_qd = counter_value(mimic_qd_counter, wp)\n'
        '        if captured_qd != capacity:\n'
        '            raise RuntimeError(f"captured pair qd {captured_qd} substeps; expected {capacity}")\n'
        '        if captured != capacity or expected_slot != capacity:\n',
        "final_qd_count_gate",
    )
    patched_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    namespace = dict(probe.__dict__)
    namespace.update({
        "OUT_ROOT": RUN_ROOT,
        "MIMIC_MODE": "baseline",
        "MIMIC_PROTOCOL": MIMIC_PROTOCOL,
        "PROBE_PATH": PROBE_PATH,
        "HOOK_PATH": HOOK_PATH,
        "RUNNER_PATH": RUNNER_PATH,
        "RUNNER_SOURCE_SHA256": _sha(RUNNER_PATH),
        "MIMIC_SOURCE_PATCH_SHA256": patched_sha,
        "PAIR_NAMES": PAIR_NAMES,
        "capture_pair_qd_residuals": capture_pair_qd_residuals,
        "resolve_pair_layout": resolve_pair_layout,
        "read_policy_pair_state": read_policy_pair_state,
        "_solver_counts": _solver_counts,
        "sha_file": _sha,
        "make_mimic_hook": lambda: None,
        "finalized_mimic_audit": hook_module.finalized_mimic_audit,
        "_assert_fixed_drive": _assert_fixed_drive,
        "_validate_mu_kf": _validate_mu_kf,
        "MU": MU,
        "KF": KF,
        "CONE": CONE,
        "IMPRATIO": IMPRATIO,
    })
    exec(compile(source, str(PROBE_PATH), "exec"), namespace)
    check = {
        "status": "static_pass",
        "patched_collector_sha256": patched_sha,
        "runner_sha256": _sha(RUNNER_PATH),
        "collector_sha256": _sha(PROBE_PATH),
        "hook_sha256": _sha(HOOK_PATH),
        "protocol_sha256": _sha(MIMIC_PROTOCOL),
        "physics_dt_s": PHYSICS_DT, "decimation": 50, "policy_dt_s": 0.05,
        "cone": CONE, "impratio": IMPRATIO, "shape_mu": MU, "shape_kf": KF,
        "pair_count": len(PAIR_NAMES), "modes": ["baseline", "mimic"],
        "gpu_used": False,
    }
    return namespace, hook_module, check


def _assert_fixed_drive(env, drive_config: dict) -> dict:
    expected = {
        "velocity_gain": KP,
        "integral_gain_nm_per_proxy_wheel_per_rad_s_s": KI,
        "integral_update_period_s": 0.05,
        "integral_state_count_per_env": 2,
        "wheel_effort_limit_nm": EFFORT_CAP,
        "actuator_effort_limit_nm": EFFORT_CAP,
        "joint_effort_limit_nm": EFFORT_CAP,
        "proxy_wheel_count_per_side": 7,
    }
    observed = {key: drive_config.get(key) for key in expected}
    for key, wanted in expected.items():
        value = observed[key]
        if not isinstance(value, (int, float)) or abs(float(value) - wanted) > 1e-9:
            raise RuntimeError(f"drive config {key}={value!r}; expected {wanted}")
    wheel_ids = env._wheel_ids.detach().cpu().numpy().reshape(-1)
    names = [env.rover.joint_names[int(index)] for index in wheel_ids]
    if len(names) != 14 or len(set(names)) != 14 or any(not name.endswith("_wheel_joint") for name in names):
        raise RuntimeError(f"expected 14 unique wheel actuator joints, got {names}")
    return {"status": "pass", "expected": expected, "observed": observed,
            "native_wheel_joint_count": len(names), "native_wheel_joint_names": names}


def _validate_mu_kf(physics: dict, expected_mu: float, expected_kf: float) -> dict:
    labels = physics.get("newton_runtime_shape_labels")
    arrays = physics.get("newton_runtime_shape_material_arrays_in_label_order")
    if not isinstance(labels, list) or not isinstance(arrays, dict):
        return {"status": "NA", "labels": labels, "arrays": arrays}
    ground = [i for i, label in enumerate(labels) if str(label).endswith("/Field/Ground")]
    wheels = [i for i, label in enumerate(labels)
              if "_wheel_link/" in str(label) and str(label).endswith("/wheel_collision_visual")]
    mu_values, kf_values = arrays.get("shape_material_mu"), arrays.get("shape_material_kf")
    if len(ground) != 1 or len(wheels) != 14 or not isinstance(mu_values, list) or not isinstance(kf_values, list):
        return {"status": "NA", "ground_count": len(ground), "wheel_count": len(wheels),
                "mu_array": mu_values, "kf_array": kf_values}
    selected = ground + wheels
    observed_mu = [float(mu_values[i]) for i in selected]
    observed_kf = [float(kf_values[i]) for i in selected]
    matches = (all(np.isfinite(observed_mu)) and all(np.isfinite(observed_kf))
               and all(abs(value - expected_mu) <= 1e-6 for value in observed_mu)
               and all(abs(value - expected_kf) <= 1e-6 for value in observed_kf))
    return {"status": "pass" if matches else "fail", "ground_shape": labels[ground[0]],
            "wheel_shapes": [labels[i] for i in wheels], "observed_mu": observed_mu,
            "observed_kf": observed_kf, "expected_mu": expected_mu, "expected_kf": expected_kf}


def make_mimic_hook(hook_module):
    names = {side: [f"{side}_wheel_{wheel}_wheel_joint" for wheel in range(7)]
             for side in ("left", "right")}
    return hook_module.SameSideMimicHook(names)


def _postprocess(mode: str, output_dir: Path, hook_module) -> dict:
    summary_path = output_dir / "staged_motion_response_summary.json"
    policy_path = output_dir / "policy_trajectories.jsonl"
    qd_path = output_dir / "native_same_side_qd_residuals.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    treatment = summary.get("same_side_treatment", {})
    for key, path in (("runner_sha256", RUNNER_PATH), ("collector_sha256", PROBE_PATH),
                      ("hook_sha256", HOOK_PATH), ("hook_protocol_sha256", MIMIC_PROTOCOL)):
        if treatment.get(key) != _sha(path):
            raise RuntimeError(f"diagnostic source changed during run: {path}")
    if summary.get("physics_dt_s") != PHYSICS_DT or summary.get("decimation") != 50:
        raise RuntimeError("physics timestep or decimation differs from the paired protocol")
    if summary.get("solver_treatment") != {"cone": CONE, "impratio": IMPRATIO}:
        raise RuntimeError("solver treatment differs from the paired protocol")
    if summary.get("runtime_physics", {}).get("ground_wheel_mu_kf_runtime_validation", {}).get("status") != "pass":
        raise RuntimeError("runtime Ground+14 wheel mu/KF gate missing or failed")
    if summary.get("runtime_drive_fixed_validation", {}).get("status") != "pass":
        raise RuntimeError("runtime P/I/14-wheel/2 Nm actuator gate missing or failed")
    audit = treatment.get("solver_equality_audit", {})
    if mode == "mimic" and not (
            audit.get("newton_model_mimic_count") == 12
            and audit.get("newton_model_equality_constraint_count") == 12
            and audit.get("solver_host_equality", {}).get("neq") == 12
            and audit.get("solver_mjwarp_equality", {}).get("neq") == 12
            and audit.get("all_expected_mjEQ_JOINT_pairs_present_once") is True):
        raise RuntimeError("candidate did not retain exactly 12 verified mimic/equality rows")
    policy_rows = [json.loads(line) for line in policy_path.read_text(encoding="utf-8").splitlines() if line]
    qd_rows = [json.loads(line) for line in qd_path.read_text(encoding="utf-8").splitlines() if line]
    expected_policy = 8 * (20 + 40 + 100)
    expected_qd = expected_policy * 50
    if len(policy_rows) != expected_policy or len(qd_rows) != expected_qd:
        raise RuntimeError(f"incomplete residual data: policy={len(policy_rows)}/{expected_policy}, qd={len(qd_rows)}/{expected_qd}")
    all_qd_abs, all_q_abs = [], []
    for row in qd_rows:
        values = row.get("native_qd_pair_residual_rad_s_by_pair_order")
        if not isinstance(values, list) or len(values) != 12 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise RuntimeError("native qd file has a missing/non-finite 12-pair sample")
        all_qd_abs.extend(abs(float(value)) for value in values)
    for row in policy_rows:
        state = row.get("same_side_pair_policy_state_after_step", {})
        relations = state.get("relations", [])
        if state.get("relation_count") != 12 or len(relations) != 12:
            raise RuntimeError("policy trajectory has a missing 12-pair sample")
        values = [float(relation["q_follower_minus_expected_rad"]) for relation in relations]
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise RuntimeError("policy q residual contains a non-finite value")
        all_q_abs.extend(abs(value) for value in values)
    thresholds = {
        "last_1s_native_physics_qd_abs_p95_max_rad_s": 0.01 * TARGET_WHEEL_SPEED,
        "last_1s_native_physics_qd_abs_max_rad_s": 0.05 * TARGET_WHEEL_SPEED,
        "last_1s_policy_q_abs_max_rad": 0.02,
    }
    per_case = {}
    for case in summary["cases"]:
        final_drive_policy = [row for row in policy_rows
                              if row.get("case") == case and row.get("phase") == "drive"
                              and 20 < int(row.get("phase_step", 0)) <= 40]
        final_drive_qd = [row for row in qd_rows
                          if row.get("case") == case and row.get("phase") == "drive"
                          and 20 < int(row.get("policy_step_in_phase", 0)) <= 40]
        q_abs = np.asarray([
            abs(float(relation["q_follower_minus_expected_rad"]))
            for row in final_drive_policy
            for relation in row["same_side_pair_policy_state_after_step"]["relations"]
        ], dtype=np.float64)
        qd_abs = np.asarray([
            abs(float(value))
            for row in final_drive_qd
            for value in row["native_qd_pair_residual_rad_s_by_pair_order"]
        ], dtype=np.float64)
        if len(final_drive_policy) != 20 or len(q_abs) != 240 or len(final_drive_qd) != 1000 or len(qd_abs) != 12000:
            raise RuntimeError(f"{case}: last-drive-second residual counts do not match 20 policy/1000 physics samples")
        if not np.isfinite(q_abs).all() or not np.isfinite(qd_abs).all():
            raise RuntimeError(f"{case}: non-finite pair residuals")
        observed = {
            "native_physics_qd_pair_abs_p95_rad_s": float(np.percentile(qd_abs, 95.0)),
            "native_physics_qd_pair_abs_max_rad_s": float(np.max(qd_abs)),
            "policy_q_pair_abs_max_rad": float(np.max(q_abs)),
            "native_qd_physics_rows": len(final_drive_qd),
            "native_qd_pair_samples": len(qd_abs),
            "policy_q_samples": len(final_drive_policy),
            "policy_q_pair_samples": len(q_abs),
        }
        cases_summary = {row["case"]: row for row in summary["cases_summary"]}
        effort_pass = bool(cases_summary[case].get("native_effort_cap_pass"))
        criteria = {
            "native_qd_abs_p95_within_v2_limit": observed["native_physics_qd_pair_abs_p95_rad_s"] <= thresholds["last_1s_native_physics_qd_abs_p95_max_rad_s"],
            "native_qd_abs_max_within_v2_limit": observed["native_physics_qd_pair_abs_max_rad_s"] <= thresholds["last_1s_native_physics_qd_abs_max_rad_s"],
            "policy_q_abs_max_within_v2_limit": observed["policy_q_pair_abs_max_rad"] <= thresholds["last_1s_policy_q_abs_max_rad"],
            "residual_samples_finite": True,
            "all_14_actuator_efforts_within_2Nm": effort_pass,
        }
        per_case[case] = {"observed": observed, "criteria": criteria, "passed": all(criteria.values())}
    result = {
        "mode": mode,
        "mimic_hook_enabled": mode == "mimic",
        "sampling": {
            "native_qd_source": "NewtonManager.get_state_0().joint_qd read in post-actuator callback before solver step",
            "native_qd_offsets_resolved_once_before_motion": True,
            "native_qd_callback_does_only_warp_tape_write": True,
            "policy_q_source": "NewtonManager.get_state_0().joint_q after each policy env.step; pre-resolved offsets",
            "last_1s_window": "drive policy steps 21-40 (1.0 to 2.0 s within drive phase)",
            "pair_order": PAIR_NAMES,
            "native_qd_rows": len(qd_rows), "policy_rows": len(policy_rows),
        },
        "thresholds_from_same_side_mimic_v2_protocol": thresholds,
        "all_run_samples": {
            "native_qd_pair_abs_p95_rad_s": float(np.percentile(np.asarray(all_qd_abs), 95.0)),
            "native_qd_pair_abs_max_rad_s": float(np.max(all_qd_abs)),
            "native_qd_pair_sample_count": len(all_qd_abs),
            "policy_q_pair_abs_max_rad": float(np.max(all_q_abs)),
            "policy_q_pair_sample_count": len(all_q_abs),
        },
        "per_case": per_case,
        "all_cases_passed_residual_and_effort_gates": all(row["passed"] for row in per_case.values()),
    }
    summary["same_side_pair_residual_acceptance"] = result
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "mimic"))
    args = parser.parse_args(argv)
    namespace, hook_module, static = prepare()
    if args.static_check:
        print(json.dumps(static, ensure_ascii=False, indent=2))
        return 0
    if args.mode is None:
        raise SystemExit("--mode baseline|mimic is required")
    if os.environ.get("RUN_STAGED_MIMIC_RESPONSE") != "1":
        raise SystemExit("Requires RUN_STAGED_MIMIC_RESPONSE=1 after parent authorization")
    mode = args.mode
    case = ("baseline_1ms_elliptic_1_mu0_8_kf1000" if mode == "baseline" else
            "mimic_v2_1ms_elliptic_1_mu0_8_kf1000")
    output_dir = RUN_ROOT / case
    log_path = RUN_ROOT / f"{case}.log"
    exit_path = RUN_ROOT / f"{case}.exit.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output: {output_dir}")
    if exit_path.exists():
        raise SystemExit(f"refusing prior exit record: {exit_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    names_by_side = {side: [f"{side}_wheel_{wheel}_wheel_joint" for wheel in range(7)]
                     for side in ("left", "right")}
    namespace["MIMIC_MODE"] = mode
    namespace["make_mimic_hook"] = lambda: hook_module.SameSideMimicHook(names_by_side)
    os.environ["RUN_STAGED_RESPONSE_PROBE"] = "1"
    arguments = ["--physics-dt", str(PHYSICS_DT), "--cone", CONE, "--impratio", str(IMPRATIO),
                 "--output-dir", str(output_dir)]
    result = int(namespace["main"](arguments))
    if result != 0:
        return result
    acceptance = _postprocess(mode, output_dir, hook_module)
    print(json.dumps({"mode": mode, "output_dir": str(output_dir),
                      "same_side_pair_residual_acceptance": acceptance}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
