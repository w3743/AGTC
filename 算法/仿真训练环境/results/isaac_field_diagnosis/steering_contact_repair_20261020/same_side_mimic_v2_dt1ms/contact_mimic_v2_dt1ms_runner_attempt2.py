#!/usr/bin/env python3
"""Reuse the frozen v4 collector with one result-only same-side mimic treatment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
MIMIC_ROOT = EXP_ROOT / "same_side_mimic_v2_dt1ms"
V4_PATH = EXP_ROOT / "contact_friction_v4.py"
HOOK_PATH = MIMIC_ROOT / "same_side_mimic_hook.py"
PROTOCOL_PATH = MIMIC_ROOT / "protocol.json"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = MIMIC_ROOT / "contact_probe_runs_dt1ms"
MECHANISM_TREATMENT = (
    "same_side_wheel0_master_6_mimic_followers_each_side_coef0_0_coef1_1; "
    "public_mujoco_equality_rows_solref_0.004_1_solimp_0.999_0.9999_0.001_0.5_2; "
    "all_original_14_per_wheel_PI_actuators_and_2Nm_caps_retained"
)
EQUALITY_SOLREF = (0.004, 1.0)
EQUALITY_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)
EXPECTED_INITIAL_STATE_SHA256 = "7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af"
EXPECTED_FIXED_CONTRACT_SHA256 = "33b4298fa1bd57f5c114d5a17d28ba0c171a697cc8ff42d95559233501142b05"
BASELINE_SUMMARY_PATH = EXP_ROOT / "same_side_mimic_v2/contact_probe_runs_v2/dt-2ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1/same_side_mimic_v2_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"v4 patch anchor {label!r} matched {count} times; expected exactly once")
    return source.replace(old, new, 1)


def _patched_v4_main_source(v4_module) -> str:
    """Apply narrow, checked edits in memory; the frozen v4 source stays untouched."""
    source = inspect.getsource(v4_module.main)
    source = _replace_once(source, "args.physics_dt != 0.002", "args.physics_dt != 0.001", "physics_dt_gate")
    source = _replace_once(
        source,
        "v4 is pinned to the authorized 2 ms / elliptic / impratio 1 baseline",
        "dt1ms runner is pinned to 1 ms / elliptic / impratio 1",
        "physics_dt_gate_message",
    )
    source = _replace_once(
        source,
        '        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})\n',
        '        if summary.get("fixed_physical_comparison_contract_sha256") != EXPECTED_FIXED_CONTRACT_SHA256:\n'
        '            raise RuntimeError("fixed physical comparison contract differs from frozen 2 ms baseline")\n'
        '        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})\n',
        "fixed_contract_pre_motion_guard",
    )
    source = _replace_once(
        source,
        '    summary_path = output_dir / "contact_friction_v4_summary.json"\n',
        '    summary_path = output_dir / "same_side_mimic_v2_dt1ms_summary.json"\n',
        "summary_path",
    )
    source = _replace_once(
        source,
        '        "schema": "isaac-field-contact-friction-v4",\n',
        '        "schema": "isaac-field-same-side-mimic-v2-dt1ms",\n'
        '        "mechanism_treatment": MECHANISM_TREATMENT,\n',
        "summary_schema_and_treatment",
    )
    source = _replace_once(
        source,
        '        summary["initial_state_canonical_sha256"] = json_sha256(initial_state)\n',
        '        summary["initial_state_canonical_sha256"] = json_sha256(initial_state)\n'
        '        if summary["initial_state_canonical_sha256"] != EXPECTED_INITIAL_STATE_SHA256:\n'
        '            raise RuntimeError("canonical reset state differs from the frozen v2 baseline")\n',
        "initial_state_hash_pre_motion_guard",
    )
    source = _replace_once(
        source,
        '        runner_path = EXP_ROOT / "run_contact_friction_v4.sh"\n',
        '        v4_runner_path = EXP_ROOT / "run_contact_friction_v4.sh"\n'
        '        runner_path = RUNNER_PATH\n',
        "runner_path",
    )
    source = _replace_once(
        source,
        '            "v4_runner_sha256": sha256_file(runner_path),\n',
        '            "v4_runner_sha256": sha256_file(v4_runner_path),\n'
        '            "same_side_mimic_runner_sha256": sha256_file(runner_path),\n'
        '            "same_side_mimic_hook_sha256": sha256_file(HOOK_PATH),\n'
        '            "same_side_mimic_protocol_sha256": sha256_file(PROTOCOL_PATH),\n'
        '            "patched_v4_main_sha256": PATCHED_MAIN_SHA256,\n',
        "provenance_hashes",
    )
    source = _replace_once(
        source,
        '        env = IsaacFieldEnv(cfg)\n',
        '        wheel_names_by_side = {\n'
        '            side: [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs if wheel.side == side]\n'
        '            for side in ("left", "right")\n'
        '        }\n'
        '        mimic_hook = make_same_side_mimic_hook(wheel_names_by_side)\n'
        '        with newton_builder_world_hook(mimic_hook):\n'
        '            env = IsaacFieldEnv(cfg)\n',
        "environment_builder_hook",
    )
    source = _replace_once(
        source,
        '        from isaaclab_newton.physics import NewtonManager\n',
        '        from isaaclab_newton.physics import NewtonManager\n'
        '        from isaaclab_newton.cloner import newton_builder_world_hook\n',
        "builder_hook_import",
    )
    source = _replace_once(
        source,
        '            "fixed_policy_dt_s": POLICY_DT,\n',
        '            "fixed_policy_dt_s": POLICY_DT,\n'
        '            "mechanism_treatment": MECHANISM_TREATMENT,\n',
        "comparison_contract_treatment",
    )
    source = _replace_once(
        source,
        '        drive_config = env.drive_config\n',
        '        drive_config = dict(env.drive_config)\n'
        '        drive_config["mechanism_treatment"] = MECHANISM_TREATMENT\n',
        "drive_config_treatment",
    )
    source = _replace_once(
        source,
        '            "runtime_physics_properties": runtime_properties(cfg, env, wheel_body_names),\n',
        '            "runtime_physics_properties": runtime_properties(cfg, env, wheel_body_names),\n'
        '            "mechanism_coupling_audit": finalized_mimic_audit(\n'
        '                mimic_hook, NewtonManager.get_model(), NewtonManager._solver\n'
        '            ),\n',
        "final_solver_equality_audit",
    )
    source = _replace_once(
        source,
        '        final_policy_rows = policy_rows[-round(1.0 / POLICY_DT):]\n',
        '        mimic_samples = [\n'
        '            row.get("same_side_mimic_runtime_state")\n'
        '            for row in policy_rows\n'
        '        ]\n'
        '        if any(not isinstance(sample, dict) or sample.get("relation_count") != 12\n'
        '               for sample in mimic_samples):\n'
        '            raise RuntimeError("native same-side mimic q/qd samples are incomplete")\n'
        '        mimic_relations = [relation for sample in mimic_samples for relation in sample["relations"]]\n'
        '        mimic_q_errors = np.asarray([row["q_follower_minus_expected_rad"] for row in mimic_relations])\n'
        '        mimic_qd_errors = np.asarray([row["qd_follower_minus_expected_rad_s"] for row in mimic_relations])\n'
        '        summary["same_side_mimic_runtime_residual_summary"] = {\n'
        '            "policy_samples": len(mimic_samples),\n'
        '            "relation_samples": len(mimic_relations),\n'
        '            "max_abs_q_residual_rad": float(np.max(np.abs(mimic_q_errors))),\n'
        '            "max_abs_qd_residual_rad_s": float(np.max(np.abs(mimic_qd_errors))),\n'
        '            "sample_semantics": "Newton native post-step q/qd; see policy JSONL for every wheel pair",\n'
        '        }\n'
        '        final_policy_rows = policy_rows[-round(1.0 / POLICY_DT):]\n',
        "runtime_residual_summary",
    )
    source = _replace_once(
        source,
        '        if sha256_file(helper_path) != summary["v4_helper_sha256"] or sha256_file(runner_path) != summary["v4_runner_sha256"]:\n'
        '            raise RuntimeError("v4 helper or runner changed during probe")\n',
        '        if (sha256_file(helper_path) != summary["v4_helper_sha256"]\n'
        '                or sha256_file(v4_runner_path) != summary["v4_runner_sha256"]\n'
        '                or sha256_file(runner_path) != summary["same_side_mimic_runner_sha256"]\n'
        '                or sha256_file(HOOK_PATH) != summary["same_side_mimic_hook_sha256"]\n'
        '                or sha256_file(PROTOCOL_PATH) != summary["same_side_mimic_protocol_sha256"]):\n'
        '            raise RuntimeError("v4 or same-side mimic diagnostic source changed during probe")\n',
        "source_integrity_check",
    )
    return source


def _install_runtime_sample_patch(v4_module, hook_holder: dict[str, Any], runtime_state_reader) -> None:
    base_sample = v4_module.V1.sample

    def sample_with_native_mimic_state(*args, **kwargs):
        row = base_sample(*args, **kwargs)
        hook = hook_holder.get("hook")
        if hook is None:
            raise RuntimeError("mimic hook did not initialize before the v4 sample loop")
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        row["same_side_mimic_runtime_state"] = runtime_state_reader(
            hook,
            NewtonManager.get_model(),
            NewtonManager.get_state_0(),
        )
        return row

    v4_module.V1.sample = sample_with_native_mimic_state


def _fake_builder_hook_check(hook_type) -> dict[str, Any]:
    import newton  # noqa: PLC0415

    class FakeBuilder:
        def __init__(self) -> None:
            self.joint_label = [
                f"/World/envs/env_0/Robot/base_link/{side}_wheel_{index}_wheel_joint"
                for side in ("left", "right") for index in range(7)
            ]
            self.calls: list[dict[str, Any]] = []
            self.equality_calls: list[dict[str, Any]] = []
            self.current_world = 0
            self.custom_attributes = newton.ModelBuilder().custom_attributes

        def add_constraint_mimic(self, **kwargs):
            self.calls.append(dict(kwargs))
            return len(self.calls) - 1

        def add_custom_values(self, **kwargs):
            self.equality_calls.append(dict(kwargs))
            index = len(self.equality_calls) - 1
            return {key: index for key in kwargs}

    names = {
        side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
        for side in ("left", "right")
    }
    builder = FakeBuilder()
    hook = hook_type(names)
    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    if len(builder.calls) != 12 or len(builder.equality_calls) != 12 or len(hook.relations) != 12:
        raise AssertionError("fake builder did not receive exactly 12 mimic relations and equality rows")
    if any(call["coef0"] != 0.0 or call["coef1"] != 1.0 for call in builder.calls):
        raise AssertionError("fake builder received unexpected mimic coefficients")
    if any(call["joint0"] == call["joint1"] for call in builder.calls):
        raise AssertionError("a mimic follower resolved to its own leader index")
    if any("mujoco:eq_solref" not in call or "mujoco:eq_solimp" not in call
           for call in builder.equality_calls):
        raise AssertionError("fake builder equality rows did not receive solref and solimp")
    return {
        "fake_builder_constraint_count": len(builder.calls),
        "fake_builder_equality_row_count": len(builder.equality_calls),
        "status": "passed",
    }


def _fake_runtime_state_check(hook_type, runtime_state_reader) -> dict[str, Any]:
    from types import SimpleNamespace  # noqa: PLC0415
    import newton  # noqa: PLC0415

    names = {
        side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
        for side in ("left", "right")
    }
    labels = [
        f"/World/envs/env_0/Robot/base_link/{name}"
        for side in ("left", "right") for name in names[side]
    ]
    hook = hook_type(names)
    builder = SimpleNamespace(
        joint_label=labels,
        current_world=0,
        custom_attributes=newton.ModelBuilder().custom_attributes,
    )
    builder._calls = []
    builder._eq_calls = []

    def add_constraint_mimic(**kwargs):
        index = len(builder._calls)
        builder._calls.append(dict(kwargs))
        return index

    builder.add_constraint_mimic = add_constraint_mimic
    def add_custom_values(**kwargs):
        index = len(builder._eq_calls)
        builder._eq_calls.append(dict(kwargs))
        return {key: index for key in kwargs}
    builder.add_custom_values = add_custom_values
    # The runtime sampler only needs relations; exercising the public hook also
    # ensures custom equality rows remain paired with every mimic.
    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    if len(builder._eq_calls) != 12:
        raise AssertionError("runtime fake hook did not add 12 equality rows")
    values_q = [0.1] * 7 + [-0.2] * 7
    values_qd = [0.3] * 7 + [-0.4] * 7
    model = SimpleNamespace(
        joint_label=labels,
        joint_q_start=list(range(15)),
        joint_qd_start=list(range(15)),
    )
    state = SimpleNamespace(joint_q=values_q, joint_qd=values_qd)
    result = runtime_state_reader(hook, model, state)
    if result["relation_count"] != 12:
        raise AssertionError("runtime sampler did not return all 12 follower/master pairs")
    if any(row["q_follower_minus_expected_rad"] != 0.0
           or row["qd_follower_minus_expected_rad_s"] != 0.0
           for row in result["relations"]):
        raise AssertionError("same-side fake runtime q/qd residuals are not zero")
    return {
        "runtime_relation_count": result["relation_count"],
        "extra_coordinate_start_sentinel_supported": True,
        "status": "passed",
    }


def _newton_public_api_check() -> dict[str, Any]:
    import newton  # noqa: PLC0415

    method = getattr(newton.ModelBuilder, "add_constraint_mimic", None)
    if method is None:
        raise RuntimeError("installed Newton ModelBuilder lacks add_constraint_mimic")
    parameters = list(inspect.signature(method).parameters)
    required = {"joint0", "joint1", "coef0", "coef1", "enabled", "label"}
    if not required.issubset(parameters):
        raise RuntimeError(f"installed mimic API signature is missing parameters: {parameters}")
    add_values = getattr(newton.ModelBuilder, "add_custom_values", None)
    if add_values is None or not any(
        name == "kwargs" for name in inspect.signature(add_values).parameters
    ):
        raise RuntimeError("installed Newton ModelBuilder lacks the public add_custom_values API")
    target_kind_public = hasattr(newton.solvers.SolverMuJoCo, "MjcEqualityTargetKind")
    return {
        "newton_version": importlib.metadata.version("newton"),
        "method": "newton.ModelBuilder.add_constraint_mimic",
        "signature_parameters": parameters,
        "required_parameters_present": True,
        "equality_api": "newton.ModelBuilder.add_custom_values",
        "equality_target_kind_public_enum_available": target_kind_public,
        "mimic_target_kind_schema_constant": 2,
    }


def _post_run_coupling_acceptance(output_dir: Path) -> dict[str, Any]:
    """Measure last-second qd at every native physics callback and q after policy steps."""
    import numpy as np  # noqa: PLC0415

    summary_path = output_dir / "same_side_mimic_v2_dt1ms_summary.json"
    policy_path = output_dir / "policy_step_contact_samples.jsonl"
    physics_path = output_dir / "physics_step_native_qd_effort.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    physics_dt = float(summary["physics_dt_s"])
    decimation = int(summary["decimation"])
    if physics_dt != 0.001 or decimation != 50:
        raise RuntimeError(f"expected 1 ms / decimation 50, got dt={physics_dt}, decimation={decimation}")
    if summary.get("fixed_physical_comparison_contract_sha256") != EXPECTED_FIXED_CONTRACT_SHA256:
        raise RuntimeError("1 ms fixed contract hash differs from authorized 2 ms baseline")
    if summary.get("initial_state_canonical_sha256") != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError("1 ms reset state differs from frozen baseline")
    if baseline.get("status") != "complete" or baseline.get("physics_dt_s") != 0.002 or baseline.get("decimation") != 25:
        raise RuntimeError("frozen 2 ms baseline summary is absent or has unexpected status/dt/decimation")
    if baseline.get("initial_state_canonical_sha256") != EXPECTED_INITIAL_STATE_SHA256 or baseline.get("fixed_physical_comparison_contract_sha256") != EXPECTED_FIXED_CONTRACT_SHA256:
        raise RuntimeError("frozen 2 ms baseline hash guard failed")
    wheel_order = summary.get("wheel_order_joints")
    if not isinstance(wheel_order, list) or len(wheel_order) != 14:
        raise RuntimeError(f"expected exactly 14 recorded native wheel-order joints, got {wheel_order}")
    wheel_index = {str(name).rsplit("/", 1)[-1]: index for index, name in enumerate(wheel_order)}
    expected_pairs = [
        (f"{side}_wheel_{follower}_wheel_joint", f"{side}_wheel_0_wheel_joint")
        for side in ("left", "right") for follower in range(1, 7)
    ]
    if any(follower not in wheel_index or leader not in wheel_index
           for follower, leader in expected_pairs):
        raise RuntimeError(f"wheel-order list cannot resolve all 12 same-side pairs: {wheel_order}")

    physics_rows = [
        json.loads(line) for line in physics_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    expected_physics_rows = int(summary["native_physics_substeps_expected"])
    if len(physics_rows) != expected_physics_rows or int(summary["native_physics_substeps_captured"]) != expected_physics_rows:
        raise RuntimeError(f"native physics qd tape is incomplete: rows={len(physics_rows)}, expected={expected_physics_rows}")
    last_second_start = float(summary["settle_s"]) + float(summary["drive_s"]) - 1.0
    last_second_physics = [
        row for row in physics_rows
        if row.get("phase") == "drive" and float(row.get("simulation_time_s", -1.0)) > last_second_start
    ]
    qd_errors: list[float] = []
    for row in last_second_physics:
        values = row.get("native_joint_qd_rad_s_by_wheel_order")
        if not isinstance(values, list) or len(values) != 14:
            raise RuntimeError("native qd tape row is missing the 14-wheel state vector")
        for follower, leader in expected_pairs:
            qd_errors.append(float(values[wheel_index[follower]]) - float(values[wheel_index[leader]]))
    qd_abs = np.abs(np.asarray(qd_errors, dtype=np.float64))
    expected_last_second_physics_rows = round(1.0 / physics_dt)
    expected_physics_pair_samples = expected_last_second_physics_rows * 12
    if (len(last_second_physics) != expected_last_second_physics_rows
            or len(qd_abs) != expected_physics_pair_samples):
        raise RuntimeError(
            f"last-second native qd slice incomplete: physics_rows={len(last_second_physics)}, "
            f"pair_samples={len(qd_abs)}, expected={expected_last_second_physics_rows} "
            f"and {expected_physics_pair_samples} for dt={physics_dt}"
        )
    if not np.isfinite(qd_abs).all():
        raise RuntimeError("native qd pair residual slice contains non-finite values")

    policy_rows = [
        json.loads(line) for line in policy_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    drive_policy_rows = [row for row in policy_rows if row.get("phase") == "drive"]
    final_policy_rows = drive_policy_rows[-20:]
    if len(final_policy_rows) != 20:
        raise RuntimeError(f"last-second policy q sample count {len(final_policy_rows)}; expected 20")
    q_errors = [
        float(relation["q_follower_minus_expected_rad"])
        for row in final_policy_rows
        for relation in row.get("same_side_mimic_runtime_state", {}).get("relations", [])
    ]
    q_abs = np.abs(np.asarray(q_errors, dtype=np.float64))
    if len(q_abs) != 240 or not np.isfinite(q_abs).all():
        raise RuntimeError(f"last-second policy q residual slice invalid: n={len(q_abs)}")

    qd_p95 = float(np.percentile(qd_abs, 95.0))
    qd_max = float(np.max(qd_abs))
    q_max = float(np.max(q_abs))
    qd_p95_limit = 0.01 * 9.25925925925926
    qd_max_limit = 0.05 * 9.25925925925926
    q_max_limit = 0.02
    effort_pass = bool(summary.get("native_effort_cap_pass", False))
    yaw_1ms = float(summary["end_last_1s_mean_root_yaw_rate_rad_s"])
    yaw_2ms = float(baseline["end_last_1s_mean_root_yaw_rate_rad_s"])
    yaw_diff = abs(yaw_1ms - yaw_2ms)
    yaw_tolerance = max(0.01, 0.10 * abs(yaw_1ms))
    criteria = {
        "last_1s_native_physics_qd_p95_le_1pct_target": qd_p95 <= qd_p95_limit,
        "last_1s_native_physics_qd_max_le_5pct_target": qd_max <= qd_max_limit,
        "last_1s_policy_q_max_le_0.02_rad": q_max <= q_max_limit,
        "all_native_qd_and_policy_q_residuals_finite": True,
        "actuator_effort_within_2Nm_cap": effort_pass,
        "last_1s_yaw_converged_to_2ms_baseline": yaw_diff <= yaw_tolerance,
    }
    result = {
        "sampling": {
            "native_qd_source": "v4 native qd tape: Newton state_0 in post-actuator callback before every solver substep",
            "native_qd_physics_rows_last_1s": len(last_second_physics),
            "native_qd_pair_samples_last_1s": len(qd_abs),
            "policy_q_samples_last_1s": len(final_policy_rows),
            "policy_q_pair_samples_last_1s": len(q_abs),
            "last_1s_time_window_s_open_lower_bound": last_second_start,
            "physics_dt_s": physics_dt,
            "decimation": decimation,
            "expected_native_qd_physics_rows": expected_last_second_physics_rows,
            "expected_native_qd_pair_samples": expected_physics_pair_samples,
        },
        "thresholds": {
            "target_wheel_speed_rad_s": 9.25925925925926,
            "native_qd_p95_rad_s": qd_p95_limit,
            "native_qd_max_rad_s": qd_max_limit,
            "policy_q_max_rad": q_max_limit,
            "yaw_1ms_vs_2ms_max_abs_difference_rad_s": yaw_tolerance,
        },
        "observed": {
            "native_qd_abs_p95_rad_s": qd_p95,
            "native_qd_abs_max_rad_s": qd_max,
            "policy_q_abs_max_rad": q_max,
            "last_1s_yaw_1ms_rad_s": yaw_1ms,
            "last_1s_yaw_2ms_rad_s": yaw_2ms,
            "last_1s_yaw_abs_difference_rad_s": yaw_diff,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    summary["same_side_mimic_acceptance"] = result
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _actual_builder_solver_cpu_check(hook_type, audit_function) -> dict[str, Any]:
    """Exercise the exact 12-row public schema on a CPU-only Newton model."""
    import numpy as np  # noqa: PLC0415
    import newton  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    names_by_side = {
        side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
        for side in ("left", "right")
    }
    builder = newton.ModelBuilder()
    for side in ("left", "right"):
        for name in names_by_side[side]:
            body = builder.add_link(
                mass=1.0,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                label=f"body_{name}",
            )
            joint = builder.add_joint_revolute(
                -1,
                body,
                axis=(0.0, 1.0, 0.0),
                label=name,
                damping=0.0,
                friction=0.0,
                armature=0.0,
            )
            builder.add_articulation([joint])
    hook = hook_type(names_by_side, solref=EQUALITY_SOLREF, solimp=EQUALITY_SOLIMP)
    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    audit = audit_function(hook, model, solver)
    if audit["newton_model_mimic_count"] != 12 or audit["newton_model_equality_constraint_count"] != 12:
        raise AssertionError("CPU Newton model did not preserve twelve mimic/equality rows")
    if audit["solver_host_equality"]["neq"] != 12 or audit["solver_mjwarp_equality"]["neq"] != 12:
        raise AssertionError("CPU MJWarp solver equality count is not exactly twelve")
    return {
        "status": "passed",
        "device": str(model.device),
        "builder_mimic_count": audit["newton_model_mimic_count"],
        "builder_equality_row_count": audit["newton_model_equality_constraint_count"],
        "host_neq": audit["solver_host_equality"]["neq"],
        "mjwarp_neq": audit["solver_mjwarp_equality"]["neq"],
        "newton_eq_solref": audit["newton_model_equality_solref"],
        "newton_eq_solimp": audit["newton_model_equality_solimp"],
    }


def prepare(*, static_check: bool = False):
    if not V4_PATH.is_file() or not HOOK_PATH.is_file():
        raise FileNotFoundError("v4 collector or same-side mimic hook is missing")
    hook_module = _load_module(HOOK_PATH, "same_side_mimic_v2_hook")
    v4_module = _load_module(V4_PATH, "contact_friction_v4_frozen")
    source = _patched_v4_main_source(v4_module)
    patched_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    hook_holder: dict[str, Any] = {}

    def make_hook(names_by_side):
        hook = hook_module.SameSideMimicHook(names_by_side)
        hook_holder["hook"] = hook
        return hook

    namespace = dict(v4_module.__dict__)
    namespace.update({
        "OUT_ROOT": OUTPUT_ROOT,
        "MECHANISM_TREATMENT": MECHANISM_TREATMENT,
        "EXPECTED_INITIAL_STATE_SHA256": EXPECTED_INITIAL_STATE_SHA256,
        "EXPECTED_FIXED_CONTRACT_SHA256": EXPECTED_FIXED_CONTRACT_SHA256,
        "RUNNER_PATH": RUNNER_PATH,
        "HOOK_PATH": HOOK_PATH,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "PATCHED_MAIN_SHA256": patched_sha,
        "make_same_side_mimic_hook": make_hook,
        "finalized_mimic_audit": hook_module.finalized_mimic_audit,
    })
    exec(compile(source, str(V4_PATH), "exec"), namespace)
    patched_main = namespace["main"]
    v4_module.OUT_ROOT = OUTPUT_ROOT
    _install_runtime_sample_patch(
        v4_module, hook_holder, hook_module.runtime_mimic_state_differences
    )
    check = {
        "v4_helper_sha256": sha256_file(V4_PATH),
        "mimic_hook_sha256": sha256_file(HOOK_PATH),
        "mimic_runner_sha256": sha256_file(RUNNER_PATH),
        "patched_v4_main_sha256": patched_sha,
        "output_root": str(OUTPUT_ROOT),
        "mechanism_treatment": MECHANISM_TREATMENT,
    }
    if static_check:
        check["hook_cpu_check"] = _fake_builder_hook_check(hook_module.SameSideMimicHook)
        check["runtime_q_qd_cpu_check"] = _fake_runtime_state_check(
            hook_module.SameSideMimicHook, hook_module.runtime_mimic_state_differences
        )
        check["newton_public_api_check"] = _newton_public_api_check()
        check["actual_builder_solver_cpu_check"] = _actual_builder_solver_cpu_check(
            hook_module.SameSideMimicHook, hook_module.finalized_mimic_audit
        )
        check["actual_builder_solver_cpu_check"] = _actual_builder_solver_cpu_check(
            hook_module.SameSideMimicHook, hook_module.finalized_mimic_audit
        )
    return patched_main, hook_module, check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-check", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    patched_main, hook_module, check = prepare(static_check=args.static_check)
    if args.static_check:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0
    if os.environ.get("RUN_CONTACT_MIMIC_V2_DT1MS") != "1":
        raise SystemExit("Requires RUN_CONTACT_MIMIC_V2_DT1MS=1 after parent authorization")
    os.environ["RUN_CONTACT_FRICTION_V4"] = "1"

    module = sys.modules["contact_friction_v4_frozen"]
    module.finalized_mimic_audit = hook_module.finalized_mimic_audit
    result = int(patched_main(remaining))
    if result != 0:
        return result
    try:
        output_index = remaining.index("--output-dir")
        output_dir = Path(remaining[output_index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--output-dir is required to collect equality acceptance evidence") from exc
    acceptance = _post_run_coupling_acceptance(output_dir)
    print(json.dumps({"same_side_mimic_acceptance": acceptance}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
