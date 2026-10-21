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
MIMIC_ROOT = EXP_ROOT / "same_side_mimic_v1"
V4_PATH = EXP_ROOT / "contact_friction_v4.py"
HOOK_PATH = MIMIC_ROOT / "same_side_mimic_hook.py"
PROTOCOL_PATH = MIMIC_ROOT / "protocol.json"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = MIMIC_ROOT / "contact_probe_runs_v1"
MECHANISM_TREATMENT = (
    "same_side_wheel0_master_6_mimic_followers_each_side_coef0_0_coef1_1; "
    "all_original_14_per_wheel_PI_actuators_and_2Nm_caps_retained"
)


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
    source = _replace_once(
        source,
        '    summary_path = output_dir / "contact_friction_v4_summary.json"\n',
        '    summary_path = output_dir / "same_side_mimic_v1_summary.json"\n',
        "summary_path",
    )
    source = _replace_once(
        source,
        '        "schema": "isaac-field-contact-friction-v4",\n',
        '        "schema": "isaac-field-same-side-mimic-v1",\n'
        '        "mechanism_treatment": MECHANISM_TREATMENT,\n',
        "summary_schema_and_treatment",
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
    class FakeBuilder:
        def __init__(self) -> None:
            self.joint_label = [
                f"/World/envs/env_0/Robot/base_link/{side}_wheel_{index}_wheel_joint"
                for side in ("left", "right") for index in range(7)
            ]
            self.calls: list[dict[str, Any]] = []

        def add_constraint_mimic(self, **kwargs):
            self.calls.append(dict(kwargs))
            return len(self.calls) - 1

    names = {
        side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
        for side in ("left", "right")
    }
    builder = FakeBuilder()
    hook = hook_type(names)
    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    if len(builder.calls) != 12 or len(hook.relations) != 12:
        raise AssertionError("fake builder did not receive exactly 12 mimic relations")
    if any(call["coef0"] != 0.0 or call["coef1"] != 1.0 for call in builder.calls):
        raise AssertionError("fake builder received unexpected mimic coefficients")
    if any(call["joint0"] == call["joint1"] for call in builder.calls):
        raise AssertionError("a mimic follower resolved to its own leader index")
    return {"fake_builder_constraint_count": len(builder.calls), "status": "passed"}


def _fake_runtime_state_check(hook_type, runtime_state_reader) -> dict[str, Any]:
    from types import SimpleNamespace  # noqa: PLC0415

    names = {
        side: [f"{side}_wheel_{index}_wheel_joint" for index in range(7)]
        for side in ("left", "right")
    }
    labels = [
        f"/World/envs/env_0/Robot/base_link/{name}"
        for side in ("left", "right") for name in names[side]
    ]
    hook = hook_type(names)
    builder = SimpleNamespace(joint_label=labels)
    builder._calls = []

    def add_constraint_mimic(**kwargs):
        index = len(builder._calls)
        builder._calls.append(dict(kwargs))
        return index

    builder.add_constraint_mimic = add_constraint_mimic
    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
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
    return {
        "newton_version": importlib.metadata.version("newton"),
        "method": "newton.ModelBuilder.add_constraint_mimic",
        "signature_parameters": parameters,
        "required_parameters_present": True,
    }


def prepare(*, static_check: bool = False):
    if not V4_PATH.is_file() or not HOOK_PATH.is_file():
        raise FileNotFoundError("v4 collector or same-side mimic hook is missing")
    hook_module = _load_module(HOOK_PATH, "same_side_mimic_v1_hook")
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
    return patched_main, hook_module, check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-check", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    patched_main, hook_module, check = prepare(static_check=args.static_check)
    if args.static_check:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0
    if os.environ.get("RUN_CONTACT_MIMIC_V1") != "1":
        raise SystemExit("Requires RUN_CONTACT_MIMIC_V1=1 after parent authorization")
    os.environ["RUN_CONTACT_FRICTION_V4"] = "1"

    module = sys.modules["contact_friction_v4_frozen"]
    module.finalized_mimic_audit = hook_module.finalized_mimic_audit
    return int(patched_main(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
