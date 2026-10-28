"""CPU-only checks for the lightweight JSON-to-WebGL state consumer."""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "view_training_state_web.py"
sys.path.insert(0, str(SCRIPT.parent))

import view_training_state_web as viewer


BODY_NAMES = (
    "base_link",
    *(f"left_wheel_{index}_wheel_link" for index in range(7)),
    *(f"right_wheel_{index}_wheel_link" for index in range(7)),
    "left_front_guide_link",
    "right_front_guide_link",
    "gimbal_yaw_link",
    "gimbal_pitch_link",
)
NATIVE_BODY_NAMES = viewer.NATIVE_BODY_LABELS
NATIVE_TASK_SHA = "a" * 64
NATIVE_REVISION = "newton_mujoco_warp_v6_test_fixture"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_physics(worlds: int, device: str) -> dict:
    return {
        "engine": "newton_mujoco_warp",
        "backend_revision": NATIVE_REVISION,
        "device": device,
        "newton": "1.7.0",
        "warp": "1.18.0",
        "mujoco": "3.15.0",
        "mujoco_warp": "3.15.0",
        "nworld": worlds,
        "dt_s": 0.001,
        "decimation": 50,
        "policy_dt_s": 0.05,
        "dof": 24 * worlds,
        "joints": 19 * worlds,
        "bodies": 19 * worlds,
        "mimic_relations_per_world": 14,
        "solver": {
            "class": "SolverMuJoCo", "backend": "mujoco_warp",
            "use_mujoco_cpu": False, "use_mujoco_contacts": True,
            "separate_worlds": True, "update_data_interval": 1,
            "deterministic_mode": "NOT_GUARANTEED",
            "iterations_actual": 40, "ls_iterations_actual": 8,
            "solver_actual": 2, "integrator_actual": 3, "cone_actual": 1,
            "impratio_invsqrt_actual": [0.31622776] * worlds,
            "impratio_actual": [10.0000001],
            "requested_config": {"njmax": 256, "nconmax": 64, "iterations": 40,
                                 "ls_iterations": 8, "solver": "newton",
                                 "integrator": "implicitfast", "cone": "elliptic", "impratio": 10.0},
            "requested_capacity_per_world": {"nconmax": 64, "njmax": 256},
            "actual_capacity_per_world": {"nconmax": 64, "njmax": 256},
            "exported_actuators": 16, "exported_equalities": 14,
        },
        "contact": {"source": "mujoco_warp_internal", "exported_geom_gap_values_per_world": [0.0],
                    "exported_geom_gap_array_shape": [worlds, 20]},
        "material_values_per_world": {"mu": [0.8], "ke": [2500.0], "kd": [100.0]},
        "chassis_inertia_diag_kg_m2": [0.2, 1.4, 1.6],
        "joint_dof_effort_limit_values_world0_nm": {"wheel_dof": [2.0], "gimbal_dof": [1.0]},
        "runtime_diagnostics": {"policy_step_sampled": False},
        "naconmax_total": 64 * worlds,
        **({"gravity_mps2": [0.0, 0.0, -9.81]} if device != "cpu" else {}),
    }


def make_native_trial(tmp_path: Path) -> tuple[Path, Path]:
    trial = tmp_path / "native_view_test"
    trial.mkdir(parents=True)
    script_root = SCRIPT.parents[3]
    backend_path = script_root / "算法" / "仿真训练环境" / "native" / "proto" / "newton_mujoco_backend.py"
    train_path = script_root / "算法" / "仿真训练环境" / "native" / "train" / "train_wsl.py"
    backend_sha, train_sha = _file_sha256(backend_path), _file_sha256(train_path)
    gpu_physics = _native_physics(64, "cuda:0")
    cpu_physics = _native_physics(1, "cpu")
    identity = {
        "physics": "newton_mujoco_warp", "device": "cuda", "num_envs": 64,
        "task_identity_sha256": NATIVE_TASK_SHA,
        "env": {"task_identity_sha256": NATIVE_TASK_SHA, "policy_dt_s": 0.05,
                "physics": gpu_physics},
    }
    contract = {
        "task_identity_sha256": NATIVE_TASK_SHA,
        "physics": {"engine": "newton_mujoco_warp", "backend_revision": NATIVE_REVISION,
                    "dt_s": 0.001, "decimation": 50},
    }
    (trial / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (trial / "observation_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    protocol_sha = "b" * 64
    (trial / "PROTOCOL_NATIVE_V2.sha256").write_text(
        json.dumps({"protocol_sha256": protocol_sha}), encoding="utf-8")
    (trial / "cli_args.txt").write_text(
        "python train_wsl.py --out runs/native_view_test --live-state-file /tmp/state.json --live-state-interval 5",
        encoding="utf-8",
    )

    bodies = [{"mujoco_body_id": 0, "name": "world", "parent_body_id": 0,
               "parent_body_name": "world", "viewer_name": None}]
    mapping = []
    geometries = [{
        "geom_id": 0, "name": "ground", "type_id": 0, "type": "plane",
        "body_id": 0, "body_name": "world", "newton_body_id": None, "viewer_name": None,
        "local_pos_xyz_m": [0, 0, 0], "local_quat_wxyz": [1, 0, 0, 0],
        "size_m": [5, 5, 5], "rgba": [0.2, 0.3, 0.4, 1.0],
    }]
    for index, viewer_name in enumerate(NATIVE_BODY_NAMES, start=1):
        body_name = f"model_body_{index}"
        bodies.append({"mujoco_body_id": index, "name": body_name,
                       "parent_body_id": 0, "parent_body_name": "world",
                       "viewer_name": viewer_name})
        mapping.append({"mujoco_body_id": index, "mujoco_body_name": body_name,
                        "newton_body_id": index - 1, "newton_native_label": f"body_{index}",
                        "viewer_name": viewer_name, "normalized_link_key": viewer_name})
        shape = "box" if index == 1 else "cylinder"
        size = [1.0, 2.0, 3.0] if shape == "box" else [0.5, 2.0, 0.0]
        quaternion = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)] if index == 1 else [1, 0, 0, 0]
        geometries.append({
            "geom_id": index, "name": f"geom_{index}", "type_id": 6 if shape == "box" else 5,
            "type": shape, "body_id": index, "body_name": body_name,
            "newton_body_id": index - 1, "viewer_name": viewer_name,
            "local_pos_xyz_m": [0.1, 0.2, 0.3], "local_quat_wxyz": quaternion,
            "size_m": size, "rgba": [0.5, 0.6, 0.7, 1.0],
        })

    source_snapshot = {
        "git_commit": "0123456789abcdef", "relevant_sources_dirty": False,
        "native_source_count": 69, "native_snapshot_files_verified": 69,
        "missing": [], "mismatches": [],
        "backend_relative_path": "算法/仿真训练环境/native/proto/newton_mujoco_backend.py",
        "backend_sha256": backend_sha,
        "train_wsl_relative_path": "算法/仿真训练环境/native/train/train_wsl.py",
        "train_wsl_sha256": train_sha,
    }
    payload = {
        "schema": "agtc.native_view_geometry.v1",
        "provenance": {
            "formal_run_dir": f"/mnt/c/test/native_view_test",
            "formal_gpu_run_task_identity_sha256_read_from_run_contract": NATIVE_TASK_SHA,
            "cpu_static_build_task_identity_sha256": None,
            "formal_run_device": "cuda", "formal_run_physics": gpu_physics,
            "formal_protocol_sha256": protocol_sha,
            "source_snapshot": source_snapshot,
            "constructed_backend_source_sha256": backend_sha,
            "construction": {"device": "cpu", "step_calls": 0, "policy_steps": 0},
        },
        "backend_identity_cpu": cpu_physics,
        "quaternion_conventions": {"mujoco_geom_quat_order": "wxyz",
                                   "newton_live_state_body_q_order": "xyz + xyzw"},
        "mujoco_static_model": {
            "nbody": 20, "ngeom": 20, "bodies": bodies,
            "body_mapping_to_newton": mapping,
            "body_mapping_verification": {"mujoco_nonworld_bodies": 19,
                                           "newton_bodies": 19, "one_to_one": True},
            "geometries": geometries,
            "ground_plane_geometries": [geometries[0]],
        },
        "newton_static_model_world0": {"body_count": 19, "shape_count": 20,
                                       "viewer_body_names_live_order": list(NATIVE_BODY_NAMES)},
    }
    model_path = tmp_path / "native_view_geometry.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    return model_path, trial


def make_native_packet(*, sequence: int = 1, run_id: str = "native_view_test") -> dict:
    origins = [[11.0, -11.0, 0.0], [11.0, 11.0, 0.0], [-11.0, -11.0, 0.0], [-11.0, 11.0, 0.0]]
    body_rows, roots = [], []
    for origin in origins:
        poses = [[origin[0] + index * 0.1, origin[1] + index * 0.2, 0.15,
                  0.0, 0.0, 0.0, 1.0] for index in range(len(NATIVE_BODY_NAMES))]
        body_rows.append(poses)
        roots.append(poses[0])
    goals_local = [[0.6, 0.9], [2.5, 3.9], [9.9, 5.4], [10.1, 6.3]]
    goals_world = [[goal[0] + origin[0], goal[1] + origin[1]]
                   for goal, origin in zip(goals_local, origins, strict=True)]
    return {
        "schema_version": 1, "sequence": sequence,
        "timestamp_unix_ns": 1_800_000_000_000_000_000,
        "timestamp_monotonic_ns": 1_000_000_000, "source_pid": 842,
        "source_start_time_unix_ns": 1_700_000_000_000_000_000,
        "run_id": run_id, "source": "isaac_field._collect_sim_cpu_samples",
        "sim_time_s": float(sequence), "sample_phase": "state_read_before_task_bookkeeping",
        "publish_status": "ok", "failed_publish_count": 0, "last_write_error": "",
        "quaternion_order": "xyzw",
        "robot_usd_path": "this/path/is/ignored/in/native/mode.usda",
        "terrain_usd_path": "this/path/is/ignored/in/native/mode.usda",
        "env_origins_w": origins, "goals_local_xy": goals_local, "goals_world_xy": goals_world,
        "task_times_s": [0.05, 6.9, 11.3, 14.2],
        "body_names": list(NATIVE_BODY_NAMES),
        "root_link_pose_w": roots, "body_link_pose_w": body_rows,
    }


def make_packet(*, sequence: int = 1, source_pid: int = 4321,
                run_id: str = "test-run") -> dict:
    origins = [[11.0, -11.0, 0.0], [11.0, 11.0, 0.0], [-11.0, -11.0, 0.0], [-11.0, 11.0, 0.0]]
    body_rows = []
    roots = []
    for origin in origins:
        poses = [[origin[0] + index * 0.1, origin[1] + index * 0.2, 0.15,
                  0.0, 0.0, 0.0, 1.0] for index in range(len(BODY_NAMES))]
        body_rows.append(poses)
        roots.append(poses[0])
    goals_local = [[0.6, 0.9], [2.5, 3.9], [9.9, 5.4], [10.1, 6.3]]
    goals_world = [[goal[0] + origin[0], goal[1] + origin[1]]
                   for goal, origin in zip(goals_local, origins, strict=True)]
    return {
        "schema_version": 1,
        "sequence": sequence,
        "timestamp_unix_ns": 1_800_000_000_000_000_000,
        "timestamp_monotonic_ns": 1_000_000_000,
        "source_pid": source_pid,
        "run_id": run_id,
        "source": "isaac_field._collect_sim_cpu_samples",
        "sim_time_s": float(sequence),
        "quaternion_order": "xyzw",
        "robot_usd_path": "C:/assets/robot.usda",
        "terrain_usd_path": "C:/assets/terrain.usda",
        "env_origins_w": origins,
        "goals_local_xy": goals_local,
        "goals_world_xy": goals_world,
        "task_times_s": [0.05, 6.9, 11.3, 14.2],
        "body_names": list(BODY_NAMES),
        "root_link_pose_w": roots,
        "body_link_pose_w": body_rows,
    }


def test_packet_selects_environment_and_converts_world_goal_once():
    selected = viewer.validate_packet(make_packet(), env_index=1)

    assert selected["body_names"] == BODY_NAMES
    assert selected["origin"] == (11.0, 11.0, 0.0)
    assert selected["body_poses"][1][0:3] == pytest.approx((11.1, 11.2, 0.15))
    assert selected["goal_local_xy"] == pytest.approx((2.5, 3.9))
    assert selected["episode_record_time_s"] == pytest.approx(6.9)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw.update(quaternion_order="wxyz"), "quaternion_order"),
        (lambda raw: raw.update(body_names=raw["body_names"][:-1]), "count"),
        (lambda raw: raw["body_link_pose_w"][0].pop(), "count"),
        (lambda raw: raw.update(robot_usd_path="relative/robot.usda"), "absolute"),
        (lambda raw: raw.update(run_id=""), "run_id"),
    ],
)
def test_rejects_invalid_schema_and_identity_fields(change, message):
    raw = make_packet()
    change(raw)

    with pytest.raises(ValueError, match=message):
        viewer.validate_packet(raw)


def test_rejects_non_finite_selected_pose():
    raw = make_packet()
    raw["body_link_pose_w"][0][3][0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        viewer.validate_packet(raw)


def test_pose_conversion_uses_xyzw_and_subtracts_environment_origin_once():
    position, wxyz = viewer.pose_to_local_wxyz(
        (12.0, -9.0, 2.0, 0.0, 0.0, 2**0.5, 2**0.5),
        (11.0, -11.0, 1.0),
    )

    assert position == pytest.approx((1.0, 2.0, 1.0))
    assert wxyz == pytest.approx((2**-0.5, 0.0, 0.0, 2**-0.5))

    with pytest.raises(ValueError, match="non-zero"):
        viewer.pose_to_local_wxyz((0, 0, 0, 0, 0, 0, 0), (0, 0, 0))


def test_stale_detection_uses_age_or_sequence_stall_and_holds_last_pose():
    assert viewer.state_is_stale(5.1, 0.1)
    assert viewer.state_is_stale(0.1, 5.1)
    assert not viewer.state_is_stale(4.9, 4.9)
    assert viewer.state_is_stale(None, 0.0)


def test_service_packet_age_reads_timestamp_from_validated_schema_packet():
    packet = viewer.validate_packet(make_packet())
    now_unix_ns = packet["timestamp_unix_ns"] + 1_500_000_000

    assert viewer._state_packet_age_s(packet, now_unix_ns) == pytest.approx(1.5)
    assert viewer._state_packet_age_s(None, now_unix_ns) is None


def test_reader_retries_torn_json_and_retains_last_decoded_packet(tmp_path):
    state_path = tmp_path / "live.json"
    reader = viewer.LatestPacketReader(state_path)

    assert reader.poll() is None
    state_path.write_text('{"schema_version":', encoding="utf-8")
    assert reader.poll() is None
    assert reader.error
    assert reader.last_good is None

    packet = make_packet()
    state_path.write_text(json.dumps(packet), encoding="utf-8")
    assert reader.poll() == packet
    assert reader.last_good == packet

    state_path.write_text('{ broken', encoding="utf-8")
    assert reader.poll() is None
    assert reader.last_good == packet


def test_packet_identity_locks_run_pid_assets_body_map_and_monotonic_sequence():
    identity = viewer.PacketIdentity(expected_run_id="test-run", expected_source_pid=4321)
    first = viewer.validate_packet(make_packet(sequence=10))
    assert identity.accept(first)
    assert not identity.accept(first)

    with pytest.raises(ValueError, match="moved backwards"):
        identity.accept(viewer.validate_packet(make_packet(sequence=9)))
    with pytest.raises(ValueError, match="run_id"):
        identity.accept(viewer.validate_packet(make_packet(sequence=11, run_id="another-run")))
    with pytest.raises(ValueError, match="source_pid"):
        identity.accept(viewer.validate_packet(make_packet(sequence=11, source_pid=8765)))

    changed_asset = make_packet(sequence=11)
    changed_asset["robot_usd_path"] = "C:/assets/other-robot.usda"
    with pytest.raises(ValueError, match="asset paths"):
        identity.accept(viewer.validate_packet(changed_asset))


def test_usd_geometry_maps_to_nearest_body_and_leaves_unmapped_paths_out():
    body_paths = {
        "base_link": "/FieldRobot/base_link",
        "left_wheel_0_wheel_link": "/FieldRobot/base_link/left_wheel_0_wheel_link",
        "gimbal_yaw_link": "/FieldRobot/gimbal_yaw_link",
    }
    paths = (
        "/FieldRobot/base_link/body_collision_visual",
        "/FieldRobot/base_link/left_wheel_0_wheel_link/wheel_collision_visual",
        "/FieldRobot/terrain_mesh",
    )

    grouped = viewer.group_geometry_by_body(body_paths, paths)

    assert grouped["base_link"] == (paths[0],)
    assert grouped["left_wheel_0_wheel_link"] == (paths[1],)
    assert grouped["gimbal_yaw_link"] == ()


def test_geometry_topology_triangulates_polygons_and_rejects_bad_boundaries():
    triangles = viewer._triangulate_face_counts((4, 3), (0, 1, 2, 3, 3, 4, 5))

    assert triangles == ((0, 1, 2), (0, 2, 3), (3, 4, 5))
    with pytest.raises(ValueError, match="topology"):
        viewer._triangulate_face_counts((2,), (0, 1))
    with pytest.raises(ValueError, match="unreferenced"):
        viewer._triangulate_face_counts((3,), (0, 1, 2, 3))


def test_viewer_has_no_forbidden_runtime_imports_and_reports_loaded_modules():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert not imports.intersection({"isaacsim", "isaaclab", "isaac_field", "omni", "torch"})
    assert viewer.forbidden_modules_loaded({"viser", "pxr.Usd", "isaacsim.core", "torch"}) == (
        "isaacsim.core", "torch"
    )

    # The combined suite may already have imported Isaac Lab or Torch in this
    # pytest process. Check the viewer's import graph in a clean interpreter.
    check_imports = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import view_training_state_web as viewer; "
        "loaded = viewer.forbidden_modules_loaded(sys.modules.keys()); "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_imports, str(SCRIPT.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_status_file_is_atomic_and_records_live_identity_and_no_simulation(tmp_path):
    packet = viewer.validate_packet(make_packet(sequence=7))
    args = argparse.Namespace(state_file="C:/state/live.json", env_index=0)
    status = viewer._status_record(args, packet, 0.2, False, True, (), "", "http://127.0.0.1:8790")
    path = tmp_path / "viewer_status.json"

    viewer._write_status(path, status)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["server_pid"] > 0 and saved["serverpid"] == saved["server_pid"]
    assert saved["source_pid"] == saved["sourcepid"] == 4321
    assert saved["sequence"] == 7
    assert saved["model_loaded"] is True
    assert saved["no_simulation"] is True
    assert saved["consumer_lightweight"] is saved["consumerlightweight"] is True
    assert saved["forbidden_modules_loaded"] == []
    assert saved["url"] == "http://127.0.0.1:8790"
    assert not list(tmp_path.glob("*.tmp"))


def test_native_context_verifies_cpu_model_source_contract_and_all_real_geometry(tmp_path):
    model_path, trial = make_native_trial(tmp_path)

    context = viewer.load_native_viewer_context(model_path, trial)

    assert context.trial_name == "native_view_test"
    assert context.engine == "newton_mujoco_warp"
    assert context.backend_revision == NATIVE_REVISION
    assert context.body_names == NATIVE_BODY_NAMES
    assert context.publish_interval_steps == 5
    assert context.sim_dt_s == pytest.approx(0.001)
    assert context.decimation == 50
    assert context.policy_dt_s == pytest.approx(0.05)
    assert context.model_sha256 == _file_sha256(model_path)
    assert tuple(context.geometry.by_body) == NATIVE_BODY_NAMES
    assert all(len(context.geometry.by_body[name]) == 1 for name in NATIVE_BODY_NAMES)
    assert context.geometry.proxy_bodies == ()
    assert len(context.geometry.terrain.vertices) == 4
    assert {vertex[2] for vertex in context.geometry.terrain.vertices} == {0.0}
    assert max(abs(vertex[0]) for vertex in context.geometry.terrain.vertices) == 200.0


def test_native_geometry_uses_mujoco_half_sizes_and_wxyz_geom_rotation():
    half = math.sqrt(0.5)
    box = viewer._native_geom_mesh({
        "name": "box", "type": "box", "size_m": [1.0, 2.0, 3.0],
        "local_pos_xyz_m": [1.0, 2.0, 3.0], "local_quat_wxyz": [half, 0.0, 0.0, half],
        "rgba": [1.0, 0.0, 0.0, 1.0],
    })
    assert box.vertices[0] == pytest.approx((3.0, 1.0, 0.0))
    assert box.color == (255, 0, 0)

    cylinder = viewer._native_geom_mesh({
        "name": "wheel", "type": "cylinder", "size_m": [0.5, 2.0, 0.0],
        "local_pos_xyz_m": [1.0, 2.0, 3.0], "local_quat_wxyz": [half, half, 0.0, 0.0],
        "rgba": [0.0, 1.0, 0.0, 1.0],
    })
    ys = [vertex[1] - 2.0 for vertex in cylinder.vertices]
    zs = [vertex[2] - 3.0 for vertex in cylinder.vertices]
    assert max(ys) == pytest.approx(2.0)
    assert min(ys) == pytest.approx(-2.0)
    assert max(zs) == pytest.approx(0.5)
    assert min(zs) == pytest.approx(-0.5)


def test_native_packet_uses_live_pose_goal_and_episode_data_without_usd_fallback(tmp_path):
    model_path, trial = make_native_trial(tmp_path)
    context = viewer.load_native_viewer_context(model_path, trial)
    raw = make_native_packet()

    selected = viewer.validate_native_packet(raw, 1, context)

    assert selected["source"] == "native Newton MuJoCo-Warp trainer"
    assert "isaac_field" not in selected["source"]
    assert selected["run_id"] == trial.name
    assert selected["source_pid"] == 842
    assert selected["body_names"] == NATIVE_BODY_NAMES
    assert selected["origin"] == (11.0, 11.0, 0.0)
    assert selected["body_poses"][0][:3] == pytest.approx((11.0, 11.0, 0.15))
    assert selected["goal_local_xy"] == pytest.approx((2.5, 3.9))
    assert selected["episode_record_time_s"] == pytest.approx(6.9)

    identity = viewer.NativePacketIdentity(context, expected_source_pid=842)
    assert identity.accept(selected)
    assert not identity.accept(selected)
    with pytest.raises(ValueError, match="run_id"):
        viewer.validate_native_packet(make_native_packet(run_id="other_run"), 1, context)
    with pytest.raises(ValueError, match="source_pid"):
        viewer.NativePacketIdentity(context, expected_source_pid=1).accept(selected)


def test_native_preflight_fails_closed_on_engine_revision_topology_or_missing_geom(tmp_path):
    model_path, trial = make_native_trial(tmp_path)
    identity_path = trial / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["physics"] = "newton_xpbd"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="must use newton_mujoco_warp"):
        viewer.load_native_viewer_context(model_path, trial)

    model_path, trial = make_native_trial(tmp_path / "revision")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["backend_identity_cpu"]["backend_revision"] = "different_revision"
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="backend revisions"):
        viewer.load_native_viewer_context(model_path, trial)

    model_path, trial = make_native_trial(tmp_path / "missing-geom")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    geoms = payload["mujoco_static_model"]["geometries"]
    geoms[-1]["body_id"] = geoms[-2]["body_id"]
    geoms[-1]["body_name"] = geoms[-2]["body_name"]
    geoms[-1]["newton_body_id"] = geoms[-2]["newton_body_id"]
    geoms[-1]["viewer_name"] = geoms[-2]["viewer_name"]
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing for 19-body links"):
        viewer.load_native_viewer_context(model_path, trial)


def test_native_packet_rejects_changed_body_map_and_nonfinite_world_pose(tmp_path):
    model_path, trial = make_native_trial(tmp_path)
    context = viewer.load_native_viewer_context(model_path, trial)
    raw = make_native_packet()
    raw["body_names"][1], raw["body_names"][2] = raw["body_names"][2], raw["body_names"][1]
    with pytest.raises(ValueError, match="body topology"):
        viewer.validate_native_packet(raw, 0, context)

    raw = make_native_packet()
    raw["body_link_pose_w"][2][4][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        viewer.validate_native_packet(raw, 0, context)


def test_native_panel_shows_training_scope_and_sample_cadence_without_long_identity(tmp_path, monkeypatch):
    model_path, trial = make_native_trial(tmp_path)
    context = viewer.load_native_viewer_context(model_path, trial)
    packet = viewer.validate_native_packet(make_native_packet(), 0, context)

    summary = viewer._panel_text(packet, 0.25, False, 0, 0, "", context)
    details = viewer._native_details_text(packet, 0.25, context)

    assert context.training_env_count == 64
    assert "**Training state:** LIVE" in summary
    assert "**Training environments:** 64 · first 4 exported · viewing 1 / 4" in summary
    assert "**Scene sampling:** every 0.25 simulation seconds (5 control steps)" in summary
    assert "Episode time" in summary
    assert "PID 842" in summary
    assert "State time" in summary
    assert "Distance to goal" in summary
    assert "19 / 19" in summary
    assert "sensor data not shown" in summary
    assert "isaac_field" not in summary
    assert "backend_revision" not in summary
    assert context.trial_name in details
    assert context.task_identity_sha256 in details
    assert context.backend_source_sha256 in details
    assert "zero simulation steps" in details
    assert "no GPU geometry arrays were read" in details

    monkeypatch.setattr(viewer, "_LANGUAGE", "zh")
    chinese_summary = viewer._panel_text(packet, 0.25, False, 0, 0, "", context)
    assert "**训练环境：** 64 个 · 实时导出前 4 个 · 当前查看第 1 个（1/4）" in chinese_summary
    assert "**画面采样步距：** 0.25 秒仿真时间（每 5 个控制步）" in chinese_summary
    assert "本回合时间" in chinese_summary
