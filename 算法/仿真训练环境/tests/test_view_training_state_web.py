"""CPU-only checks for the lightweight JSON-to-WebGL state consumer."""
from __future__ import annotations

import ast
import argparse
import json
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
