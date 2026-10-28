"""CPU checks for the opt-in atomic live-state packet."""
import json

import numpy as np
import pytest
import torch

from isaac_field.__main__ import _extract_live_state_file
from isaac_field.live_state import LiveStateExporter
from test_sampling_copy_cache import _bare_env


def _exporter(path, body_names=("base_link", "gimbal_pitch_link", "wheel_link")):
    return LiveStateExporter(
        path,
        robot_usd_path="C:/assets/robot.usda",
        terrain_usd_path="C:/assets/terrain.usda",
        body_names=body_names,
    )


def test_live_state_packet_is_atomic_complete_and_explicit_about_frames(tmp_path):
    path = tmp_path / "live.json"
    exporter = _exporter(path)
    exporter.write(
        root_link_pose_w=np.asarray([[1, 2, 3, 0, 0, 0, 1]], dtype=np.float32),
        body_link_pose_w=np.zeros((1, 3, 7), dtype=np.float32),
        env_origins_w=np.asarray([[10, 20, 0]], dtype=np.float32),
        goals_local_xy=np.asarray([[4, 5]], dtype=np.float32),
        task_times_s=np.asarray([1.25]),
    )

    packet = json.loads(path.read_text(encoding="utf-8"))
    assert packet["schema_version"] == 1
    assert packet["sequence"] == 1
    assert packet["timestamp_unix_ns"] > 0
    assert packet["timestamp_monotonic_ns"] > 0
    assert packet["source_pid"] == exporter.source_pid
    assert packet["run_id"] == exporter.run_id
    assert packet["quaternion_order"] == "xyzw"
    assert packet["robot_usd_path"] == "C:/assets/robot.usda"
    assert packet["terrain_usd_path"] == "C:/assets/terrain.usda"
    assert packet["body_names"] == ["base_link", "gimbal_pitch_link", "wheel_link"]
    assert packet["root_link_pose_w"] == [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]
    assert packet["goals_local_xy"] == [[4.0, 5.0]]
    assert packet["goals_world_xy"] == [[14.0, 25.0]]
    assert packet["task_times_s"] == [1.25]
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_atomic_replace_retains_last_packet_and_reports_recovery(tmp_path, monkeypatch, caplog):
    path = tmp_path / "live.json"
    exporter = _exporter(path)
    args = dict(
        root_link_pose_w=np.zeros((1, 7)), body_link_pose_w=np.zeros((1, 3, 7)),
        env_origins_w=np.zeros((1, 3)), goals_local_xy=np.zeros((1, 2)), task_times_s=np.zeros(1),
    )
    exporter.write(**args)
    previous = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("isaac_field.live_state.os.replace", fail_replace)
    assert not exporter.write(**args)
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []
    assert exporter.failed_publish_count == 1
    assert "export_error" in caplog.text

    monkeypatch.undo()
    assert exporter.write(**args)
    recovered = json.loads(path.read_text(encoding="utf-8"))
    assert recovered["publish_status"] == "recovered"
    assert recovered["failed_publish_count"] == 1
    assert recovered["last_write_error"] == "OSError: simulated disk failure"
    assert recovered["source_start_time_unix_ns"] > 0

    assert exporter.write(**args)
    latest = json.loads(path.read_text(encoding="utf-8"))
    assert latest["publish_status"] == "ok"
    assert latest["failed_publish_count"] == 1


def test_live_state_run_identity_uses_explicit_environment_value(tmp_path, monkeypatch):
    monkeypatch.setenv("ISAAC_FIELD_LIVE_STATE_RUN_ID", "a03-live-ui")
    exporter = _exporter(tmp_path / "live.json")
    assert exporter.run_id == "a03-live-ui"


def test_exported_body_pose_mapping_refreshes_after_partial_reset(tmp_path):
    env = _bare_env()
    env.rover.body_names = ["base_link", "gimbal_pitch_link", "wheel_link"]
    body_poses = torch.zeros((2, 3, 7), dtype=torch.float32)
    body_poses[..., 6] = 1.0
    body_poses[:, 0, :3] = env.rover.data.root_link_pose_w.torch[:, :3]
    body_poses[0, 1, :3] = torch.tensor([11.25, 22.0, 0.3])
    body_poses[1, 1, :3] = torch.tensor([101.25, 202.0, 0.3])
    body_poses[0, 2, 0] = 12.0
    body_poses[1, 2, 0] = 102.0
    env.rover.data.body_link_pose_w.torch = body_poses
    env.tasks[0].goal = np.asarray([3.0, 4.0])
    env.tasks[1].goal = np.asarray([5.0, 6.0])
    env.tasks[0].time = 0.1
    env.tasks[1].time = 0.2
    env._sim_step_counter = 0
    path = tmp_path / "live.json"
    env._live_state_exporter = _exporter(path, env.rover.body_names)

    env._refresh_state()
    first = json.loads(path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(first["body_link_pose_w"][1][1][:3], [101.25, 202.0, 0.3])
    assert first["goals_world_xy"] == [[13.0, 24.0], [105.0, 206.0]]

    # Reset only env 1. The next packet must carry its new physical pose and goal.
    env.rover.data.body_link_pose_w.torch[1, 1, :3] = torch.tensor([109.5, 211.0, 0.45])
    env.tasks[1].goal = np.asarray([7.0, 8.0])
    env.tasks[1].time = 0.0
    env._invalidate_cached_samples()
    env._refresh_state()
    second = json.loads(path.read_text(encoding="utf-8"))
    assert second["sequence"] == first["sequence"] + 1
    np.testing.assert_allclose(second["body_link_pose_w"][0], first["body_link_pose_w"][0])
    np.testing.assert_allclose(second["body_link_pose_w"][1][1][:3], [109.5, 211.0, 0.45])
    assert second["task_times_s"] == [0.1, 0.0]
    assert second["goals_world_xy"] == [[13.0, 24.0], [107.0, 208.0]]


def test_packet_clock_uses_global_physics_steps_and_marks_task_sample_phase(tmp_path):
    env = _bare_env()
    env.rover.body_names = ["gimbal_pitch_link"]
    env.tasks[0].goal = np.asarray([3.0, 4.0])
    env.tasks[1].goal = np.asarray([5.0, 6.0])
    env.tasks[0].time = 0.05
    env.tasks[1].time = 0.1
    env._sim_step_counter = 125
    path = tmp_path / "live.json"
    env._live_state_exporter = _exporter(path, env.rover.body_names)

    env._refresh_state()
    packet = json.loads(path.read_text(encoding="utf-8"))
    assert packet["sim_time_s"] == pytest.approx(0.125)
    assert packet["task_times_s"] == [0.05, 0.1]
    assert packet["sample_phase"] == "state_read_before_task_bookkeeping"
    assert packet["task_time_semantics"].startswith("FieldTask.time recorded at sample time")


def test_default_disabled_keeps_one_existing_host_transfer_and_writes_nothing(tmp_path, monkeypatch):
    env = _bare_env()
    env._live_state_exporter = None
    path = tmp_path / "disabled.json"
    cpu_calls = []
    original_cpu = torch.Tensor.cpu

    def count_cpu(tensor, *args, **kwargs):
        cpu_calls.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", count_cpu)
    env._refresh_state(include_feedback=True)
    env._refresh_feedback(after_step=False)

    assert len(cpu_calls) == 1
    assert not path.exists()


def test_live_state_cli_argument_is_removed_before_forwarding():
    path, forwarded = _extract_live_state_file(["--seed", "7", "--live-state-file", "latest.json", "--headless"])
    assert path == "latest.json"
    assert forwarded == ["--seed", "7", "--headless"]
    assert _extract_live_state_file(["--live-state-file=latest.json", "--headless"]) == (
        "latest.json", ["--headless"]
    )
    with pytest.raises(SystemExit, match="requires PATH"):
        _extract_live_state_file(["--live-state-file"])
