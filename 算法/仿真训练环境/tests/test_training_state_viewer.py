import math

import pytest

from scripts.view_training_state import (
    packet_age_seconds,
    _apply_poses,
    _build_scene,
    _find_body_prims,
    state_is_stale,
    usd_pose_matrix_xyzw,
    validate_packet,
)


def test_world_pose_uses_xyzw_and_subtracts_origin_for_nested_links():
    half = math.sqrt(0.5)
    matrix = usd_pose_matrix_xyzw((11.0, 22.0, 3.0, 0.0, 0.0, half, half), (10.0, 20.0, 1.0))

    # USD/Gf row-vector matrix: a positive 90-degree yaw maps local +X to +Y.
    assert matrix[0][:3] == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert matrix[3] == (1.0, 2.0, 2.0, 1.0)
    # This is the full world pose relative to the selected env, suitable for a
    # nested body prim after SetResetXformStack(True), not a parent-local pose.


def test_pose_rejects_invalid_quaternion_and_nonfinite_values():
    for pose in ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, float("nan"), 0, 0, 1)):
        try:
            usd_pose_matrix_xyzw(pose, (0, 0, 0))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pose was accepted")


def test_nested_usd_body_poses_reset_parent_transforms():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Env").AddTranslateOp().Set(Gf.Vec3d(-10, -20, -1))
    robot = UsdGeom.Xform.Define(stage, "/World/Env/Robot")
    base = UsdGeom.Xform.Define(stage, "/World/Env/Robot/base_link")
    base.AddTranslateOp().Set(Gf.Vec3d(50, 50, 50))
    gimbal = UsdGeom.Xform.Define(stage, "/World/Env/Robot/base_link/gimbal_yaw_link")
    gimbal.AddTranslateOp().Set(Gf.Vec3d(3, 4, 5))

    body_prims = _find_body_prims(Usd, robot.GetPrim(), ("base_link", "gimbal_yaw_link"))
    assert set(body_prims) == {"base_link", "gimbal_yaw_link"}
    for name, world_pose in (
        ("base_link", (11, 22, 3, 0, 0, 0, 1)),
        ("gimbal_yaw_link", (11.5, 22, 3.4, 0, 0, 0, 1)),
    ):
        prim = body_prims[name]
        xformable = UsdGeom.Xformable(prim)
        op = xformable.MakeMatrixXform()
        xformable.SetResetXformStack(True)
        op.Set(Gf.Matrix4d(usd_pose_matrix_xyzw(world_pose, (10, 20, 1))))

    cache = UsdGeom.XformCache()
    assert tuple(cache.GetLocalToWorldTransform(base.GetPrim()).ExtractTranslation()) == (1, 2, 2)
    assert tuple(cache.GetLocalToWorldTransform(gimbal.GetPrim()).ExtractTranslation()) == (1.5, 2, 2.4)


def test_packet_validation_selects_env_and_requires_xyzw_contract():
    identity_pose = [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    packet = {
        "schema_version": 1,
        "sequence": 42,
        "timestamp_unix_ns": 2_000_000_000,
        "timestamp_monotonic_ns": 100,
        "source_pid": 29076,
        "run_id": "run-a04",
        "source": "isaac_field._collect_sim_cpu_samples",
        "quaternion_order": "xyzw",
        "root_link_pose_w": [identity_pose] * 4,
        "body_names": ["base_link", "gimbal_yaw_link", "gimbal_pitch_link"],
        "body_link_pose_w": [[identity_pose] * 3 for _ in range(4)],
        "env_origins_w": [[i * 20.0, 0.0, 0.0] for i in range(4)],
        "sim_time_s": 98.765,
        "task_times_s": [0.1, 0.2, 0.3, 0.4],
        "robot_usd_path": "robot.usda",
        "terrain_usd_path": "terrain.usda",
    }

    selected = validate_packet(packet, env_index=2)
    assert selected["sequence"] == 42
    assert selected["origin"] == (40.0, 0.0, 0.0)
    assert selected["sim_time_s"] == 98.765
    assert selected["episode_record_time_s"] == 0.3
    packet["quaternion_order"] = "wxyz"
    try:
        validate_packet(packet)
    except ValueError as exc:
        assert "xyzw" in str(exc)
    else:
        raise AssertionError("unsupported quaternion order was accepted")


def test_packet_age_and_sequence_stall_drive_stale_indicator():
    assert packet_age_seconds(1_000_000_000, 1_400_000_000) == 0.4
    assert not state_is_stale(1.7, 1.7)
    assert not state_is_stale(4.9, 4.9)
    assert state_is_stale(5.1, 0.1)
    assert state_is_stale(0.1, 5.1)
    assert state_is_stale(None, 0.0)


def test_nonzero_env_origin_keeps_terrain_local_and_offsets_robot(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    terrain_path = tmp_path / "terrain.usda"
    terrain_stage = Usd.Stage.CreateNew(str(terrain_path))
    terrain_root = UsdGeom.Xform.Define(terrain_stage, "/Terrain").GetPrim()
    terrain_stage.SetDefaultPrim(terrain_root)
    ground = UsdGeom.Mesh.Define(terrain_stage, "/Terrain/Ground")
    ground.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(12, 0, 0), Gf.Vec3f(12, 12, 0), Gf.Vec3f(0, 12, 0),
    ]))
    ground.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    ground.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    terrain_stage.GetRootLayer().Save()

    robot_path = tmp_path / "robot.usda"
    robot_stage = Usd.Stage.CreateNew(str(robot_path))
    robot_root = UsdGeom.Xform.Define(robot_stage, "/Robot").GetPrim()
    robot_stage.SetDefaultPrim(robot_root)
    base = UsdGeom.Xform.Define(robot_stage, "/Robot/base_link")
    base.AddTranslateOp().Set(Gf.Vec3d(50, 50, 50))
    robot_stage.GetRootLayer().Save()

    class Context:
        stage = None

        def new_stage(self):
            self.stage = Usd.Stage.CreateInMemory()

        def get_stage(self):
            return self.stage

    packet = {
        "origin": (100.0, 200.0, 0.0),
        "terrain_usd_path": str(terrain_path),
        "robot_usd_path": str(robot_path),
    }
    stage, robot_prim = _build_scene(Context(), packet, UsdGeom, UsdLux, 0)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    ground_prim = stage.GetPrimAtPath("/World/Env0/Terrain/Ground")
    terrain_xf = UsdGeom.XformCache().GetLocalToWorldTransform(ground_prim)
    ground_points = [terrain_xf.Transform(Gf.Vec3d(p)) for p in ground.GetPointsAttr().Get()]
    assert tuple(ground_points[0]) == (0, 0, 0)
    assert tuple(ground_points[2]) == (12, 12, 0)

    bodies = _find_body_prims(Usd, robot_prim, ("base_link",))
    body_packet = {
        "body_names": ("base_link",),
        "body_poses": ((111.0, 222.0, 3.0, 0.0, 0.0, 0.0, 1.0),),
        "origin": packet["origin"],
    }
    _apply_poses(Gf, UsdGeom, bodies, body_packet)
    robot_xf = UsdGeom.XformCache().GetLocalToWorldTransform(bodies["base_link"])
    assert tuple(robot_xf.ExtractTranslation()) == (11, 22, 3)
