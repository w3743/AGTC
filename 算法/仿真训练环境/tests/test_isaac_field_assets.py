"""Focused contract checks for the local Isaac rover USD asset."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import re

import pytest

import isaac_field.assets as assets

from isaac_field.assets import (
    BASE_LINK_PATH,
    GIMBAL_PITCH_JOINT,
    GIMBAL_PITCH_LINK_PATH,
    GIMBAL_YAW_JOINT,
    GIMBAL_YAW_LINK_PATH,
    FrontGuideSpec,
    ROBOT_ROOT_PATH,
    ROBOT_SPEC,
    author_corn_leaf,
    ensure_robot_asset,
    robot_articulation_cfg,
    robot_asset_spec_sha256,
)


def test_robot_proxy_mass_geometry_and_sensor_mount_contract():
    assert ROBOT_SPEC.assembled_mass_kg == pytest.approx(30.0)
    assert ROBOT_SPEC.total_mass_kg == pytest.approx(30.0)
    assert (ROBOT_SPEC.body_length_m, ROBOT_SPEC.body_width_m) == pytest.approx((0.8, 0.3))
    assert ROBOT_SPEC.track_ground_contact_length_m == pytest.approx(0.595)
    assert ROBOT_SPEC.vehicle_width_m == pytest.approx(0.47)
    assert ROBOT_SPEC.drive_sprocket_radius_m == pytest.approx(0.03)
    assert ROBOT_SPEC.drive_sprocket_diameter_m == pytest.approx(0.06)
    assert ROBOT_SPEC.drive_reduction_ratio == pytest.approx(4.0)
    assert ROBOT_SPEC.track_separation_m == pytest.approx(0.41)
    assert ROBOT_SPEC.wheel_radius_m == pytest.approx(0.045)
    assert ROBOT_SPEC.wheel_width_m == pytest.approx(0.06)
    assert ROBOT_SPEC.wheel_count_per_side == 7
    assert ROBOT_SPEC.reference_height_m + ROBOT_SPEC.gimbal_offset_z_m == pytest.approx(0.4)
    assert (
        ROBOT_SPEC.chassis_mass_kg * ROBOT_SPEC.chassis_com_offset_z_m
        + 2.0 * ROBOT_SPEC.wheel_count_per_side * ROBOT_SPEC.wheel_mass_kg * ROBOT_SPEC.wheel_z_m
        + 2.0 * ROBOT_SPEC.gimbal_link_mass_kg * ROBOT_SPEC.gimbal_offset_z_m
    ) == pytest.approx(0.0)
    assert ROBOT_SPEC.pitch_limits_deg == (-25.0, 60.0)
    assert len(ROBOT_SPEC.wheel_specs) == 14
    assert sorted({wheel.x_m for wheel in ROBOT_SPEC.wheel_specs}) == pytest.approx(
        [-0.2975, -0.1983333333, -0.0991666667, 0.0, 0.0991666667, 0.1983333333, 0.2975]
    )
    assert ROBOT_SPEC.wheel_spacing_m == pytest.approx(0.595 / 6.0)
    assert ROBOT_SPEC.wheel_spacing_m > 2.0 * ROBOT_SPEC.wheel_radius_m
    assert ROBOT_SPEC.reference_height_m + ROBOT_SPEC.wheel_z_m == pytest.approx(ROBOT_SPEC.wheel_radius_m)
    assert {wheel.side for wheel in ROBOT_SPEC.wheel_specs} == {"left", "right"}
    assert {round(wheel.y_m, 6) for wheel in ROBOT_SPEC.wheel_specs} == {-0.205, 0.205}


def test_invalid_leaf_mesh_is_rejected_before_loading_pxr():
    with pytest.raises(ValueError, match="face_vertex_indices"):
        author_corn_leaf(
            stage=None,
            prim_path="/World/plant_2/leaf_1",
            plant_id=2,
            leaf_id=1,
            points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            face_vertex_counts=[3],
            face_vertex_indices=[0, 1, 3],
            normals=None,
            color=(0.2, 0.5, 0.1),
        )


def test_usd_robot_asset_masses_links_colliders_and_gimbal_limits(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    asset_path = ensure_robot_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath().pathString == ROBOT_ROOT_PATH

    prims = list(stage.Traverse())
    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert len(rigid_bodies) == 17  # chassis, fourteen wheels, and two gimbal links
    assert len(collisions) == 15  # chassis box and fourteen wheel cylinders
    assert len(joints) == 16  # fourteen wheel axes and the yaw/pitch gimbal
    assert sum(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()) for prim in rigid_bodies) == pytest.approx(30.0)

    pitch = UsdPhysics.RevoluteJoint.Get(stage, f"{ROBOT_ROOT_PATH}/{GIMBAL_PITCH_JOINT}")
    assert pitch.GetBody0Rel().GetTargets() == [stage.GetPrimAtPath(GIMBAL_YAW_LINK_PATH).GetPath()]
    assert pitch.GetBody1Rel().GetTargets() == [stage.GetPrimAtPath(GIMBAL_PITCH_LINK_PATH).GetPath()]
    assert pitch.GetAxisAttr().Get() == "Y"
    assert pitch.GetLowerLimitAttr().Get() == pytest.approx(-25.0)
    assert pitch.GetUpperLimitAttr().Get() == pytest.approx(60.0)
    pitch_frame = pitch.GetLocalRot0Attr().Get()
    assert float(pitch_frame.GetReal()) == pytest.approx(0.0, abs=1e-6)
    assert tuple(float(v) for v in pitch_frame.GetImaginary()) == pytest.approx((0.0, 0.0, 1.0))

    yaw = UsdPhysics.RevoluteJoint.Get(stage, f"{ROBOT_ROOT_PATH}/{GIMBAL_YAW_JOINT}")
    assert yaw.GetAxisAttr().Get() == "Z"
    assert yaw.GetLocalPos0Attr().Get()[2] == pytest.approx(0.25)
    assert not stage.GetPrimAtPath(GIMBAL_PITCH_LINK_PATH).HasAPI(UsdPhysics.CollisionAPI)
    assert float(stage.GetPrimAtPath(ROBOT_ROOT_PATH).GetAttribute("field:comHeightAssumptionM").Get()) == pytest.approx(
        0.15
    )
    root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    assert root.GetAttribute("field:trackGroundContactLengthM").Get() == pytest.approx(0.595)
    assert root.GetAttribute("field:trackContactWidthM").Get() == pytest.approx(0.06)
    assert root.GetAttribute("field:trackCenterSeparationM").Get() == pytest.approx(0.41)
    assert root.GetAttribute("field:bodyWidthM").Get() == pytest.approx(0.3)
    assert root.GetAttribute("field:vehicleOverallWidthM").Get() == pytest.approx(0.47)
    assert "0.47 m overall width" in root.GetAttribute("field:trackSeparationStatus").Get()
    assert root.GetAttribute("field:driveSprocketRadiusM").Get() == pytest.approx(0.03)
    assert root.GetAttribute("field:driveSprocketDiameterM").Get() == pytest.approx(0.06)
    assert root.GetAttribute("field:driveReductionRatio").Get() == pytest.approx(4.0)
    assert root.GetAttribute("field:proxyRollerRadiusM").Get() == pytest.approx(0.045)
    assert root.GetAttribute("field:assetSpecSha256").Get() == robot_asset_spec_sha256(ROBOT_SPEC)
    assert "separate from 45 mm" in root.GetAttribute("field:driveGeometryNote").Get()


def test_robot_asset_reuses_matching_spec_and_atomically_refreshes_changed_spec(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    from pxr import Usd

    asset_path = ensure_robot_asset(tmp_path)
    original_author = assets._author_robot
    authored_specs = []

    def counting_author(stage, spec):
        authored_specs.append(spec)
        original_author(stage, spec)

    monkeypatch.setattr(assets, "_author_robot", counting_author)
    assert ensure_robot_asset(tmp_path) == asset_path
    assert authored_specs == []

    changed_spec = replace(assets.ROBOT_SPEC, drive_reduction_ratio=4.25)
    monkeypatch.setattr(assets, "ROBOT_SPEC", changed_spec)
    refreshed_path = ensure_robot_asset(tmp_path)
    assert refreshed_path == asset_path
    assert authored_specs == [changed_spec]

    stage = Usd.Stage.Open(str(refreshed_path))
    root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    assert root.GetAttribute("field:driveReductionRatio").Get() == pytest.approx(4.25)
    assert root.GetAttribute("field:assetSpecSha256").Get() == robot_asset_spec_sha256(changed_spec)


def test_front_guide_spec_is_separate_from_driven_wheel_specs_and_hashes_stably():
    guide = FrontGuideSpec()
    free_guide = replace(guide, rotation_coupling="free")
    left, right = guide.wheel_specs(ROBOT_SPEC)

    assert (left.name, right.name) == ("left_front_guide", "right_front_guide")
    assert (left.link_name, right.link_name) == ("left_front_guide_link", "right_front_guide_link")
    assert (left.joint_name, right.joint_name) == ("left_front_guide_joint", "right_front_guide_joint")
    assert all(re.fullmatch(r".*_wheel_joint", wheel.joint_name) is None for wheel in (left, right))
    assert [
        component
        for wheel in (left, right)
        for component in (wheel.x_m, wheel.y_m, wheel.z_m)
    ] == pytest.approx([0.42, 0.205, -0.055, 0.42, -0.205, -0.055])
    assert all((wheel.radius_m, wheel.width_m, wheel.mass_kg) == pytest.approx((0.045, 0.06, 0.25)) for wheel in (left, right))
    assert guide.rotation_coupling == "same_side_track_speed"
    assert free_guide.rotation_coupling == "free"
    assert len(ROBOT_SPEC.wheel_specs) == 14
    assert all("front_guide" not in wheel.name for wheel in ROBOT_SPEC.wheel_specs)

    legacy_payload = json.dumps(asdict(ROBOT_SPEC), sort_keys=True, separators=(",", ":"), allow_nan=False)
    legacy_hash = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
    assert robot_asset_spec_sha256(ROBOT_SPEC) == legacy_hash
    assert robot_asset_spec_sha256(ROBOT_SPEC, guide) != legacy_hash
    assert asdict(guide)["rotation_coupling"] == "same_side_track_speed"
    assert robot_asset_spec_sha256(ROBOT_SPEC, free_guide) != robot_asset_spec_sha256(ROBOT_SPEC, guide)
    assert robot_asset_spec_sha256(ROBOT_SPEC, replace(guide, x_m=0.43)) != robot_asset_spec_sha256(
        ROBOT_SPEC, guide
    )
    with pytest.raises(ValueError, match="rotation_coupling"):
        FrontGuideSpec(rotation_coupling="independent_drive")


def test_front_guide_usd_is_passive_and_keeps_legacy_asset_separate(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    legacy_path = ensure_robot_asset(tmp_path)
    legacy_bytes = legacy_path.read_bytes()
    guide_spec = FrontGuideSpec()
    guide_path = ensure_robot_asset(tmp_path, front_guide_spec=guide_spec)
    assert guide_path != legacy_path
    assert guide_path.parent.name == f"front_guides_{robot_asset_spec_sha256(ROBOT_SPEC, guide_spec)}"
    assert legacy_path.read_bytes() == legacy_bytes
    guide_bytes = guide_path.read_bytes()
    assert ensure_robot_asset(tmp_path, front_guide_spec=guide_spec) == guide_path
    assert guide_path.read_bytes() == guide_bytes

    free_spec = replace(guide_spec, rotation_coupling="free")
    free_path = ensure_robot_asset(tmp_path, front_guide_spec=free_spec)
    assert free_path != guide_path
    assert free_path.parent.name == f"front_guides_{robot_asset_spec_sha256(ROBOT_SPEC, free_spec)}"
    free_bytes = free_path.read_bytes()
    assert ensure_robot_asset(tmp_path, front_guide_spec=free_spec) == free_path
    assert free_path.read_bytes() == free_bytes
    assert guide_path.read_bytes() == guide_bytes
    assert legacy_path.read_bytes() == legacy_bytes

    changed_guide_path = ensure_robot_asset(tmp_path, front_guide_spec=replace(guide_spec, x_m=0.43))
    assert changed_guide_path != guide_path
    assert guide_path.read_bytes() == guide_bytes
    assert legacy_path.read_bytes() == legacy_bytes

    stage = Usd.Stage.Open(str(guide_path))
    assert stage is not None
    root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    assert root.GetAttribute("field:frontGuideRotationCoupling").Get() == "same_side_track_speed"
    assert "No independent DriveAPI" in root.GetAttribute("field:frontGuidePassiveSemantics").Get()
    assert "R_main/R_guide" in root.GetAttribute("field:frontGuideModelNote").Get()
    assert root.GetAttribute("field:frontGuideAngularVelocityRatioIfCoupled").Get() == pytest.approx(1.0)
    prims = list(stage.Traverse())
    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert len(rigid_bodies) == 19  # chassis, fourteen driven rollers, two guides, two gimbal links
    assert len(collisions) == 17  # chassis, fourteen rollers, and two guides
    assert len(joints) == 18  # fourteen driven rollers, two guides, and two gimbal axes
    assert sum(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()) for prim in rigid_bodies) == pytest.approx(30.0)
    assert root.GetAttribute("field:totalMassKg").Get() == pytest.approx(30.0)
    assert root.GetAttribute("field:assembledMassKg").Get() == pytest.approx(30.0)
    assert root.GetAttribute("field:frontGuideWheelCount").Get() == 2
    assert root.GetAttribute("field:frontGuidePassive").Get() is True
    assert root.GetAttribute("field:frontGuideRadiusM").Get() == pytest.approx(0.045)
    assert root.GetAttribute("field:frontGuideWidthM").Get() == pytest.approx(0.06)
    assert root.GetAttribute("field:frontGuideMassPerWheelKg").Get() == pytest.approx(0.25)
    assert root.GetAttribute("field:frontGuideAxisCenterXM").Get() == pytest.approx(0.42)
    assert root.GetAttribute("field:frontGuideZOffsetFromSupportM").Get() == pytest.approx(0.05)
    assert root.GetAttribute("field:frontGuideAxisCenterZM").Get() == pytest.approx(-0.055)
    assert root.GetAttribute("field:frontGuideLeftCenterYM").Get() == pytest.approx(0.205)
    assert root.GetAttribute("field:frontGuideRightCenterYM").Get() == pytest.approx(-0.205)
    assert "Trial-only" in root.GetAttribute("field:frontGuideModelNote").Get()
    assert root.GetAttribute("field:assetSpecSha256").Get() == robot_asset_spec_sha256(ROBOT_SPEC, guide_spec)

    free_stage = Usd.Stage.Open(str(free_path))
    free_root = free_stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    assert free_root.GetAttribute("field:frontGuideRotationCoupling").Get() == "free"
    assert "rotate freely" in free_root.GetAttribute("field:frontGuideModelNote").Get()
    assert free_root.GetAttribute("field:assetSpecSha256").Get() == robot_asset_spec_sha256(ROBOT_SPEC, free_spec)

    different_radius_spec = replace(guide_spec, radius_m=0.03)
    different_radius_path = ensure_robot_asset(tmp_path, front_guide_spec=different_radius_spec)
    different_radius_stage = Usd.Stage.Open(str(different_radius_path))
    different_radius_root = different_radius_stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    assert different_radius_root.GetAttribute("field:frontGuideAngularVelocityRatioIfCoupled").Get() == pytest.approx(
        ROBOT_SPEC.wheel_radius_m / different_radius_spec.radius_m
    )

    base = stage.GetPrimAtPath(BASE_LINK_PATH)
    base_mass = UsdPhysics.MassAPI(base)
    chassis_mass_kg = ROBOT_SPEC.chassis_mass_kg - 2 * guide_spec.mass_per_wheel_kg
    assert base_mass.GetMassAttr().Get() == pytest.approx(chassis_mass_kg)
    expected_base_inertia = assets._box_diagonal_inertia(
        chassis_mass_kg,
        (ROBOT_SPEC.body_length_m, ROBOT_SPEC.body_width_m, ROBOT_SPEC.body_height_m),
    )
    assert tuple(float(value) for value in base_mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        expected_base_inertia
    )
    expected_base_com_x = -2 * guide_spec.mass_per_wheel_kg * guide_spec.x_m / chassis_mass_kg
    expected_base_com_z = -(
        2 * ROBOT_SPEC.wheel_count_per_side * ROBOT_SPEC.wheel_mass_kg * ROBOT_SPEC.wheel_z_m
        + 2 * ROBOT_SPEC.gimbal_link_mass_kg * ROBOT_SPEC.gimbal_offset_z_m
        + 2 * guide_spec.mass_per_wheel_kg * (ROBOT_SPEC.wheel_z_m + guide_spec.z_offset_from_support_m)
    ) / chassis_mass_kg
    base_com = tuple(float(value) for value in base_mass.GetCenterOfMassAttr().Get())
    assert base_com == pytest.approx((expected_base_com_x, 0.0, expected_base_com_z))
    assert chassis_mass_kg * base_com[0] + 2 * guide_spec.mass_per_wheel_kg * guide_spec.x_m == pytest.approx(
        0.0, abs=1e-7
    )
    assert (
        chassis_mass_kg * base_com[2]
        + 2 * ROBOT_SPEC.wheel_count_per_side * ROBOT_SPEC.wheel_mass_kg * ROBOT_SPEC.wheel_z_m
        + 2 * ROBOT_SPEC.gimbal_link_mass_kg * ROBOT_SPEC.gimbal_offset_z_m
        + 2 * guide_spec.mass_per_wheel_kg * (ROBOT_SPEC.wheel_z_m + guide_spec.z_offset_from_support_m)
    ) == pytest.approx(0.0, abs=1e-7)

    for wheel in guide_spec.wheel_specs(ROBOT_SPEC):
        link = stage.GetPrimAtPath(f"{BASE_LINK_PATH}/{wheel.link_name}")
        joint = UsdPhysics.RevoluteJoint.Get(stage, f"{ROBOT_ROOT_PATH}/{wheel.joint_name}")
        cylinder = UsdGeom.Cylinder.Get(stage, f"{BASE_LINK_PATH}/{wheel.link_name}/front_guide_collision_visual")
        assert link.IsValid() and link.GetAttribute("field:passiveFrontGuide").Get() is True
        assert UsdPhysics.MassAPI(link).GetMassAttr().Get() == pytest.approx(wheel.mass_kg)
        assert joint.GetBody0Rel().GetTargets() == [base.GetPath()]
        assert joint.GetBody1Rel().GetTargets() == [link.GetPath()]
        assert joint.GetAxisAttr().Get() == "Y"
        assert not joint.GetLowerLimitAttr().HasAuthoredValue()
        assert not joint.GetUpperLimitAttr().HasAuthoredValue()
        assert not any("DriveAPI" in schema for schema in joint.GetPrim().GetAppliedSchemas())
        assert cylinder.GetRadiusAttr().Get() == pytest.approx(wheel.radius_m)
        assert cylinder.GetHeightAttr().Get() == pytest.approx(wheel.width_m)


def test_robot_articulation_cfg_includes_passive_guide_joints_only_for_guide_asset(tmp_path):
    pytest.importorskip("pxr")
    guide_spec = FrontGuideSpec()
    guide_asset = ensure_robot_asset(tmp_path, front_guide_spec=guide_spec)
    guide_cfg = robot_articulation_cfg(guide_asset, front_guide_spec=guide_spec)
    assert guide_cfg.init_state.joint_pos == {
        GIMBAL_YAW_JOINT: 0.0,
        GIMBAL_PITCH_JOINT: 0.0,
        "left_front_guide_joint": 0.0,
        "right_front_guide_joint": 0.0,
    }

    legacy_cfg = robot_articulation_cfg(ensure_robot_asset(tmp_path / "legacy"))
    assert legacy_cfg.init_state.joint_pos == {
        GIMBAL_YAW_JOINT: 0.0,
        GIMBAL_PITCH_JOINT: 0.0,
    }


def test_corn_leaf_usd_mesh_keeps_semantics_and_has_no_collision():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    path = author_corn_leaf(
        stage,
        "/World/plant_2/leaf_1",
        plant_id=2,
        leaf_id=1,
        points=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.1, 0.0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 2],
        normals=[(0.0, 0.0, 1.0)] * 3,
        color=(0.2, 0.5, 0.1),
        damage_state="dry",
    )
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid()
    assert prim.GetAttribute("field:plantId").Get() == 2
    assert prim.GetAttribute("field:leafId").Get() == 1
    assert prim.GetAttribute("field:damageState").Get() == "dry"
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert "SemanticsLabelsAPI:class" in prim.GetAppliedSchemas()



def test_generated_leaves_are_visible_from_both_sides_without_backend_flags(tmp_path):
    pytest.importorskip("pxr")
    import numpy as np
    from pxr import Usd, UsdGeom
    from isaac_field.scene_assets import plant_assets
    from isaac_field.task import Config, Scene

    cfg = Config(plants=1, leaves_per_plant=2, terrain_enabled=False, validate_scene=False)
    scene = Scene(cfg, np.array([[10., 10.]]), np.array([True]), np.random.default_rng(7))
    asset = plant_assets(scene, tmp_path)[0]
    stage = Usd.Stage.Open(str(asset))
    leaf = UsdGeom.Mesh.Get(stage, "/Corn/Leaf_0")
    points = np.asarray(leaf.GetPointsAttr().Get())
    faces = np.asarray(leaf.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    half = len(faces) // 2
    triangles = points[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    areas = np.linalg.norm(normals, axis=1)
    valid = areas[:half] > 1e-8
    assert valid.any()
    # Every nondegenerate surface has an opposite face at exactly the same
    # coordinates, so front-face-only renderers show the true leaf on either side.
    np.testing.assert_allclose(normals[:half][valid], -normals[half:][valid], atol=1e-7)
    assert np.all(normals[:half][valid] @ np.cross(scene.u[0], scene.v[0]) > 0)
