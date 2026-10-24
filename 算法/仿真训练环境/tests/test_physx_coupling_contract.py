"""Constraint topology can be checked without loading Kit or changing assets."""
import math

import pytest

from isaac_field.assets import FrontGuideSpec, ROBOT_SPEC, ensure_robot_asset
from isaac_field.physx_drive import (
    coupling_relations,
    ensure_physx_robot_asset,
    same_side_wheel_coupling_metadata,
)


def test_default_guides_preserve_fourteen_per_world_relations():
    guide = FrontGuideSpec()
    rows = coupling_relations(ROBOT_SPEC, guide)
    assert len(rows) == 14
    assert len({row['follower'] for row in rows}) == 14
    for side in ('left', 'right'):
        side_rows = [row for row in rows if row['leader'].startswith(side)]
        assert len(side_rows) == 7
        assert all(row['follower'].startswith(side) for row in side_rows)
        supports = [row for row in side_rows if row['kind'] == 'support_wheel']
        assert len(supports) == 6 and all(row['ratio'] == 1 for row in supports)
        passive = next(row for row in side_rows if row['kind'] == 'front_guide')
        assert math.isclose(passive['ratio'], ROBOT_SPEC.wheel_radius_m / guide.radius_m)


def test_coupling_switch_does_not_remove_optional_passive_guide_constraint():
    rows = coupling_relations(ROBOT_SPEC, FrontGuideSpec(), couple_support_wheels=False)
    assert len(rows) == 2 and all(row['kind'] == 'front_guide' for row in rows)
    assert coupling_relations(ROBOT_SPEC, None, couple_support_wheels=False) == []
    assert same_side_wheel_coupling_metadata(True, guide_relations_per_world=2)['relations_per_world'] == 14


def test_physx_usd_asset_authors_and_remaps_fourteen_mimic_relations(tmp_path):
    """Read real PhysxSchema data from a referenced USD stage without starting Kit."""
    PhysxSchema = pytest.importorskip(
        "pxr.PhysxSchema",
        reason="requires Isaac Sim PhysxSchema bindings initialized by Kit",
    )
    from pxr import Usd, UsdGeom

    guide_spec = FrontGuideSpec()
    legacy_source = ensure_robot_asset(tmp_path)
    legacy_bytes = legacy_source.read_bytes()
    guide_source = ensure_robot_asset(tmp_path, front_guide_spec=guide_spec)
    guide_bytes = guide_source.read_bytes()

    physx_asset = ensure_physx_robot_asset(tmp_path, front_guide_spec=guide_spec)
    physx_bytes = physx_asset.read_bytes()

    assert guide_source != legacy_source
    assert physx_asset != guide_source
    assert physx_bytes != guide_bytes
    assert legacy_source.read_bytes() == legacy_bytes
    assert guide_source.read_bytes() == guide_bytes

    reused_asset = ensure_physx_robot_asset(tmp_path, front_guide_spec=guide_spec)
    assert reused_asset == physx_asset
    assert reused_asset.read_bytes() == physx_bytes
    assert guide_source.read_bytes() == guide_bytes
    assert legacy_source.read_bytes() == legacy_bytes

    world_file = tmp_path / "two_worlds.usda"
    stage = Usd.Stage.CreateNew(str(world_file))
    UsdGeom.Xform.Define(stage, "/World")
    robot_paths = ("/World/env_0/Robot", "/World/env_1/Robot")
    for robot_path in robot_paths:
        UsdGeom.Xform.Define(stage, robot_path).GetPrim().GetReferences().AddReference(str(physx_asset))
    stage.GetRootLayer().Save()
    del stage

    stage = Usd.Stage.Open(str(world_file))
    assert stage is not None
    rows = coupling_relations(ROBOT_SPEC, guide_spec)
    assert len(rows) == 14
    for robot_path, other_robot_path in ((robot_paths[0], robot_paths[1]), (robot_paths[1], robot_paths[0])):
        world_relation_count = 0
        for row in rows:
            follower_path = f"{robot_path}/{row['follower']}"
            follower = stage.GetPrimAtPath(follower_path)
            assert follower.IsValid(), follower_path

            mimic = PhysxSchema.PhysxMimicJointAPI.Get(follower, "rotX")
            assert mimic, follower_path
            targets = [str(path) for path in mimic.GetReferenceJointRel().GetTargets()]
            expected_target = f"{robot_path}/{row['leader']}"
            assert targets == [expected_target]
            assert targets[0].startswith(robot_path + "/")
            assert not targets[0].startswith(other_robot_path + "/")
            assert math.isclose(
                float(mimic.GetGearingAttr().Get()),
                -float(row["ratio"]),
                rel_tol=0,
                abs_tol=1e-7,
            )
            world_relation_count += 1
        assert world_relation_count == 14
