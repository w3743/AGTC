"""CPU checks for canonical drive-coupling metadata."""

import pytest

from isaac_field.newton_drive import same_side_wheel_coupling_metadata


def test_legacy_wheel_coupling_metadata_is_unchanged_without_guides():
    assert same_side_wheel_coupling_metadata(True) == {
        "enabled": True,
        "treatment": "same_side_wheel0_master_six_mimic_followers_each_side",
        "relation": "q_follower = 0 + 1*q_leader",
        "relations_per_world": 12,
        "leader_wheel_index_per_side": 0,
        "follower_wheel_indices_per_side": [1, 2, 3, 4, 5, 6],
        "equality_solref": [0.004, 1.0],
        "equality_solimp": [0.999, 0.9999, 0.001, 0.5, 2.0],
        "execution": "retain existing per-wheel actuators, PI state and configured effort limits",
        "source_usd_changed": False,
    }
    assert same_side_wheel_coupling_metadata(False) == {
        "enabled": False,
        "treatment": "independent_wheel_joints",
        "relation": None,
        "relations_per_world": 0,
        "leader_wheel_index_per_side": None,
        "follower_wheel_indices_per_side": [],
        "equality_solref": None,
        "equality_solimp": None,
        "execution": "retain existing per-wheel actuators, PI state and configured effort limits",
        "source_usd_changed": False,
    }


@pytest.mark.parametrize(
    ("enabled", "guide_count", "total", "wheel_count"),
    [(True, 2, 14, 12), (True, 0, 12, 12), (False, 2, 2, 0), (False, 0, 0, 0)],
)
def test_guided_coupling_metadata_reports_each_relation_class(
    enabled, guide_count, total, wheel_count,
):
    metadata = same_side_wheel_coupling_metadata(
        enabled,
        guide_relations_per_world=guide_count,
    )
    assert metadata["relations_per_world"] == total
    assert metadata["wheel_relations_per_world"] == wheel_count
    assert metadata["guide_relations_per_world"] == guide_count
    if guide_count:
        assert metadata["front_guide_relation"] == (
            "q_guide = (R_main/R_guide)*q_same_side_wheel0"
        )
    else:
        assert "front_guide_relation" not in metadata


@pytest.mark.parametrize("guide_count", [-1, 1, 3, 2.5, True])
def test_guided_coupling_metadata_rejects_invalid_relation_counts(guide_count):
    with pytest.raises(ValueError, match="guide_relations_per_world"):
        same_side_wheel_coupling_metadata(True, guide_relations_per_world=guide_count)
