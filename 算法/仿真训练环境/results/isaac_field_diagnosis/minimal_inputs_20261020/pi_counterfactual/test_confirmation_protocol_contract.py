from __future__ import annotations

import json
from pathlib import Path

from compare_pi_counterfactual_runs import confirmation_protocol_mismatches


PROTOCOL_PATH = Path(__file__).with_name("confirmation_protocol.json")


def test_existing_protocol_timing_and_environment_fields_are_mapped_to_run_metadata():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    timing = protocol["timing_s"]
    expected_environment = protocol["environment"]
    run_environment = {
        "settle_s": timing["p_only_settle"],
        "drive_s": timing["drive"],
        "brake_observation_s": timing["brake_observation"],
        "policy_dt_s": timing["policy_dt_expected"],
        "physics_dt_s": timing["physics_dt_expected"],
        "decimation": timing["decimation_expected"],
        **{
            field: expected_environment[field]
            for field in (
                "plants", "terrain_enabled", "seed", "start_pose_xy_yaw", "goal_xy", "max_time_s",
            )
        },
    }
    drive_config = {"interface": expected_environment["drive_contract"]}

    assert confirmation_protocol_mismatches(run_environment, drive_config, protocol) == []

    run_environment["drive_s"] += 0.05
    mismatches = confirmation_protocol_mismatches(run_environment, drive_config, protocol)
    assert any("drive_s" in mismatch and "protocol drive" in mismatch for mismatch in mismatches)

    run_environment["drive_s"] = timing["drive"]
    run_environment["seed"] += 1
    mismatches = confirmation_protocol_mismatches(run_environment, drive_config, protocol)
    assert any("seed" in mismatch for mismatch in mismatches)

    run_environment["seed"] = expected_environment["seed"]
    drive_config["interface"] = "independent_left_right_track_speed_explicit_v4"
    mismatches = confirmation_protocol_mismatches(run_environment, drive_config, protocol)
    assert any("drive_contract" in mismatch for mismatch in mismatches)
