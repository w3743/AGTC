"""CPU-only validation of paired gSDE experiment bookkeeping."""

import copy

from scripts.compare_isaac_exploration import _pair_initial_conditions, preflight


def _report(label):
    rows = []
    for seed in range(82_001, 82_025):
        rows.append({
            "seed": seed,
            "reset_initial_condition": {
                "seed": seed,
                "start_pose_xy_yaw": [1.0, 2.0, 0.1],
                "goal_xy": [3.0, 4.0],
                "plant_count": 16,
                "plant_xy_sha256": "same-map-hash",
            },
            "trajectory_start_matches_reset": True,
            "variant": label,
        })
    return {"rows": rows}


def test_exploration_pairing_accepts_same_24_reset_conditions():
    reports = {name: _report(name) for name in ("baseline", "gaussian", "gsde")}
    result = _pair_initial_conditions(reports)
    assert result == {"valid": True, "paired_seed_count": 24, "mismatches": []}


def test_exploration_pairing_rejects_changed_map_pose_goal_or_trace_reset():
    reports = {name: _report(name) for name in ("baseline", "gaussian", "gsde")}
    reports["gsde"] = copy.deepcopy(reports["gsde"])
    row = reports["gsde"]["rows"][0]
    row["reset_initial_condition"]["plant_xy_sha256"] = "different-map"
    row["reset_initial_condition"]["goal_xy"] = [3.01, 4.0]
    row["trajectory_start_matches_reset"] = False
    result = _pair_initial_conditions(reports)
    assert result["valid"] is False
    assert result["mismatches"] == [{
        "seed": 82_001,
        "variant": "gsde",
        "differences": ["plant_xy_sha256", "goal_xy", "trajectory_start_matches_reset"],
    }]


def test_launch_preflight_refuses_nonprepared_status(tmp_path):
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text('{"status":"running"}', encoding="utf-8")
    try:
        preflight(tmp_path)
    except ValueError as exc:
        assert "unchanged prepared run" in str(exc)
    else:
        raise AssertionError("launch must refuse an already-started output directory")
