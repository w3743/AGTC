import math

import pytest

from isaac_field.curriculum_control import (
    CurriculumControlConfig,
    CurriculumController,
    ProbeSpec,
    compare_best_checkpoints,
)


def _stage(name, plants, goal_distance, angle):
    return {
        "name": name,
        "mode": "goto",
        "goal_distance": goal_distance,
        "goal_angle": angle,
        "plants": plants,
    }


STAGES = (
    _stage("near", 0, (0.45, 1.05), math.radians(15)),
    _stage("field8", 8, (1.0, 3.0), math.pi / 3.0),
)


def _contract(plants=0):
    return {
        "action_config": {"schema": "direct-track-speed-v1"},
        "drive_config": {"interface": "coupled-v7"},
        "observation_config": {"schema": "sensor-minimal-v6"},
        "damage_config": {"metric": "root-protection-v1"},
        "damage_metric": "root-protection-v1",
        "distance_cost": 0.0,
        "goal_distance": [1.0, 3.0],
        "goal_angle_rad": math.pi,
        "mode": "goto",
        "deterministic": True,
        "evaluation_config": {
            "plants": plants,
            "terrain_enabled": False,
            "environment_seed": 80_003,
        },
    }


def _probe_config():
    mastery = {
        "near": ProbeSpec(tuple(range(88_101, 88_117)), _contract(0)),
        "field8": ProbeSpec(tuple(range(88_201, 88_217)), _contract(8)),
    }
    best = ProbeSpec(tuple(range(89_101, 89_117)), _contract(16))
    return CurriculumControlConfig(
        total_timesteps=65_536,
        probe_interval_timesteps=8_192,
        training_seeds=(7_001, 7_002),
        mastery_probes=mastery,
        best_probe=best,
        required_consecutive_passes=2,
        min_clean_success_rate=0.8,
        max_damaged_episodes=0,
        max_mean_damage=0.0,
        max_total_collisions=0,
        minimum_stage_timesteps=8_192,
    )


def _report(purpose, stage, seeds, contract, *, clean_count=None, damage_seed=None, collisions_seed=None):
    rows = []
    for index, seed in enumerate(seeds):
        clean = clean_count is None or index < clean_count
        damage = 1 if seed == damage_seed else 0
        collisions = 1 if seed == collisions_seed else 0
        rows.append({
            "seed": seed,
            "success": bool(clean),
            "damage": damage,
            "collisions": collisions,
            "reset_initial_condition": {
                "plant_xy_sha256": f"plants-{seed}",
                "start_pose_xy_yaw": [float(seed) / 100_000.0, 2.0, 0.1],
                "goal_xy": [4.0, 5.0],
            },
            "trajectory_start_matches_reset": True,
        })
    return {
        "checkpoint": f"{stage}-{purpose}.zip",
        **contract,
        "probe": {
            "schema": "isaac-field-fixed-probe-v1",
            "purpose": purpose,
            "probe_stage": stage,
            "seeds": list(seeds),
            "config": contract,
        },
        "rows": rows,
    }


def _mastery_report(stage="near", clean_count=13, *, damaged=False):
    seeds = tuple(range(88_101, 88_117)) if stage == "near" else tuple(range(88_201, 88_217))
    plants = 0 if stage == "near" else 8
    return _report(
        "mastery",
        stage,
        seeds,
        _contract(plants),
        clean_count=clean_count,
        damage_seed=seeds[0] if damaged else None,
    )


def _complete_block(controller):
    plan = controller.next_training_plan(max_timesteps=8_192)
    actual = dict(plan["sampling"])
    controller.record_training(
        plan["timesteps"],
        sampled_stage=plan["sampled_stage"],
        actual_sampling=actual,
    )
    return plan


def test_plan_exposes_real_sampling_and_rejects_stage_name_only_relabel():
    controller = CurriculumController(STAGES, _probe_config())
    plan = controller.next_training_plan()
    assert plan["timesteps"] == 8_192
    assert plan["sampling"] == {
        "mode": "goto",
        "goal_distance": [0.45, 1.05],
        "goal_angle": math.radians(15),
        "plants": 0,
    }
    wrong_sampling = dict(plan["sampling"], plants=8)
    with pytest.raises(ValueError, match="actual environment sampling"):
        controller.record_training(
            plan["timesteps"], sampled_stage="near", actual_sampling=wrong_sampling
        )
    assert controller.consumed_timesteps == 0


def test_two_consecutive_fixed_mastery_probes_promote_and_keep_old_stage_mixed():
    controller = CurriculumController(STAGES, _probe_config())
    for _ in range(2):
        _complete_block(controller)
        assert controller.decision()["probe_due"] is True
        with pytest.raises(ValueError, match="expected probe_stage 'near'"):
            controller.record_probe(_mastery_report("field8", clean_count=13))
        assert controller.decision()["probe_due"] is True
        decision = controller.record_probe(_mastery_report("near", clean_count=13))
    assert decision["active_stage"]["name"] == "field8"
    assert decision["last_event"]["promotion"] == "advanced"
    cycle = decision["long_run_cycle"]
    distribution = {
        row["stage"]["name"]: row["weight"]
        for row in cycle["target_distribution"]
    }
    assert distribution == {"near": 0.25, "field8": 0.75}
    assert cycle["block_order"] == ["near", "field8", "field8", "field8"]
    assert controller.decision()["active_stage_timesteps"] == 0
    replay = _complete_block(controller)
    assert replay["sampled_stage"] == "near"
    assert controller.decision()["probe_due"] is False
    for _ in range(2):
        plan = _complete_block(controller)
        assert plan["sampled_stage"] == "field8"
        assert controller.decision()["probe_due"] is True
        decision = controller.record_probe(_mastery_report("field8", clean_count=13))
    assert decision["mastered"] is True


def test_probe_interval_counts_active_stage_steps_not_replay_blocks():
    config = _probe_config()
    config = CurriculumControlConfig(**{**config.__dict__, "total_timesteps": 65_536})
    controller = CurriculumController(STAGES, config)
    _complete_block(controller)
    controller.record_probe(_mastery_report("near", clean_count=13))
    _complete_block(controller)
    promoted = controller.record_probe(_mastery_report("near", clean_count=13))
    assert promoted["active_stage"]["name"] == "field8"

    replay = _complete_block(controller)
    assert replay["sampled_stage"] == "near"
    assert controller.decision()["probe_due"] is False
    _complete_block(controller)
    assert controller.decision()["probe_due"] is True
    controller.record_probe(_mastery_report("field8", clean_count=13))
    for _ in range(2):
        current = controller.next_training_plan()
        assert current["sampled_stage"] == "field8"
        _complete_block(controller)
        assert controller.decision()["probe_due"] is True
        controller.record_probe(_mastery_report("field8", clean_count=0))
    replay = controller.next_training_plan()
    assert replay["sampled_stage"] == "near"
    _complete_block(controller)
    assert controller.decision()["probe_due"] is False


def test_failed_probe_resets_streak_and_safety_is_mandatory():
    controller = CurriculumController(STAGES, _probe_config())
    _complete_block(controller)
    first = controller.record_probe(_mastery_report("near", clean_count=13))
    assert first["consecutive_passes"] == 1
    _complete_block(controller)
    failed = controller.record_probe(_mastery_report("near", clean_count=15, damaged=True))
    assert failed["consecutive_passes"] == 0
    assert failed["active_stage"]["name"] == "near"


def test_global_budget_can_block_promotion_without_marking_stage_mastered():
    config = _probe_config()
    config = CurriculumControlConfig(**{
        **config.__dict__,
        "total_timesteps": 8_192,
        "required_consecutive_passes": 1,
    })
    controller = CurriculumController(STAGES, config)
    _complete_block(controller)
    decision = controller.record_probe(_mastery_report("near", clean_count=13))
    assert decision["active_stage"]["name"] == "near"
    assert decision["last_event"]["promotion"] == "blocked_by_budget"
    assert decision["promotion_blocked_reason"] == "insufficient_global_budget_for_next_stage"
    assert decision["mastered"] is False


def test_held_out_and_training_overlapping_probe_seeds_are_rejected_but_new_dev_bank_is_allowed():
    cfg = _probe_config()
    assert 88_101 in cfg.mastery_probes["near"].seeds
    bad = ProbeSpec((83_001,), _contract(0))
    with pytest.raises(ValueError, match="830xx"):
        CurriculumController(STAGES, CurriculumControlConfig(
            **{**cfg.__dict__, "mastery_probes": {"near": bad, "field8": cfg.mastery_probes["field8"]}}
        ))
    overlap = ProbeSpec((7_001,), _contract(0))
    with pytest.raises(ValueError, match="overlap training seeds"):
        CurriculumController(STAGES, CurriculumControlConfig(
            **{**cfg.__dict__, "mastery_probes": {"near": overlap, "field8": cfg.mastery_probes["field8"]}}
        ))


def test_best_selection_requires_same_global_probe_stage_contract_and_initial_conditions():
    seeds = tuple(range(89_101, 89_117))
    config = _contract(16)
    best = _report("best_checkpoint", "global_suite", seeds, config, clean_count=8)
    candidate = _report("best_checkpoint", "global_suite", seeds, config, clean_count=10)
    comparison = compare_best_checkpoints(candidate, best)
    assert comparison["comparable"] is True
    assert comparison["replace_best"] is True
    assert comparison["assessment"] == "improved_under_predeclared_safety_criterion"

    wrong_stage = _report("best_checkpoint", "near", seeds, config, clean_count=16)
    rejected = compare_best_checkpoints(wrong_stage, best)
    assert rejected["assessment"] == "incomparable_probe_stage"
    assert rejected["replace_best"] is False

    changed_initial = _report("best_checkpoint", "global_suite", seeds, config, clean_count=16)
    changed_initial["rows"][0]["reset_initial_condition"]["goal_xy"] = [9.0, 5.0]
    rejected = compare_best_checkpoints(changed_initial, best)
    assert rejected["assessment"] == "invalid_initial_conditions"


def test_missing_safety_fields_and_initial_condition_proof_are_rejected():
    seeds = tuple(range(89_101, 89_117))
    report = _report("best_checkpoint", "global_suite", seeds, _contract(16), clean_count=16)
    del report["rows"][0]["collisions"]
    with pytest.raises(ValueError, match="collisions"):
        compare_best_checkpoints(report, report)
    report = _report("best_checkpoint", "global_suite", seeds, _contract(16), clean_count=16)
    report["rows"][0]["trajectory_start_matches_reset"] = False
    with pytest.raises(ValueError, match="does not prove"):
        compare_best_checkpoints(report, report)


def test_controller_state_round_trips_pending_plan_and_active_sampling_options():
    controller = CurriculumController(STAGES, _probe_config())
    plan = controller.next_training_plan()
    restored = CurriculumController.from_state_dict(controller.state_dict())
    assert restored.next_training_plan() == plan
    _complete_block(restored)
    restored_state = restored.state_dict()
    again = CurriculumController.from_state_dict(restored_state)
    assert again.decision()["probe_due"] is True
    assert again.decision()["active_stage"]["sampling_options"] == {"plants": 0}


def test_probe_cannot_be_reused_to_build_a_passing_streak_without_training():
    controller = CurriculumController(STAGES, _probe_config())
    _complete_block(controller)
    report = _mastery_report("near", clean_count=13)
    controller.record_probe(report)
    with pytest.raises(RuntimeError, match="only when"):
        controller.record_probe(report)


def test_best_checkpoint_validation_can_run_with_curriculum_promotion_disabled():
    config = _probe_config()
    config = CurriculumControlConfig(**{
        **config.__dict__,
        "enable_curriculum_promotion": False,
        "mastery_probes": {},
    })
    controller = CurriculumController(STAGES, config)
    plan = controller.next_training_plan()
    controller.record_training(
        plan["timesteps"],
        sampled_stage=plan["sampled_stage"],
        actual_sampling=plan["sampling"],
    )
    assert controller.decision()["probe_due"] is False
    global_seeds = tuple(range(89_101, 89_117))
    checked = controller.validate_best_probe(
        _report("best_checkpoint", "global_suite", global_seeds, _contract(16), clean_count=16)
    )
    assert checked["probe_stage"] == "global_suite"

