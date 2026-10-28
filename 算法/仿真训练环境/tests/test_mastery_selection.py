"""Cross-scene safety and promotion budget regressions."""
from copy import deepcopy

import pytest

import scripts.train_isaac_mastery as runner
from test_mastery_runner import workspace, _prepared, _install_launch_fakes


def _protocol_and_report(workspace, monkeypatch, *, budget=24_576):
    _install_launch_fakes(monkeypatch)
    output = _prepared(workspace, budget=budget)
    protocol = runner.preflight(output)
    protocol["device"] = "cpu"
    baseline = output / "untrained_baseline.pt"
    baseline.write_bytes(b"test baseline")
    return output, protocol, runner._global_probe(output, protocol, baseline)


def test_more_easy_success_cannot_hide_lost_field_success(workspace, monkeypatch):
    _, protocol, best = _protocol_and_report(workspace, monkeypatch)
    field_seed = protocol["selection"]["seed_banks"][-1][0]
    for row in best["rows"]:
        row["success"] = row["seed"] == field_seed
    candidate = deepcopy(best)
    easy_seeds = protocol["selection"]["seed_banks"][0][:2]
    for row in candidate["rows"]:
        row["success"] = row["seed"] in easy_seeds
    comparison = runner._select_best(protocol, candidate, best)
    assert comparison["comparable"]
    assert comparison["candidate_metrics"]["clean_successes"] > comparison["best_metrics"]["clean_successes"]
    assert not comparison["replace_best"]
    assert comparison["assessment"] == "suite_member_regression"


def test_improved_clean_success_without_scene_regressions_replaces_best(workspace, monkeypatch):
    _, protocol, best = _protocol_and_report(workspace, monkeypatch)
    candidate = deepcopy(best)
    candidate["rows"][0]["success"] = True
    comparison = runner._select_best(protocol, candidate, best)
    assert comparison["replace_best"]
    assert all(member["non_regression"] for member in comparison["member_checks"])


def test_remaining_single_block_cannot_promote_to_an_untrainable_stage(workspace, monkeypatch):
    output, protocol, best = _protocol_and_report(workspace, monkeypatch)
    controller = runner._controller(protocol, runner._mastery_specs(protocol, best), best)
    for _ in range(2):
        plan = controller.next_training_plan(max_timesteps=8192)
        controller.record_training(plan["timesteps"], sampled_stage=plan["sampled_stage"],
                                   actual_sampling=plan["sampling"])
        stage = protocol["stages"][0]
        report = runner._evaluate_stage(output, protocol, output / "untrained_baseline.zip", stage,
                                       protocol["mastery"]["seed_banks"][stage["name"]])
        for row in report["rows"]:
            row["success"] = True
        report = runner._attach_probe(report, purpose="mastery", stage=stage["name"])
        decision = controller.record_probe(report)
    assert decision["active_stage"]["name"] == "near"
    assert decision["promotion_blocked_reason"] == "insufficient_global_budget_for_next_stage"


def test_preflight_rejects_dependency_drift(workspace, monkeypatch):
    output = _prepared(workspace)
    monkeypatch.setattr(runner.foundation, "_package_versions", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="runtime dependencies"):
        runner.preflight(output)


def test_geometry_contract_normalizes_json_tuples_before_comparison(workspace, monkeypatch):
    monkeypatch.setattr(runner.foundation, "_robot_geometry", lambda: {"spec": {"size": (0.8, 0.3, 0.3)}})
    output = _prepared(workspace)
    assert runner.preflight(output)["robot_geometry"]["spec"]["size"] == [0.8, 0.3, 0.3]
