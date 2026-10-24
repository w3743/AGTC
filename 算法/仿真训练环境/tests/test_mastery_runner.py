"""CPU integration checks for the frozen mastery experiment runner.

These tests keep the runner's prepare/preflight, environment context helpers,
probe composition, controller, budget loop, and status persistence real. Only
the simulator factory, policy/optimizer entry points, and diagnostic callback
are faked, so no training or physics simulation is started.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import scripts.train_isaac_mastery as runner


FAKE_SOURCES = (
    "isaac_field/frozen_component.py",
    "scripts/train_isaac_mastery.py",
)
ACTION_CONFIG = {"schema": "test-direct-track-v1", "dimensions": 4}
DRIVE_CONFIG = {"interface": "test-track-drive-v1"}
OBSERVATION_CONFIG = {"schema": "test-observation-v1"}
DAMAGE_CONFIG = {"metric": "root-protection-v1"}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    for relative in FAKE_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen: {relative}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "_sources", lambda: FAKE_SOURCES)
    monkeypatch.setattr(runner.foundation, "_package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(runner.foundation, "_isaac_commit", lambda: "test-commit")
    monkeypatch.setattr(runner.foundation, "_robot_geometry", lambda: {"spec": "test-proxy"})
    monkeypatch.setattr(runner.foundation, "_action_config", lambda: dict(ACTION_CONFIG))
    return root


def _prepared(root: Path, name: str = "trial", **kwargs) -> Path:
    output = root / "results" / name
    runner.prepare(output, **kwargs)
    return output


def test_prepare_freezes_without_launch_and_refuses_output_reuse(workspace, monkeypatch):
    def simulator_must_not_start(*args, **kwargs):
        raise AssertionError("prepare must not create an environment")

    monkeypatch.setattr(runner, "_open_env", simulator_must_not_start)
    output = _prepared(workspace, "prepared")
    protocol_before = (output / "protocol.json").read_bytes()
    status_before = json.loads((output / "run_status.json").read_text(encoding="utf-8"))

    assert status_before["status"] == "prepared"
    assert status_before["num_timesteps"] == 0
    assert not (output / "untrained_baseline.zip").exists()
    assert not (output / "process_exit.json").exists()
    assert runner.preflight(output)["schema"] == runner.SCHEMA
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        runner.prepare(output)
    assert (output / "protocol.json").read_bytes() == protocol_before
    assert json.loads((output / "run_status.json").read_text(encoding="utf-8")) == status_before


@pytest.mark.parametrize("mutation", ("live_source", "snapshot", "protocol"))
def test_preflight_rejects_source_and_protocol_drift(workspace, mutation):
    output = _prepared(workspace, f"drift_{mutation}")
    if mutation == "live_source":
        path = workspace / FAKE_SOURCES[0]
        path.write_text(path.read_text(encoding="utf-8") + "changed live source\n", encoding="utf-8")
        message = "frozen source mismatch"
    elif mutation == "snapshot":
        snapshot = output / "source_snapshot" / FAKE_SOURCES[0]
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + "changed snapshot\n", encoding="utf-8")
        message = "frozen source mismatch"
    else:
        protocol_path = output / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["budget"] += 512
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        message = "prepared protocol hash changed"

    with pytest.raises(ValueError, match=message):
        runner.preflight(output)


@pytest.mark.parametrize(
    "method,budget,block_steps",
    (
        ("mastery", 513, 512),
        ("mastery", 512, 513),
        ("fixed", 512, 512),
        ("fixed", 4096, 768),
    ),
)
def test_prepare_enforces_complete_512_transition_budget_alignment(
    workspace, method, budget, block_steps
):
    with pytest.raises(ValueError, match="multiple of 512|equal, complete stage rollouts"):
        _prepared(
            workspace,
            f"bad_budget_{method}_{budget}_{block_steps}",
            method=method,
            budget=budget,
            block_steps=block_steps,
        )


def test_fixed_and_mastery_arms_hold_gamma_and_gae_constant_with_single_variable_variants(workspace):
    common = {"budget": 4096, "block_steps": 512, "gamma": 0.99, "gae_lambda": 0.93}
    fixed = json.loads((_prepared(workspace, "fixed", method="fixed", **common) / "protocol.json").read_text())
    mastery = json.loads((_prepared(workspace, "mastery", method="mastery", **common) / "protocol.json").read_text())
    gamma_changed = json.loads((_prepared(
        workspace, "gamma_changed", method="mastery", **{**common, "gamma": 0.98}
    ) / "protocol.json").read_text())
    gae_changed = json.loads((_prepared(
        workspace, "gae_changed", method="mastery", **{**common, "gae_lambda": 0.92}
    ) / "protocol.json").read_text())

    assert fixed["config"] == mastery["config"] == gamma_changed["config"] == gae_changed["config"]
    assert fixed["hyperparameters"] == mastery["hyperparameters"]
    assert fixed["hyperparameters"]["gamma"] == mastery["hyperparameters"]["gamma"] == 0.99
    assert fixed["hyperparameters"]["gae_lambda"] == mastery["hyperparameters"]["gae_lambda"] == 0.93
    base_hp = mastery["hyperparameters"]
    assert {key for key in base_hp if base_hp[key] != gamma_changed["hyperparameters"][key]} == {"gamma"}
    assert {key for key in base_hp if base_hp[key] != gae_changed["hyperparameters"][key]} == {"gae_lambda"}


class _FakeArray:
    dtype = "float32"
    shape = (1,)

    @staticmethod
    def tobytes():
        return b"fake-policy-weight"


class _FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    @staticmethod
    def numpy():
        return _FakeArray()


class _FakeModel:
    def __init__(self):
        self.num_timesteps = 0
        self.policy = SimpleNamespace(
            state_dict=lambda: {"policy.weight": _FakeTensor()},
            optimizer=SimpleNamespace(state={}),
        )

    @staticmethod
    def save(path: str):
        Path(path).write_bytes(b"fake baseline checkpoint")


class _FakeEnv:
    def __init__(self, *, cfg, mode, num_envs, seed, auto_reset, goal_distance, goal_angle,
                 distance_cost, opened, rsl=True, mismatch=False):
        assert rsl is True
        self.training = bool(auto_reset)
        self.seed = int(seed)
        self.num_envs = int(num_envs)
        self.cfg = SimpleNamespace(
            mode=mode,
            goal_distance=tuple(goal_distance),
            goal_angle=float(goal_angle),
            distance_cost=float(distance_cost),
        )
        self.field_cfg = replace(cfg, plants=cfg.plants + int(mismatch and auto_reset))
        self.drive_config = dict(DRIVE_CONFIG)
        self.observation_config = dict(OBSERVATION_CONFIG)
        self.action_config = dict(ACTION_CONFIG)
        self.tasks = [SimpleNamespace(damage_config=dict(DAMAGE_CONFIG))]
        self.diagnostic_step_infos = tuple({"step": index} for index in range(num_envs))
        self.closed = False
        opened.append(self)

    def close(self):
        assert not self.closed
        self.closed = True


class _FakeDiagnosticsCallback:
    def __init__(self, path, stage, info_provider):
        self.path = Path(path)
        self.stage = stage
        self.info_provider = info_provider


def _install_launch_fakes(monkeypatch, *, mismatch=False, actual_sampling_drift=False, train_error=None):
    from isaac_field import runtime as isaac_runtime
    from isaac_field import rsl_rl_policy

    # The workflow is exercised with fake environments and policies; own a
    # no-op outer runtime so the CPU test cannot start Isaac Sim.
    monkeypatch.setattr(isaac_runtime, "launch_app", lambda *args, **kwargs: nullcontext())
    opened = []
    train_calls = []
    eval_calls = []
    events = []
    checkpoint_payloads = {}

    def make_env(**kwargs):
        return _FakeEnv(**kwargs, opened=opened, mismatch=mismatch)

    def train_stage(env, checkpoint, timesteps, output, *, seed, device, **kwargs):
        assert env in opened
        call = {"seed": seed, "timesteps": timesteps, "output": Path(output), **kwargs}
        train_calls.append(call)
        events.append({"kind": "initialization" if int(timesteps) == 0 else "train",
                       "timesteps": int(timesteps)})
        if train_error is not None:
            raise train_error
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        previous_steps = 0
        if checkpoint is not None:
            previous = checkpoint_payloads[Path(checkpoint)]
            previous_steps = (0 if kwargs.get("weights_transfer")
                              else int(previous.get("num_timesteps", 0)))
        total_steps = previous_steps + int(timesteps)
        target.write_bytes(f"fake RSL checkpoint {len(train_calls)}".encode("utf-8"))
        if actual_sampling_drift:
            # Model a runtime change after _open_env validated the frozen task.
            env.cfg.goal_angle += 0.01
        contract = {
            "task_config": asdict(env.field_cfg),
            "simulator": "isaac_sim",
            "physics_backend": "physx",
            "learning_backend": "rsl_rl",
        }
        checkpoint_payloads[target] = {
            "field_policy_contract": contract,
            "num_timesteps": total_steps,
            "optimizer_state_dict": {},
        }
        return SimpleNamespace(
            checkpoint=target,
            trained_timesteps=int(timesteps),
            total_timesteps=total_steps,
            model=SimpleNamespace(field_policy_contract=contract),
        )

    def evaluate_model(adapter, checkpoint, seeds, *, goal_distance, goal_angle, mode,
                       record_initial_conditions, device):
        env = adapter
        assert env in opened
        assert not env.training
        seeds = list(seeds)
        eval_calls.append({"checkpoint": Path(checkpoint), "seeds": seeds, "plants": env.field_cfg.plants})
        events.append({"kind": "evaluate", "checkpoint": Path(checkpoint), "seeds": seeds})
        rows = []
        for seed in seeds:
            rows.append({
                "seed": int(seed), "success": False, "damage": 0, "collisions": 0,
                "damage_metric": DAMAGE_CONFIG["metric"],
                "reset_initial_condition": {
                    "plant_xy_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                    "start_pose_xy_yaw": [float(seed % 7), 1.0, 0.0],
                    "goal_xy": [2.0, 3.0],
                },
                "trajectory_start_matches_reset": True,
            })
        return {
            "action_config": dict(ACTION_CONFIG),
            "drive_config": dict(DRIVE_CONFIG),
            "observation_config": dict(OBSERVATION_CONFIG),
            "damage_config": dict(DAMAGE_CONFIG),
            "damage_metric": DAMAGE_CONFIG["metric"],
            "distance_cost": 0.0,
            "goal_distance": list(goal_distance),
            "goal_angle_rad": float(goal_angle),
            "mode": mode,
            "deterministic": True,
            "seeds": seeds,
            "rows": rows,
        }

    fake_env = ModuleType("isaac_field.env")
    fake_env.make_env = make_env
    fake_env.SingleFieldAdapter = lambda env: env
    fake_learning = ModuleType("isaac_field.learning")
    fake_learning.DEFAULT_INITIAL_CHECKPOINT = Path("fake-initial-checkpoint.pt")
    fake_learning.train_stage = train_stage
    fake_learning.evaluate_model = evaluate_model
    fake_rsl_policy = ModuleType("isaac_field.rsl_rl_policy")
    fake_rsl_policy.load_rsl_checkpoint_payload = lambda path, **kwargs: checkpoint_payloads[Path(path)]
    for name, module in (
        ("isaac_field.env", fake_env),
        ("isaac_field.learning", fake_learning),
        ("isaac_field.rsl_rl_policy", fake_rsl_policy),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.foundation, "_live_environment_contract", lambda env: {})
    monkeypatch.setattr(runner.foundation, "_runtime_assets", lambda env: [])
    return SimpleNamespace(
        opened=opened,
        train_calls=train_calls,
        eval_calls=eval_calls,
        events=events,
    )


def test_launch_uses_real_context_helpers_runs_to_budget_and_keeps_baseline_best(
    workspace, monkeypatch
):
    fakes = _install_launch_fakes(monkeypatch)
    output = _prepared(workspace, "launch_success", budget=512, block_steps=512)
    protocol_before = (output / "protocol.json").read_bytes()

    summary = runner.launch(output, device="cpu")

    assert summary["num_timesteps"] == summary["budget"] == 512
    assert summary["training_finished"] is True
    assert summary["capability_improvement_established"] is False
    assert len(fakes.train_calls) == 2
    assert fakes.train_calls[-1]["timesteps"] == 512
    assert fakes.train_calls[0]["timesteps"] == 0
    assert (output / "untrained_baseline.pt").is_file()
    manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["best_checkpoint"]) == output / "untrained_baseline.pt"
    assert Path(manifest["latest_checkpoint"]) == fakes.train_calls[-1]["output"]
    assert Path(summary["best_checkpoint"]) == output / "untrained_baseline.pt"

    # The baseline global suite must be evaluated through evaluate_model before
    # the fake training call, with every configured selection seed bank.
    selection_banks = json.loads((output / "protocol.json").read_text())["selection"]["seed_banks"]
    baseline_selection_positions = [
        index for index, event in enumerate(fakes.events)
        if event["kind"] == "evaluate"
        and event["checkpoint"] == output / "untrained_baseline.pt"
        and event["seeds"] in selection_banks
    ]
    assert len(baseline_selection_positions) == len(selection_banks)
    first_train_position = next(index for index, event in enumerate(fakes.events) if event["kind"] == "train")
    assert max(baseline_selection_positions) < first_train_position
    assert all(env.closed for env in fakes.opened)
    assert len(fakes.opened) == len(fakes.eval_calls) + 2  # evals, baseline init, and one training context
    assert (output / "protocol.json").read_bytes() == protocol_before
    assert json.loads((output / "python_outcome.json").read_text())["intended_exit_code"] == 0
    assert json.loads((output / "run_status.json").read_text())["status"] == "completed"


def test_launch_records_failure_closes_contexts_and_reraises_original_exception(
    workspace, monkeypatch
):
    failure = RuntimeError("fake train failure sentinel")
    fakes = _install_launch_fakes(monkeypatch, train_error=failure)
    output = _prepared(workspace, "launch_exception", budget=512, block_steps=512)

    with pytest.raises(RuntimeError, match="fake train failure sentinel") as caught:
        runner.launch(output, device="cpu")

    assert caught.value is failure
    assert fakes.train_calls
    assert all(env.closed for env in fakes.opened)
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    process_exit = json.loads((output / "python_outcome.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["exit_code"] == 1
    assert "fake train failure sentinel" in status["exception"]
    assert process_exit["status"] == "failed"
    assert process_exit["intended_exit_code"] == 1
    assert process_exit["num_timesteps"] == 0


def test_launch_rejects_actual_sampling_mismatch_after_closing_training_context(
    workspace, monkeypatch
):
    fakes = _install_launch_fakes(monkeypatch, actual_sampling_drift=True)
    output = _prepared(workspace, "launch_sampling_mismatch", budget=512, block_steps=512)

    with pytest.raises(ValueError, match="actual environment sampling values differ"):
        runner.launch(output, device="cpu")

    assert len(fakes.train_calls) == 2
    assert fakes.train_calls[-1]["timesteps"] == 512
    assert all(env.closed for env in fakes.opened)
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    process_exit = json.loads((output / "python_outcome.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "actual environment sampling values differ" in status["exception"]
    assert process_exit["intended_exit_code"] == 1
    assert process_exit["num_timesteps"] == 512
