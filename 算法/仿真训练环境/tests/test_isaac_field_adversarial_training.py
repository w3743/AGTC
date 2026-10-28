"""Closed-loop orchestration tests with fake simulator and trainer dependencies."""
from types import SimpleNamespace
import json

import numpy as np

from isaac_field.adversarial_training import run_round


class FakeScene:
    def __init__(self, xy=None):
        self.xy = np.asarray(xy if xy is not None else [[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        self.targets = np.asarray([True, False])
        self.terrain = None
        self.terrain_seed = 17


class FakeModel:
    def __init__(self, weight):
        self.weight = float(weight)

    def state_dict(self):
        return {"weight": np.asarray([self.weight], dtype=np.float32)}


class FakeAdapter:
    def __init__(self, env):
        self.engine = env
        self.cfg = env.field_cfg
        self.scene = None

    def reset(self, *, seed=None, options=None):
        self.scene = (options or {}).get("scene", FakeScene())
        return np.zeros(2, dtype=np.float32), {"seed": seed}

    def step(self, action):
        assert np.asarray(action).shape == (1,)
        return np.zeros(2, dtype=np.float32), 2.0, True, False, {
            "reason": "time_limit", "is_success": False, "damage": 1,
            "collisions": 2, "distance": 0.75,
        }


class FakeRunner:
    def __init__(self, adapter, checkpoint, *, device):
        self.model = FakeModel(0.0)

    def reset(self):
        pass

    def action(self, observation):
        return np.zeros(1, dtype=np.float32)


class FakeDesigner:
    def __init__(self, cfg, evaluator, *, seed, layout_template, terrain, edit_budget):
        self.cfg, self.evaluator = cfg, evaluator
        self.template = layout_template
        self.terrain = terrain

    def _scene_at_positions(self, positions):
        scene = FakeScene(positions)
        scene.targets = self.template.targets.copy()
        return scene

    def _layout_valid(self, positions, *, scene=None):
        return True, ""


class ConstantRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value

    def integers(self, high):
        return 0


def test_round_scores_teacher_layout_then_trains_student_and_saves_summary(tmp_path):
    native = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(mode="mixed", auto_reset=True, distance_cost=0.5,
                            goal_distance=None, goal_angle=3.14),
        field_cfg=SimpleNamespace(plants=2, dt=0.1, max_time=0.1),
        terrain=None,
        episode_results=[],
    )

    def train_teacher(designer, output, timesteps, **kwargs):
        assert timesteps == 16
        score = designer.evaluator(FakeScene())
        assert score == 2.0
        return SimpleNamespace(model=SimpleNamespace(num_timesteps=16, state_dict=lambda: {"weight": np.array([1.])}),
                               checkpoint=output.with_suffix(".zip"))

    def train_student(env, checkpoint, timesteps, output, **kwargs):
        assert env.cfg.auto_reset is True
        assert env.layout_sampler.teacher_scene_probability == 0.5
        teacher_layout = env.layout_sampler(ConstantRng(0.9))
        assert isinstance(teacher_layout, FakeScene)
        random_layout = env.layout_sampler(ConstantRng(0.1))
        np.testing.assert_array_equal(random_layout.xy, [[8.0, 8.0], [9.0, 9.0]])
        env.episode_results.append({"mode": "goto", "return": 3.0})
        return SimpleNamespace(model=FakeModel(2.0), checkpoint=output.with_suffix(".zip"),
                               trained_timesteps=timesteps, total_timesteps=timesteps)

    result = run_round(
        native, "old.zip", tmp_path, dependencies={
            "adapter_factory": FakeAdapter,
            "runner_factory": FakeRunner,
            "designer_factory": FakeDesigner,
            "teacher_trainer": train_teacher,
            "student_trainer": train_student,
            "initial_teacher_model_factory": lambda *args, **kwargs: FakeModel(0.0),
            "row_positions": lambda cfg, rng, terrain=None: np.asarray([[8.0, 8.0], [9.0, 9.0]], np.float32),
        },
    )

    assert native.cfg.auto_reset is True
    assert native.cfg.distance_cost == 0.0
    assert result["teacher_weights_changed"] is True
    assert result["student_weights_changed"] is True
    assert result["teacher_rollouts"][0]["raw_return"] == 2.0
    assert result["teacher_rollouts"][0]["damage"] == 1
    assert result["teacher_rollouts"][0]["collisions"] == 2
    assert result["teacher_rollouts"][0]["final_distance"] == 0.75
    assert result["student_episodes"] == [{"mode": "goto", "return": 3.0}]
    assert (tmp_path / "round_summary.json").is_file()
    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["target_student_backend_identity"] == {
        "simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl",
    }
    assert protocol["teacher"]["runtime_identity"] == {
        "simulator": "gymnasium", "physics_backend": "none", "learning_backend": "rsl_rl",
    }
    assert protocol["teacher"]["evaluator_kind"] == "test_or_custom_score_callback"
    assert protocol["teacher"]["student_physx_evaluated"] is False
    assert result["student_physx_evaluated"] is False
