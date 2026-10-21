"""The viewer must preserve its selected backend when rebuilding a scene."""
import pytest
import numpy as np

from field_sim.env import Config, FieldEnv
from field_sim.server import Application


class ResourceEnv(FieldEnv):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.closed = 0

    def close(self):
        self.closed += 1


def config():
    return Config(size=8, plants=0, validate_scene=False, terrain_enabled=False,
                  lidar_rays=8, depth_height=2, depth_width=2)


def test_backend_survives_reset_and_releases_previous_resource():
    created = []

    def factory(cfg, mode):
        result = ResourceEnv(cfg, mode)
        created.append(result)
        return result

    app = Application(config(), env_factory=factory, backend='gazebo', initially_paused=True)
    assert app.paused and app.controller == 'manual'
    assert app.state()['backend'] == 'gazebo'
    old_scene_id = app.env.scene_id
    app.reset_environment({'seed': 8, 'mode': 'goto'})
    assert len(created) == 2 and app.env is created[-1]
    assert created[0].closed == 1
    assert app.env.mode == 'goto' and app.env.scene_id == old_scene_id + 1
    assert app.env.observation_space.contains(app.obs)
    app.env.close()


def test_failed_backend_reset_keeps_current_world_and_closes_replacement():
    old = ResourceEnv(config(), 'search')
    app = Application(config(), env_factory=lambda *_: old, backend='gazebo')

    class Broken(ResourceEnv):
        def reset(self, **kwargs):
            raise RuntimeError('engine unavailable')

    broken = Broken(config(), 'search')
    app.env_factory = lambda *_: broken
    with pytest.raises(RuntimeError, match='engine unavailable'):
        app.reset_environment({'seed': 9})
    assert app.env is old and old.closed == 0 and broken.closed == 1
    assert app.state()['backend'] == 'gazebo'
    old.close()


def test_saved_policy_can_be_replaced_without_rebuilding_the_world(tmp_path):
    from field_sim.policy import make_model

    env = ResourceEnv(config(), 'goto')
    app = Application(config(), env_factory=lambda *_: env, backend='gazebo',
                      initially_paused=True)
    model = make_model(env, goal_gain=12.5)
    model.num_timesteps = 1234  # Metadata fixture; no training is performed here.
    checkpoint = tmp_path / 'candidate.zip'
    model.save(checkpoint)
    before = env.robot.copy()
    app.load_checkpoint(checkpoint)
    assert app.env is env and env.closed == 0
    assert app.paused and app.controller == 'policy'
    assert app.state()['policy_training_steps'] == 1234
    assert app.runner.model.policy.features_extractor.goal_gain == 12.5
    np.testing.assert_array_equal(env.robot, before)
    action = app.runner.action(app.obs)
    assert action.shape == (4,) and np.isfinite(action).all()
    previous = app.runner
    with pytest.raises(ValueError, match='checkpoint'):
        app.load_checkpoint(tmp_path / 'missing.zip')
    assert app.runner is previous and app.env is env
    env.close()
