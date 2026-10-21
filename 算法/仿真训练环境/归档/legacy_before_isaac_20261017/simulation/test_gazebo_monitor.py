import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from field_sim.env import Config, FieldEnv
from simulation.gazebo_monitor import TrainingMonitor


def test_monitor_serves_policy_frame_and_cannot_modify_training():
    cfg = Config(size=8, plants=0, validate_scene=False, terrain_enabled=False,
                 lidar_rays=8, depth_width=2, depth_height=2)
    env = FieldEnv(cfg)
    observation, _ = env.reset(seed=7)
    monitor = TrainingMonitor(port=0)
    url = 'http://127.0.0.1:' + str(monitor.server.server_address[1])
    try:
        monitor.publish(env, steps=17, total=512, force=True)
        with urlopen(url + '/api/state') as response:
            state = json.load(response)
        with urlopen(url + '/api/scene') as response:
            scene = json.load(response)
        assert state['read_only'] and state['training']['same_environment']
        assert state['training']['steps'] == 17
        assert state['performance']['policy_ms'] is None
        assert state['scene_id'] == scene['scene_id']
        np.testing.assert_allclose(np.asarray(state['depth']['values']) / cfg.depth_range,
                                   observation['depth'][0])
        np.testing.assert_array_equal(state['depth']['valid'], observation['depth'][1])
        before = env.robot.copy()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + '/api/control', data=b'{"action":[1,0,0,0]}',
                            headers={'Content-Type': 'application/json'}))
        assert error.value.code == 403
        np.testing.assert_array_equal(env.robot, before)
        monitor.publish(env, phase='smoke', steps=0, force=True)
        state = json.loads(monitor.state_bytes)
        assert state['controller'] == 'fixed'
        assert state['performance']['loop_hz'] == 0
        assert len(monitor._intervals) == 1
    finally:
        monitor.close()
        env.close()
