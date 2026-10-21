"""Bounded real-physics/API check; no browser and no performance benchmark."""
from pathlib import Path
import json
import numpy as np
from isaac_field.env import make_env, SingleFieldAdapter
from isaac_field.task import Config
from isaac_field.viewer import attach_field_controls, _observation_batch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CHECKPOINT = ROOT / 'results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192/stage_04_turn90.zip'
env = make_env(Config(plants=16, terrain_enabled=False, max_time=20, damage_penalty=20,
                      goal_progress_budget=5), 'goto', 1, 80003, False,
               goal_distance=(1, 3), goal_angle=np.deg2rad(60), distance_cost=0.0, viewer_port=8769)
try:
    adapter = SingleFieldAdapter(env)
    adapter.reset(seed=82001)
    visualizer = next(v for v in env.sim.visualizers if v.cfg.visualizer_type == 'viser')
    controls = attach_field_controls(visualizer, adapter, str(CHECKPOINT), reset_seed=82001)
    panel = controls._network_panel
    assert panel is not None and panel.inspector is not None
    assert panel.inspector.snapshot is not None
    assert not panel.inspector.snapshot['errors'], panel.inspector.snapshot['errors']
    assert controls._runner.state is None  # automatic preview is non-advancing
    records = []
    for index in range(8):
        action = panel.action(_observation_batch(adapter.current_observation(), adapter.observation_space), adapter.task.time)
        transition = adapter.step(np.asarray(action).reshape(4))
        assert transition is not None
        controls.record_transition(*transition)
        panel.publish(force=True)
        snapshot = panel.inspector.snapshot
        assert not snapshot['errors'], snapshot['errors']
        assert all(layer['values'] is None or np.isfinite(layer['values']).all() for layer in snapshot['layers'])
        records.append(dict(frame=snapshot['sequence'], input_time=snapshot['sim_time'],
                            output_time=snapshot['output_time'], action=np.asarray(action).tolist(),
                            tracked_layers=sum(layer['values'] is not None for layer in snapshot['layers'])))
        if index == 0:
            panel.command('network_freeze', True, None, adapter.task.time)
            frozen = panel.display_snapshot['layers'][0]['values'].copy()
        if index == 2:
            np.testing.assert_array_equal(panel.display_snapshot['layers'][0]['values'], frozen)
            panel.command('network_freeze', False, None, adapter.task.time)
    panel.command('network_select', 'features_extractor.depth.0', None, adapter.task.time)
    assert panel.detail.value == '全部层与记忆门'
    panel.command('network_focus_layer', None, None, adapter.task.time)
    panel.command('network_layout', None, None, adapter.task.time)
    state = tuple(value.copy() for value in controls._runner.state)
    before_time = adapter.task.time
    panel.command('network_preview', None, _observation_batch(adapter.current_observation(), adapter.observation_space), before_time)
    for a, b in zip(state, controls._runner.state):
        np.testing.assert_array_equal(a, b)
    assert adapter.task.time == before_time
    panel.export()
    result = dict(status='passed', checkpoint=str(CHECKPOINT), browser_verified=False,
                  description='8 actual physics steps and actual Viser API calls; no browser UI operated',
                  frames=records, freeze_isolated=True, preview_preserves_actor_and_physics=True,
                  layers=[dict(key=l['key'], shape=list(l['values'].shape) if l['values'] is not None else None,
                               source=l['source']) for l in panel.inspector.layers.values()])
    controls._apply_task('goto', 3, 4, reset=True)
    assert panel.inspector.snapshot is None and panel.inspector.critic_state is None
    result['reset_clears_stale_activations'] = True
    with (OUT / 'native_network_probe_final.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('frames', 'layers')}, ensure_ascii=False), flush=True)
finally:
    env.close()
