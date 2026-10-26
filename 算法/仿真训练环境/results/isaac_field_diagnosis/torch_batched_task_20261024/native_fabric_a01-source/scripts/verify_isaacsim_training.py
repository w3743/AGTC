"""Bounded native RSL training/resume/evaluation/export check with frozen sources.

This records a plumbing check, never a full fast-stop acceptance score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new trial directory')
    parser.add_argument('--portable-root', type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite a trial: {output}')
    output.mkdir(parents=True)

    from scripts import train_isaac_foundation_fresh as fresh
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.runtime import create_env, launch_app, simulator_identity
    from isaac_field.task import Config

    task = Config(**{**asdict(Config()), **fresh.FAST_STOP_CONFIG,
        'size': 12.0, 'plants': 0, 'terrain_enabled': False, 'rock_density': 0.0,
        'max_time': 8.0, 'validate_scene': False, 'input_schema': ABSOLUTE_GOAL_INPUT_SCHEMA,
        'goto_reward_profile': 'fast_stop_v2', 'fast_stop_failure_remaining_time_cost': True})
    transfer_task = replace(task, max_time=10.0)
    manifest = fresh._source_manifest(output, copy_files=True)
    extra_sources = []
    for relative in ('scripts/verify_isaacsim_training.py', 'scripts/export_rsl_policy.py',
                     'scripts/install_isaaclab_windows.ps1', 'scripts/isaaclab_python.ps1',
                     '安装IsaacLab.cmd', '启动IsaacLab.cmd'):
        source, target = ROOT / relative, output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        extra_sources.append({'path': relative, 'snapshot_path': str(target.relative_to(output)), 'sha256': _sha(source)})
    protocol = {'schema': 'isaac-field-isaacsim-training-smoke-v1', **simulator_identity(),
        'purpose': 'bounded train/resume/inference/export plumbing; not control acceptance',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'isaaclab_commit': fresh._isaac_commit(),
        'packages': fresh._package_versions(), 'task_config': asdict(task),
        'front_guide_spec': asdict(FrontGuideSpec()), 'seed': 97051,
        'training': {'stages': 3, 'transitions_per_stage': 32, 'num_envs': 2,
            'num_steps_per_env': 16, 'batch_size': 16, 'epochs': 1, 'lstm_hidden_dim': 512,
            'gamma': 1.0, 'gae_lambda': 0.99, 'initialization': ['fresh', 'strict_resume', 'weights_transfer'],
            'expected_cumulative_timesteps': [32, 64, 32],
            'transfer_task_config': asdict(transfer_task),
            'registered_training_default_rollout_steps_unchanged': 128},
        'evaluation': {'seeds': [97061], 'max_steps': 2, 'bounded': True},
        'source_manifest': manifest, 'additional_sources': extra_sources}
    protocol_path = output / 'protocol.json'
    _json(protocol_path, protocol)
    protocol_sha = _sha(protocol_path)

    def verify_sources():
        fresh._verify_sources(output, manifest)
        assert _sha(protocol_path) == protocol_sha, 'protocol changed during check'
        for row in extra_sources:
            assert _sha(ROOT / row['path']) == row['sha256'], row['path']
            assert _sha(output / row['snapshot_path']) == row['sha256'], row['snapshot_path']

    kit_args = ''
    if args.portable_root is not None:
        portable = args.portable_root.resolve().as_posix()
        if any(character.isspace() for character in portable):
            parser.error('portable-root must contain no spaces')
        kit_args = f'--portable-root {portable}'
    result = {'passed': False, 'protocol_sha256': protocol_sha, 'stages': []}

    @contextmanager
    def record_failure_before_kit_shutdown():
        try:
            yield
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            _json(output / 'result.json', result)
            raise

    try:
        verify_sources()
        with launch_app(device='cuda:0', headless=True, kit_args=kit_args), record_failure_before_kit_shutdown():
            import torch
            from isaac_field.env import SingleFieldAdapter
            from isaac_field.learning import evaluate_model, train_stage
            from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload
            from scripts.export_rsl_policy import export

            parent = None
            for index in (1, 2, 3):
                verify_sources()
                stage_task = transfer_task if index == 3 else task
                env = create_env(cfg=stage_task, mode='goto', num_envs=2, seed=97051,
                    auto_reset=True, distance_cost=0.0, front_guide_spec=FrontGuideSpec(), rsl=True)
                try:
                    trained = train_stage(env, parent, 32, output / f'stage_{index:02}.pt',
                        device='cuda:0', n_steps=16, batch_size=16, n_epochs=1,
                        lstm_hidden_size=512, gamma=1.0, gae_lambda=0.99, seed=97051,
                        weights_transfer=(index == 3))
                    expected_total = 32 if index == 3 else index * 32
                    assert trained.trained_timesteps == 32
                    assert trained.total_timesteps == expected_total
                    parent = trained.checkpoint
                    payload = load_rsl_checkpoint_payload(parent, map_location='cpu')
                    assert payload['num_timesteps'] == expected_total
                    assert payload['field_policy_contract']['lstm']['hidden_dim'] == 512
                    optimizer_steps = {
                        str(key): float(value['step'])
                        for key, value in payload['optimizer_state_dict']['state'].items()
                        if 'step' in value
                    }
                    assert optimizer_steps, 'official PPO optimizer has no updates'
                    if index == 2:
                        prior_steps = result['stages'][0]['optimizer_steps']
                        assert set(prior_steps) <= set(optimizer_steps)
                        assert all(optimizer_steps[key] > count for key, count in prior_steps.items())
                    if index == 3:
                        assert max(optimizer_steps.values()) < max(result['stages'][1]['optimizer_steps'].values())
                    result['stages'].append({'checkpoint': parent.name, 'sha256': _sha(parent),
                        'trained_timesteps': trained.trained_timesteps, 'total_timesteps': trained.total_timesteps,
                        'optimizer_steps': optimizer_steps})
                    if index == 3:
                        initialization = payload['training_initialization']
                        assert initialization['optimizer'] == 'reinitialized'
                        assert initialization['timestep_counter'] == 'reset_to_zero'
                        assert initialization['parent_checkpoint_sha256'] == result['stages'][1]['sha256']
                        result['weights_transfer'] = initialization
                    _json(output / 'result.json', result)
                finally:
                    env.close()

            verify_sources()
            engine = create_env(cfg=transfer_task, mode='goto', num_envs=1, seed=97061,
                auto_reset=False, distance_cost=0.0, front_guide_spec=FrontGuideSpec(), rsl=True)
            adapter = SingleFieldAdapter(engine)
            try:
                evaluation = evaluate_model(adapter, parent, seeds=[97061], max_steps=2,
                    device='cpu', deterministic=True, record_trajectories=True,
                    record_initial_conditions=True, input_schema=transfer_task.input_schema)
                _json(output / 'evaluation.json', evaluation)
            finally:
                adapter.close()
            result['export'] = export(parent, output / 'field.ts.pt')
            for stage in result['stages']:
                assert _sha(output / stage['checkpoint']) == stage['sha256'], stage['checkpoint']
            assert all(torch.isfinite(value).all() for key in ('actor_state_dict', 'critic_state_dict')
                       for value in payload[key].values() if isinstance(value, torch.Tensor))
            verify_sources()
            result.update(passed=True, **simulator_identity(), evaluation_steps_cap=2)
            _json(output / 'result.json', result)
    except BaseException as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        _json(output / 'result.json', result)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
