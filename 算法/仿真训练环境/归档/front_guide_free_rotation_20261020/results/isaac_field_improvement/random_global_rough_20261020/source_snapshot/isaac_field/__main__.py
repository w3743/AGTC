"""One Isaac Lab backend, with task-specific train/evaluate/view commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _viewer_initial_state(adapter, cfg, *, env_seed, episode_seed, repeat_reset_seed,
                          mode, reward_profile, goal_distance, goal_angle, checkpoint):
    payload = adapter.state_payload()
    robot = payload.get("robot", {})
    gimbal = payload.get("gimbal", {})
    scene = getattr(getattr(adapter, "task", None), "scene", None)
    engine = getattr(adapter, "engine", None)
    return _json_safe({
        "env_seed": env_seed,
        "episode_seed": episode_seed,
        "repeat_reset_seed": repeat_reset_seed,
        "reward_profile": reward_profile,
        "task_config": asdict(cfg),
        "environment": {
            "mode": mode,
            "initial_task_mode": payload.get("mode"),
            "goal_distance": goal_distance,
            "goal_angle_rad": goal_angle,
            "distance_cost": 0.0,
            "auto_reset": False,
        },
        "robot": {key: robot.get(key) for key in ("x", "y", "z", "yaw", "v", "w")},
        "goal": payload.get("goal"),
        "gimbal": {key: gimbal.get(key) for key in ("yaw", "pitch")},
        "plant_xy": getattr(scene, "xy", None),
        "drive_config": getattr(engine, "drive_config", None),
        "observation_config": getattr(engine, "observation_config", None),
        "action_config": getattr(engine, "action_config", None),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
    })


def main():
    parser=argparse.ArgumentParser(description='Isaac Lab agricultural robot')
    parser.add_argument('command',choices=['viewer','smoke','train','evaluate','adversary'])
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--episode-seed',type=int,default=None,
                        help='Viewer reset seed, independent of --seed; explicit values repeat the same episode on Reset')
    parser.add_argument('--reward-profile',choices=['legacy','protect_plants'],default='legacy',
                        help='Task reward settings; protect_plants uses damage_penalty=20 and goal_progress_budget=5')
    parser.add_argument('--plants',type=int,default=400)
    parser.add_argument('--rock-density',type=float,default=0.01,
                        help='Rock density in rocks/m² (default: 0.01)')
    parser.add_argument('--front-guides',action='store_true',
                        help='Add the paired passive front-guide wheel proxy')
    parser.add_argument('--body-stability',action='store_true',
                        help='Enable the body-attitude stability reward term')
    parser.add_argument('--envs',type=int,default=4)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--teacher-steps',type=int,default=16)
    parser.add_argument('--mode',choices=['goto','search','mixed'],default='mixed')
    parser.add_argument('--flat',action='store_true')
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--max-time',type=float,default=30.)
    parser.add_argument('--checkpoint')
    parser.add_argument('--policy-device',default='cpu')
    parser.add_argument('--output',default=None)
    parser.add_argument('--goal-distance',type=float,nargs=2)
    parser.add_argument('--goal-angle',type=float,default=180.,help='Goal bearing range in degrees')
    args=parser.parse_args()
    from .task import Config
    from .env import make_env,SingleFieldAdapter
    output=Path(args.output or 'results/isaac_field/run').resolve()
    if args.command != 'viewer' or args.output is not None:
        output.mkdir(parents=True,exist_ok=True)
    reward_settings = ({'damage_penalty': 20.0, 'goal_progress_budget': 5.0}
                       if args.reward_profile == 'protect_plants'
                       else {'damage_penalty': 4.0, 'goal_progress_budget': None})
    config_kwargs = dict(plants=args.plants, terrain_enabled=not args.flat,
                         max_time=args.max_time, **reward_settings)
    # Leave Config's established 0.01 default intact unless the CLI overrides it.
    if args.rock_density != 0.01:
        config_kwargs['rock_density'] = args.rock_density
    cfg=Config(**config_kwargs)
    front_guide_spec = None
    if args.front_guides:
        from .assets import FrontGuideSpec
        front_guide_spec = FrontGuideSpec()
    body_stability_config = None
    if args.body_stability:
        from .body_stability import BodyStabilityConfig
        body_stability_config = BodyStabilityConfig()
    single=args.command in ('viewer','evaluate','adversary')
    env=make_env(cfg,args.mode,1 if single else args.envs,args.seed,not single,
                 goal_distance=tuple(args.goal_distance) if args.goal_distance else None,
                 goal_angle=np.deg2rad(args.goal_angle),distance_cost=0.0,
                 viewer_port=args.port if args.command=='viewer' else None,
                 front_guide_spec=front_guide_spec,
                 body_stability_config=body_stability_config)
    try:
        if args.command=='viewer':
            from .viewer import run_viewer
            adapter=SingleFieldAdapter(env)
            episode_seed=args.episode_seed if args.episode_seed is not None else args.seed
            adapter.reset(seed=episode_seed)
            initial_state=_viewer_initial_state(
                adapter,cfg,env_seed=args.seed,episode_seed=episode_seed,
                repeat_reset_seed=args.episode_seed,mode=args.mode,reward_profile=args.reward_profile,
                goal_distance=tuple(args.goal_distance) if args.goal_distance else None,
                goal_angle=float(np.deg2rad(args.goal_angle)),checkpoint=args.checkpoint)
            initial_json=json.dumps(initial_state,indent=2)
            if args.output is not None:
                initial_path=output/'viewer_initial_state.json'
                if initial_path.exists():
                    raise FileExistsError(f'refusing to overwrite viewer replay record: {initial_path}')
                initial_path.write_text(initial_json,encoding='utf-8')
            print(initial_json,flush=True)
            visualizer=next(v for v in env.sim.visualizers if v.cfg.visualizer_type=='viser')
            print(f'Official Isaac Lab Viser: http://127.0.0.1:{args.port}/',flush=True)
            run_viewer(adapter,visualizer,args.checkpoint,reset_seed=args.episode_seed)
        elif args.command=='smoke':
            import torch
            obs,_=env.reset(seed=args.seed)
            initial=[t.robot.copy() for t in env.tasks]
            layout=[t.scene.xy.copy() for t in env.tasks]
            for k in range(args.steps):
                action=torch.zeros((env.num_envs,4),device=env.device)
                action[:,:2]=.2
                action[:,2]=.1
                action[:,3]=-.1 if k<args.steps//2 else .1
                obs,reward,term,trunc,_=env.step(action)
                assert all(torch.isfinite(v).all() for v in obs['policy'].values())
                assert torch.isfinite(reward).all()
            task=env.tasks[0]
            result={'backend':env.sim.physics_manager.__name__,'device':env.device,
                    'num_envs':env.num_envs,'steps':args.steps,
                    'wheels':len(env._wheel_names),'joints':env.rover.joint_names,
                    'observations':{k:list(v.shape) for k,v in obs['policy'].items()},
                    'depth_valid_pixels':int(task.frames['depth']['valid'].sum()),
                    'lidar_valid_points':int(task.frames['lidar']['valid'].sum()),
                    'sensor_sequences':task._info()['sensor_sequences'],
                    'initial_pose':initial[0].tolist(),'final_pose':task.robot.tolist(),
                    'task_metrics':task._info(), 'episode_results':env.episode_results,
                    'validation_scope':'physical step/sensor/training interface; no throughput benchmark'}
            env.reset(seed=args.seed+1)
            result['reset_changes_layout']=bool(not np.array_equal(layout[0],env.tasks[0].scene.xy)) if cfg.plants else None
            result['reset_changes_spawn']=bool(not np.allclose(initial[0][:3],env.tasks[0].robot[:3]))
            (output/'smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2))
        elif args.command=='train':
            from .learning import train_stage
            result=train_stage(env,args.checkpoint,args.steps,output/'checkpoint.zip',device=args.policy_device)
            summary={k:v for k,v in vars(result).items() if k!='model'}
            summary['checkpoint']=str(summary['checkpoint'])
            summary.update(task_config=asdict(cfg),mode=args.mode,episodes=env.episode_results,
                           claim='Training execution is verified separately from task competence.')
            (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(json.dumps(summary,indent=2))
        elif args.command=='evaluate':
            if not args.checkpoint:parser.error('evaluate requires --checkpoint')
            from .learning import evaluate_model
            result=evaluate_model(SingleFieldAdapter(env),args.checkpoint,list(range(100007,100017)),
                goal_distance=tuple(args.goal_distance) if args.goal_distance else None,
                goal_angle=np.deg2rad(args.goal_angle),mode=args.mode,device=args.policy_device)
            (output/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2))
        else:
            if not args.checkpoint:parser.error('adversary requires --checkpoint')
            from .adversarial_training import run_round
            result=run_round(env,args.checkpoint,output,teacher_steps=args.teacher_steps,
                student_steps=args.steps,seed=args.seed,
                goal_distance=tuple(args.goal_distance) if args.goal_distance else (1.,3.),
                goal_angle=np.deg2rad(args.goal_angle),student_device=args.policy_device)
            print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=='__main__':main()
