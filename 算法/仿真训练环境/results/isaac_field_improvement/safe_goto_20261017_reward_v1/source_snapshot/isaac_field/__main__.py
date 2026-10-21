"""One Isaac Lab backend, with task-specific train/evaluate/view commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description='Isaac Lab agricultural robot')
    parser.add_argument('command',choices=['viewer','smoke','train','evaluate','adversary'])
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--plants',type=int,default=400)
    parser.add_argument('--envs',type=int,default=4)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--teacher-steps',type=int,default=16)
    parser.add_argument('--mode',choices=['goto','search','mixed'],default='mixed')
    parser.add_argument('--flat',action='store_true')
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--max-time',type=float,default=30.)
    parser.add_argument('--checkpoint')
    parser.add_argument('--policy-device',default='cpu')
    parser.add_argument('--output',default='results/isaac_field/run')
    parser.add_argument('--goal-distance',type=float,nargs=2)
    parser.add_argument('--goal-angle',type=float,default=180.,help='Goal bearing range in degrees')
    args=parser.parse_args()
    from .task import Config
    from .env import make_env,SingleFieldAdapter
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg=Config(plants=args.plants,terrain_enabled=not args.flat,max_time=args.max_time)
    single=args.command in ('viewer','evaluate','adversary')
    env=make_env(cfg,args.mode,1 if single else args.envs,args.seed,not single,
                 goal_distance=tuple(args.goal_distance) if args.goal_distance else None,
                 goal_angle=np.deg2rad(args.goal_angle),viewer_port=args.port if args.command=='viewer' else None)
    try:
        if args.command=='viewer':
            from .viewer import run_viewer
            adapter=SingleFieldAdapter(env)
            adapter.reset(seed=args.seed)
            visualizer=next(v for v in env.sim.visualizers if v.cfg.visualizer_type=='viser')
            print(f'Official Isaac Lab Viser: http://127.0.0.1:{args.port}/',flush=True)
            run_viewer(adapter,visualizer,args.checkpoint)
        elif args.command=='smoke':
            import torch
            obs,_=env.reset(seed=args.seed)
            initial=[t.robot.copy() for t in env.tasks]
            layout=[t.scene.xy.copy() for t in env.tasks]
            for k in range(args.steps):
                action=torch.zeros((env.num_envs,4),device=env.device)
                action[:,0]=.2
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
