"""One Isaac Lab backend, with task-specific train/evaluate/view commands."""
import argparse
import importlib
import sys
from dataclasses import asdict
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from .feedback import INPUT_SCHEMA, SUPPORTED_INPUT_SCHEMAS


RSL_TASK = "Isaac-Field-FastStop-Rsl-Direct"


def _checkpoint_argument(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token.startswith("--checkpoint="):
            return token.split("=", 1)[1]
        if token == "--checkpoint" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _reject_zip_for_rsl(argv: list[str], *, command: str) -> None:
    checkpoint = _checkpoint_argument(argv)
    if checkpoint and Path(checkpoint).suffix.lower() != ".pt":
        raise SystemExit(
            f"{command} uses the official RSL-RL backend and requires a native .pt checkpoint; "
            "legacy SB3 .zip checkpoints are replay/evaluation artifacts only"
        )


def _run_official_rsl_cli(arguments: list[str]) -> int:
    from isaaclab_rl.entrypoints import run_train_cli

    return run_train_cli(["--rl_library", "rsl_rl", "--task", RSL_TASK, *arguments])


def _run_official_rsl_resume(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="isaac_field resume", add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parsed, remaining = parser.parse_known_args(arguments)
    if Path(parsed.checkpoint).suffix.lower() != ".pt":
        parser.error("resume requires a native RSL-RL .pt checkpoint; SB3 .zip is legacy-only")
    return _run_official_rsl_cli(["--checkpoint", parsed.checkpoint, *remaining])


def _forward_checkpoint_viewer(args) -> int:
    checkpoint = Path(args.checkpoint).expanduser()
    extension = checkpoint.suffix.lower()
    if args.legacy_sb3:
        if extension != ".zip":
            raise SystemExit("--legacy-sb3 viewer mode requires an SB3 .zip checkpoint")
    elif extension != ".pt":
        raise SystemExit("viewer requires an RSL .pt checkpoint; pass --legacy-sb3 explicitly for an SB3 .zip")
    view_fast_stop_goal = importlib.import_module("scripts.view_fast_stop_goal")

    forwarded = ["--checkpoint", str(checkpoint)]
    if args.legacy_sb3:
        forwarded.append("--legacy-sb3")
    if args.launch:
        forwarded.append("--launch")
    if args.output is not None:
        forwarded.extend(("--output", str(args.output)))
    if args.port is not None:
        forwarded.extend(("--port", str(args.port)))
    return view_fast_stop_goal.main(forwarded)


def _run_rsl_evaluation(args, parser) -> int:
    if not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    extension = Path(args.checkpoint).suffix.lower()
    if extension == ".zip":
        if not args.legacy_sb3:
            parser.error("SB3 .zip evaluation requires --legacy-sb3")
        parser.error("SB3 .zip evaluation must be routed through the legacy evaluation path")
    if args.legacy_sb3:
        parser.error("--legacy-sb3 is only valid with an SB3 .zip checkpoint")
    if extension != ".pt":
        parser.error("new RSL evaluation requires an .pt checkpoint; SB3 .zip requires --legacy-sb3")
    evaluate_fast_stop_mastery = importlib.import_module("scripts.evaluate_fast_stop_mastery")

    if args.smoke and args.max_steps is None:
        parser.error("RSL smoke requires --max-steps 1 or 2")
    if args.smoke and args.scenario_count not in (None, 1, 2):
        parser.error("RSL smoke supports at most two scenarios")

    output = Path(args.output or "results/isaac_field/run").resolve()
    report_path = output / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {report_path}")
    output.mkdir(parents=True, exist_ok=True)
    forwarded = ["--checkpoint", str(Path(args.checkpoint).expanduser())]
    if args.smoke:
        forwarded.append("--smoke")
    if args.max_steps is not None:
        forwarded.extend(("--max-steps", str(args.max_steps)))
    if args.scenario_count is not None:
        forwarded.extend(("--scenario-count", str(args.scenario_count)))
    if args.seed is not None:
        forwarded.extend(("--seed", str(args.seed)))
    forwarded.extend(("--device", args.policy_device, "--output", str(report_path)))
    return evaluate_fast_stop_mastery.main(forwarded)


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
    # Current training and resume use Isaac Lab's official RSL-RL CLI. SB3
    # remains reachable only through commands that name the legacy backend.
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        _reject_zip_for_rsl(sys.argv[2:], command="train")
        return _run_official_rsl_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        return _run_official_rsl_resume(sys.argv[2:])
    parser=argparse.ArgumentParser(description='Isaac Lab agricultural robot')
    parser.add_argument('command',choices=['viewer','smoke','train','resume','train-sb3',
                                           'evaluate','adversary','adversary-sb3'])
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--episode-seed',type=int,default=None,
                        help='Viewer reset seed, independent of --seed; explicit values repeat the same episode on Reset')
    parser.add_argument('--reward-profile',choices=['legacy','protect_plants'],default='legacy',
                        help='Task reward settings; protect_plants uses damage_penalty=20 and goal_progress_budget=5')
    parser.add_argument('--plants',type=int,default=400)
    parser.add_argument('--rock-density',type=float,default=0.01,
                        help='Rock density in rocks/m² (default: 0.01)')
    parser.add_argument('--front-guides',action='store_true',
                        help='Add paired front-guide wheels coupled to their tracks (no independent actuators)')
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
    parser.add_argument('--legacy-sb3',action='store_true',
                        help='explicitly enable an SB3 .zip viewer/evaluation path')
    parser.add_argument('--weights-transfer', action='store_true',
                        help='explicitly initialize a new RSL trial from compatible checkpoint weights')
    parser.add_argument('--launch',action='store_true',
                        help='launch a checkpoint viewer after its preflight check')
    parser.add_argument('--smoke', action='store_true',
                        help='run a bounded RSL native smoke check (not a competence evaluation)')
    parser.add_argument('--max-steps',type=int,default=None,
                        help='optional step cap for RSL evaluation; RSL smoke is limited to 1 or 2')
    parser.add_argument('--scenario-count',type=int,default=None,
                        help='RSL evaluation scenario count; smoke is limited to 1 or 2')
    parser.add_argument('--input-schema', choices=SUPPORTED_INPUT_SCHEMAS, default=INPUT_SCHEMA,
                        help='Versioned state encoding; checkpoint metadata must match')
    parser.add_argument('--policy-device',default='cpu')
    parser.add_argument('--output',default=None)
    parser.add_argument('--goal-distance',type=float,nargs=2)
    parser.add_argument('--goal-angle',type=float,default=180.,help='Goal bearing range in degrees')
    args=parser.parse_args()
    if args.command in ('train-sb3', 'adversary-sb3'):
        parser.error(
            "new training is available only through the official Isaac Lab RSL-RL runner; "
            "SB3 .zip artifacts remain legacy-only for explicitly requested replay/evaluation"
        )
    if args.weights_transfer and args.command != 'adversary':
        parser.error('--weights-transfer is only valid for adversary checkpoint initialization')
    if args.command == 'viewer' and args.checkpoint:
        extension = Path(args.checkpoint).suffix.lower()
        if extension == '.pt':
            return _forward_checkpoint_viewer(args)
        if extension != '.zip' or not args.legacy_sb3:
            return _forward_checkpoint_viewer(args)
    if args.command == 'viewer' and args.legacy_sb3 and not args.checkpoint:
        parser.error('viewer --legacy-sb3 requires --checkpoint <SB3 .zip>')
    if args.command == 'evaluate':
        extension = Path(args.checkpoint or '').suffix.lower()
        if extension == '.pt':
            return _run_rsl_evaluation(args, parser)
        if extension == '.zip' and args.legacy_sb3:
            if args.smoke:
                parser.error('--smoke is only available for RSL .pt checkpoints')
            if args.max_steps is not None or args.scenario_count is not None:
                parser.error('--max-steps/--scenario-count are only available for RSL checkpoint evaluation')
        else:
            _run_rsl_evaluation(args, parser)
    elif args.legacy_sb3 and not (
            args.command == 'viewer' and args.checkpoint
            and Path(args.checkpoint).suffix.lower() == '.zip'):
        parser.error('--legacy-sb3 is only valid with viewer/evaluate checkpoint commands')
    if args.launch and args.command != 'viewer':
        parser.error('--launch is only valid with viewer')
    if args.smoke and args.command != 'evaluate':
        parser.error('--smoke is only valid with evaluate --checkpoint <RSL .pt>')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.scenario_count is not None and args.scenario_count < 1:
        parser.error('--scenario-count must be positive')
    if args.seed is None:
        args.seed = 7
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
    if args.input_schema != INPUT_SCHEMA:
        config_kwargs['input_schema'] = args.input_schema
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
    if args.command == 'adversary':
        if not args.checkpoint:
            parser.error('adversary requires --checkpoint <Isaac Sim PhysX RSL .pt student policy>')
        if Path(args.checkpoint).suffix.lower() != '.pt':
            parser.error('adversary accepts only an RSL .pt student checkpoint; SB3 .zip is legacy-only')
        from .runtime import create_env, launch_app
        from .adversarial_training import run_round

        with launch_app(device='cuda:0', headless=True):
            env = create_env(
                cfg=cfg,
                mode=args.mode,
                num_envs=1,
                seed=args.seed,
                auto_reset=False,
                goal_distance=tuple(args.goal_distance) if args.goal_distance else None,
                goal_angle=np.deg2rad(args.goal_angle),
                distance_cost=0.0,
                front_guide_spec=front_guide_spec,
                body_stability_config=body_stability_config,
                rsl=True,
            )
            try:
                result = run_round(
                    env,
                    args.checkpoint,
                    output,
                    teacher_steps=args.teacher_steps,
                    student_steps=args.steps,
                    seed=args.seed,
                    goal_distance=tuple(args.goal_distance) if args.goal_distance else (1.0, 3.0),
                    goal_angle=np.deg2rad(args.goal_angle),
                    student_device=args.policy_device,
                    weights_transfer=args.weights_transfer,
                )
            finally:
                env.close()
        print(json.dumps(_json_safe(result), indent=2), flush=True)
        return 0
    single=args.command in ('viewer','evaluate','adversary-sb3')
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
        elif args.command=='train-sb3':
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
                goal_angle=np.deg2rad(args.goal_angle),mode=args.mode,device=args.policy_device,
                legacy_sb3=bool(args.legacy_sb3))
            (output/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2))
        else:
            from .adversarial_training import run_round
            result=run_round(env,args.checkpoint,output,teacher_steps=args.teacher_steps,
                student_steps=args.steps,seed=args.seed,
                goal_distance=tuple(args.goal_distance) if args.goal_distance else (1.,3.),
                goal_angle=np.deg2rad(args.goal_angle),student_device=args.policy_device)
            print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=='__main__':raise SystemExit(main())
