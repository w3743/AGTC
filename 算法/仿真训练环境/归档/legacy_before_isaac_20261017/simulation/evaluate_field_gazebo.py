"""Run held-out, finite-duration search/goto episodes with saved policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from field_sim.env import Config, FieldEnv
from simulation.gazebo_env import GazeboFieldEnv


DEFAULT_HELDOUT_SEEDS = (10007, 10008, 10009)


def _line_clearance(env, start, end):
    """Conservative circular-footprint clearance along a straight local route."""
    scene = env.scene
    obstacle_points = []
    obstacle_radii = []
    if len(scene.xy):
        obstacle_points.append(np.asarray(scene.xy, dtype=np.float64))
        obstacle_radii.append(np.asarray(scene.radius, dtype=np.float64))
    terrain = getattr(scene, "terrain", None)
    if terrain is not None and len(getattr(terrain, "rock_xy", ())):
        # The evaluation route avoids even small traversable rocks so the short
        # goto is a clean arrival check rather than a rough-terrain trial.
        obstacle_points.append(np.asarray(terrain.rock_xy, dtype=np.float64))
        obstacle_radii.append(np.asarray(terrain.rock_radius, dtype=np.float64))
    if not obstacle_points:
        return math.inf

    points = np.concatenate(obstacle_points, axis=0)
    radii = np.concatenate(obstacle_radii, axis=0)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        distance = np.linalg.norm(points - start, axis=1)
    else:
        fraction = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
        nearest = start[None, :] + fraction[:, None] * delta[None, :]
        distance = np.linalg.norm(points - nearest, axis=1)
    vehicle_radius = math.hypot(env.cfg.robot_length, env.cfg.robot_width) / 2 + 0.05
    return float(np.min(distance - radii - vehicle_radius))


def _select_safe_goto_goal(env, nominal_distance=3.0):
    """Pick a deterministic clear local goal, preferring the nominal 3 m range."""
    start = np.asarray(env.robot[:2], dtype=np.float64)
    yaw = float(env.robot[2])
    angle_offsets = [0.0]
    for degrees in range(15, 181, 15):
        angle_offsets.extend((math.radians(degrees), -math.radians(degrees)))

    distances = tuple(dict.fromkeys((float(nominal_distance), 2.5, 2.0, 1.5)))
    for distance in distances:
        candidates = []
        for order, offset in enumerate(angle_offsets):
            angle = yaw + offset
            candidate = start + distance * np.array([math.cos(angle), math.sin(angle)])
            if not env.scene.navigable(candidate):
                continue
            clearance = _line_clearance(env, start, candidate)
            if clearance >= 0.05:
                # Keep steering changes modest once the route has safe margin;
                # the best-clearance route can require an unnecessary U-turn.
                candidates.append((-abs(offset), clearance, -order, candidate, offset))
        if candidates:
            _, clearance, _, goal, offset = max(candidates, key=lambda row: row[:3])
            return goal, {
                "requested_distance_m": float(nominal_distance),
                "selected_distance_m": float(np.linalg.norm(goal - start)),
                "corridor_clearance_m": None if not math.isfinite(clearance) else float(clearance),
                "heading_offset_rad": float(offset),
                "fallback_distance_used": distance != float(nominal_distance),
            }
    raise RuntimeError("held-out scene has no obstacle-clear local goto route")


def _scene_identity(env):
    scene_id = getattr(env, "scene_id", None)
    payload_fn = getattr(env, "scene_payload", None)
    if not callable(payload_fn):
        return {"scene_id": scene_id, "sha256": None}
    payload = dict(payload_fn())
    # Scene IDs are episode counters; hash geometry/config only so the digest
    # can confirm that independently reset modes reused the same held-out map.
    payload.pop("scene_id", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"scene_id": scene_id,
            "sha256": hashlib.sha256(serialized).hexdigest()}


def _make_fieldsim_env(cfg, mode, *, output_dir):
    """Match the Gazebo factory signature without creating runtime artifacts."""
    return FieldEnv(cfg, mode)


def evaluate(checkpoint, output, *, seeds=DEFAULT_HELDOUT_SEEDS, seconds=10.0,
             monitor=None, env_factory=None, runner_factory=None,
             backend="gazebo", mode="mixed", episode_budget=20.0):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("held-out seeds must be a non-empty sequence without duplicates")
    if backend not in ("gazebo", "fieldsim"):
        raise ValueError("backend must be gazebo or fieldsim")
    if mode not in ("mixed", "search", "goto"):
        raise ValueError("mode must be mixed, search, or goto")
    if not math.isfinite(float(seconds)) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(float(episode_budget)) or episode_budget <= 0:
        raise ValueError("episode_budget must be finite and positive")
    if runner_factory is None:
        from field_sim.policy import PolicyRunner
        runner_factory = PolicyRunner
    if env_factory is None:
        env_factory = GazeboFieldEnv if backend == "gazebo" else _make_fieldsim_env
    output = Path(output)
    cfg = Config(max_time=float(episode_budget))
    max_steps = int(math.floor(float(seconds) / cfg.dt + 1e-9))
    if max_steps < 1:
        raise ValueError(f"seconds must allow at least one decision step ({cfg.dt:g} s)")
    evaluated_seconds = max_steps * cfg.dt
    task_modes = ("search", "goto") if mode == "mixed" else (mode,)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    started = time.perf_counter()
    for seed in seeds:
        hashes_by_mode = {}
        for task_mode in task_modes:
            env = env_factory(cfg, task_mode, output_dir=output / f'{task_mode}_{seed}')
            try:
                obs, info = env.reset(seed=seed)
                scene_identity = _scene_identity(env)
                hashes_by_mode[task_mode] = scene_identity["sha256"]
                goal_route = None
                if task_mode == 'goto':
                    # Keep the full held-out field, but test a stable local
                    # arrival on a clear corridor so it is not a crop-crushing
                    # or rock-contact stress test.
                    goal, goal_route = _select_safe_goto_goal(env)
                    env.set_goal(goal)
                    obs, info = env._observation(), env._info()
                runner = runner_factory(env, checkpoint)
                initial_distance = float(info['distance'])
                start_xy = env.robot[:2].copy()
                start_sim_time = float(info.get('sim_time_s', getattr(env, 'time', 0.0)))
                path_length = 0.0
                total_reward = 0.0
                rows = []
                ended = False
                terminal = False
                time_limit = False
                for step in range(max_steps):
                    before = env.robot[:2].copy()
                    action = runner.action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    path_length += float(np.linalg.norm(env.robot[:2]-before))
                    total_reward += float(reward)
                    rows.append({'time': env.time, 'xy': env.robot[:2].tolist(),
                                 'yaw': float(env.robot[2]), 'action': action.tolist(),
                                 'reward': float(reward), 'distance': info['distance']})
                    if monitor:
                        monitor.publish(env, phase='evaluation', steps=step+1,
                                        total=max_steps)
                    if terminated or truncated:
                        ended = True
                        terminal = bool(terminated)
                        time_limit = bool(truncated)
                        break
                sim_seconds = max(0.0, float(info.get('sim_time_s', env.time)) - start_sim_time)
                goal_success = task_mode == 'goto' and info.get('reason') == 'goal_reached'
                episode = {'mode': task_mode, 'seed': seed, 'steps': len(rows),
                           'split': 'heldout', 'scene_id': scene_identity['scene_id'],
                           'scene_sha256': scene_identity['sha256'],
                           'goal_route': goal_route,
                           'backend': backend,
                           'requested_seconds': float(seconds),
                           'episode_budget_seconds': float(episode_budget),
                           'sim_seconds': sim_seconds,
                           'environment_time_s': float(env.time), 'reward': total_reward,
                           'reason': info.get('reason'),
                           'goal_success': goal_success,
                           'arrival_time_s': sim_seconds if goal_success else None,
                           'search_success': task_mode == 'search' and info.get('reason') == 'all_targets_found_by_evaluator',
                           'episode_ended': ended,
                           'terminated': terminal,
                           'truncated': time_limit,
                           'rollout_limit_reached': not ended and len(rows) == max_steps,
                           'plants_seen_count': len(getattr(env, 'seen', ())),
                           'plants_reported_count': int(info.get('found', len(getattr(env, 'reported', ())))),
                           'target_plant_count': int(info.get('targets', np.asarray(
                               getattr(env.scene, 'targets', ())).sum())),
                           'false_report_count': int(info.get('false_reports', len(getattr(env, 'false_reports', ())))),
                           'crushed_plant_count': int(info.get('damage', len(getattr(env, 'damaged', ())))),
                           'rock_collision_count': int(info.get('collisions', getattr(env, 'collisions', 0))),
                           'start_goal_distance_m': initial_distance,
                           'initial_goal_distance_m': initial_distance,
                           'final_goal_distance_m': float(info['distance']),
                           'goal_progress_m': initial_distance-float(info['distance']),
                           'path_length_m': path_length,
                           'net_displacement_m': float(np.linalg.norm(env.robot[:2]-start_xy)),
                           **info}
                episodes.append(episode)
                destination = output / f'{task_mode}_{seed}'
                destination.mkdir(parents=True, exist_ok=True)
                (destination/'trajectory.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
                (destination/'final_state.json').write_text(json.dumps(env.state_payload(), allow_nan=False), encoding='utf8')
                np.savez(destination/'policy_observation.npz', **obs)
            finally:
                env.close()
        if (len(task_modes) == 2 and hashes_by_mode['search'] is not None
                and hashes_by_mode['goto'] is not None
                and hashes_by_mode['search'] != hashes_by_mode['goto']):
            raise RuntimeError(f"held-out seed {seed} did not reproduce the same field geometry in search and goto")
    goto_episodes = [episode for episode in episodes if episode['mode'] == 'goto']
    search_episodes = [episode for episode in episodes if episode['mode'] == 'search']
    mean = lambda values: float(np.mean(values)) if values else None
    successful_arrival_times = [e['arrival_time_s'] for e in goto_episodes if e['arrival_time_s'] is not None]
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint': str(Path(checkpoint).resolve()), 'config': asdict(cfg),
              'backend': backend, 'mode': mode,
              'requested_seconds_per_episode': float(seconds),
              'rollout_seconds_per_episode': float(evaluated_seconds),
              'episode_budget_seconds': float(episode_budget),
              'rollout_max_steps': max_steps,
              'heldout_seeds': list(seeds),
              'scope': f'Fixed-seed held-out {backend} episodes in mode={mode}; each rollout is capped at requested seconds while the observation time fraction uses episode_budget seconds. Search is a 100m field with a short time budget, while goto tests arrival on a short obstacle-clear local route (nominally 3 m), not obstacle-avoidance competence. Search seen/reported counts retain the environment geometry-based detector/report proxy; damage and rock collisions use field_sim proxy geometry. This is execution evidence, not a full-field competence benchmark.',
              'wall_seconds_including_resets': time.perf_counter()-started,
              'episodes': episodes,
              'successes': sum(bool(e['is_success']) for e in episodes),
              'episode_count': len(episodes),
              'goto_summary': {
                  'episodes': len(goto_episodes),
                  'arrivals': sum(bool(e['goal_success']) for e in goto_episodes),
                  'arrival_rate': (sum(bool(e['goal_success']) for e in goto_episodes) / len(goto_episodes)
                                  if goto_episodes else 0.0),
                  'mean_initial_distance_m': (float(np.mean([e['initial_goal_distance_m'] for e in goto_episodes]))
                                              if goto_episodes else None),
                  'mean_final_distance_m': (float(np.mean([e['final_goal_distance_m'] for e in goto_episodes]))
                                            if goto_episodes else None),
                  'mean_goal_progress_m': mean([e['goal_progress_m'] for e in goto_episodes]),
                  'mean_sim_seconds': mean([e['sim_seconds'] for e in goto_episodes]),
                  'mean_arrival_time_s': mean(successful_arrival_times),
                  'crushed_plant_count_total': sum(e['crushed_plant_count'] for e in goto_episodes),
                  'rock_collision_count_total': sum(e['rock_collision_count'] for e in goto_episodes),
              },
              'search_summary': {
                  'episodes': len(search_episodes),
                  'evaluator_successes': sum(bool(e['search_success']) for e in search_episodes),
                  'plants_seen_total': sum(e['plants_seen_count'] for e in search_episodes),
                  'plants_reported_total': sum(e['plants_reported_count'] for e in search_episodes),
                  'targets_total': sum(e['target_plant_count'] for e in search_episodes),
                  'false_reports_total': sum(e['false_report_count'] for e in search_episodes),
                  'mean_sim_seconds': mean([e['sim_seconds'] for e in search_episodes]),
                  'crushed_plant_count_total': sum(e['crushed_plant_count'] for e in search_episodes),
                  'rock_collision_count_total': sum(e['rock_collision_count'] for e in search_episodes),
                  'target_report_fraction': (sum(e['plants_reported_count'] for e in search_episodes) /
                                             max(1, sum(e['target_plant_count'] for e in search_episodes))),
              },
              'crushed_plant_count_total': sum(e['crushed_plant_count'] for e in episodes),
              'rock_collision_count_total': sum(e['rock_collision_count'] for e in episodes),
              'sim_seconds_total': sum(e['sim_seconds'] for e in episodes),
              'task_mastery_claim': False}
    (output/'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('gazebo', 'fieldsim'), default='gazebo')
    parser.add_argument('--mode', choices=('mixed', 'search', 'goto'), default='mixed')
    parser.add_argument('--heldout-seeds', '--seeds', dest='seeds', default=','.join(map(str, DEFAULT_HELDOUT_SEEDS)))
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--episode-budget', type=float, default=20.0,
                        help='episode horizon encoded in the state observation; independent of rollout seconds')
    parser.add_argument('--monitor-port', type=int)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    try:
        seeds = tuple(int(x.strip()) for x in args.seeds.split(',') if x.strip())
    except ValueError:
        parser.error('--heldout-seeds must be comma-separated integers')
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('--heldout-seeds must be non-empty and contain no duplicates')
    monitor = None
    try:
        if args.monitor_port:
            from simulation.gazebo_monitor import TrainingMonitor
            monitor = TrainingMonitor(args.monitor_port)
        result = evaluate(args.checkpoint, args.output,
                          seeds=seeds,
                          seconds=args.seconds, monitor=monitor,
                          backend=args.backend, mode=args.mode,
                          episode_budget=args.episode_budget)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        if monitor:
            monitor.close()


if __name__ == '__main__':
    main()
