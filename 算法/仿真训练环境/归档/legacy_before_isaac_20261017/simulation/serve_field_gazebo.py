"""Serve the existing field browser against the Gazebo Gym backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from field_sim.env import Config
from field_sim.server import serve


def latest_checkpoint(root):
    """Select only a successfully verified Gazebo run, never a proxy-backend run."""
    candidates = []
    for folder in ('gazebo_field_training', 'gazebo_training'):
        for report in (Path(root) / 'results' / folder).glob('**/train_summary.json'):
            try:
                summary = json.loads(report.read_text(encoding='utf8'))
                checkpoint = report.parent / 'checkpoint.zip'
                if (summary.get('backend') == 'gazebo' and
                        summary.get('checkpoint_reload_passed') and checkpoint.is_file()):
                    candidates.append(checkpoint)
            except (OSError, ValueError):
                continue
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--mode', choices=('search', 'goto'), default='goto')
    parser.add_argument('--size', type=float, default=100)
    parser.add_argument('--plants', type=int, default=400)
    parser.add_argument('--max-time', type=float, default=20, help='episode budget in simulated seconds; matches short training by default')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manual', action='store_true', help='start manual debugging without loading a checkpoint')
    parser.add_argument('--output', type=Path, default=Path('results/gazebo_web'))
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024..65535')
    if not args.checkpoint and not args.manual:
        args.checkpoint = latest_checkpoint(Path(__file__).resolve().parents[1])
        if args.checkpoint is None:
            parser.error('No verified Gazebo checkpoint found. Run short training first, pass --checkpoint, or use --manual for debugging.')
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f'checkpoint not found: {args.checkpoint}')
    if args.checkpoint:
        print(f'Evaluating saved policy: {args.checkpoint.resolve()}', flush=True)
        # Load ML libraries before Transport exposes Ubuntu's ROS packages;
        # otherwise SB3's optional cv2 import can pick an incompatible system ABI.
        from field_sim import policy  # noqa: F401
    from simulation.gazebo_env import GazeboFieldEnv
    cfg = Config(size=args.size, plants=args.plants, max_time=args.max_time)

    class EvaluationFieldEnv(GazeboFieldEnv):
        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            if self.mode == 'goto' and 'goal' not in (options or {}):
                from simulation.evaluate_field_gazebo import _select_safe_goto_goal
                goal, _ = _select_safe_goto_goal(self)
                self.set_goal(goal)
                observation, info = self._observation(), self._info()
            return observation, info

    def factory(config, mode):
        kind = EvaluationFieldEnv if args.checkpoint else GazeboFieldEnv
        return kind(config, mode, output_dir=args.output.resolve(), gui=False)

    serve(cfg, port=args.port, seed=args.seed, mode=args.mode,
          checkpoint=args.checkpoint, env_factory=factory, backend='gazebo',
          initially_paused=True, initial_controller='policy' if args.checkpoint else 'manual')


if __name__ == '__main__':
    main()
