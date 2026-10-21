"""Bounded real-engine probe for the paused stepping bridge."""
from __future__ import annotations

import json
import time
from pathlib import Path

from simulation.gazebo_driver import GazeboDriver


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/gazebo_training/driver_probe'
    out.mkdir(parents=True, exist_ok=True)
    with GazeboDriver(root / 'results/gazebo_field/benchmark/world.sdf', out) as driver:
        begin = time.perf_counter()
        snapshot = driver.warmup()
        print('WARMUP', snapshot['sim_time'], snapshot['pose'], flush=True)
        rows = []
        for index in range(12):
            tick = time.perf_counter()
            snapshot = driver.step(.3, .1, index * .02, index * .01)
            rows.append({'sim_time': snapshot['sim_time'], 'wall_ms': (time.perf_counter()-tick)*1000,
                         'pose': snapshot['pose'], 'gimbal': snapshot['gimbal']})
            print('STEP', index, rows[-1], flush=True)
        result = {'wall_seconds': time.perf_counter()-begin, 'steps': rows}
        (out / 'probe.json').write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
