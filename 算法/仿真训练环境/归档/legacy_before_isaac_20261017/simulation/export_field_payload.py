"""Materialize one seeded field_sim scene/state pair for the Gazebo exporter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from field_sim.env import Config, FieldEnv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = FieldEnv(Config(), "search")
    try:
        env.reset(seed=args.seed)
        (args.output_dir / "scene.json").write_text(
            json.dumps(env.scene_payload(), separators=(",", ":")), encoding="utf-8")
        (args.output_dir / "state.json").write_text(
            json.dumps(env.state_payload(), separators=(",", ":")), encoding="utf-8")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
