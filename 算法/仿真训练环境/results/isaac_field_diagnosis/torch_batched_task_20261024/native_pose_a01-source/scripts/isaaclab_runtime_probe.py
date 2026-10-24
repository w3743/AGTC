"""Run official Cartpole on Isaac Sim PhysX; require a new result path."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from scripts.verify_isaacsim_environment import main
    raise SystemExit(main(["--upstream", *sys.argv[1:]]))
