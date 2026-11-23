"""Compare exported Circuit JSON with the independently generated FOC reference."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[2] / "检查/联合仿真/code_as_eda"))
from reference import verify_cli

if __name__ == "__main__":
    verify_cli(PROJECT, 165)
