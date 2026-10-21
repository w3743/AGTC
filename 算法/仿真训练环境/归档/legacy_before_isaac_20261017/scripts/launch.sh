#!/usr/bin/env bash
set -e
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo '尚未安装环境，请先双击 安装环境.cmd。'; exit 1
fi
source "$(dirname "$0")/env.sh"
cd "$LAB_ROOT"
if [[ ! -f simulation/lab.sdf ]]; then /usr/bin/python3 scripts/generate_world.py; fi
case "${1:-mapping}" in
  mapping) exec /usr/bin/python3 simulation/run_lab.py --gui --seconds 180 ;;
  manual) exec /usr/bin/python3 simulation/run_lab.py --gui --manual --seconds 3600 ;;
  teleop) exec ros2 run teleop_twist_keyboard teleop_twist_keyboard ;;
  lightstereo) exec /usr/bin/python3 -B simulation/run_lightstereo.py ;;
  lightstereo-study) exec /usr/bin/python3 -B simulation/run_lightstereo.py --study ;;
  lightstereo-extend) exec /usr/bin/python3 -B simulation/run_lightstereo.py --extend ;;
  lightstereo-refine-capture) exec /usr/bin/python3 -B simulation/run_lightstereo.py --refine-capture ;;
  lightstereo-appearance-capture) exec /usr/bin/python3 -B simulation/run_lightstereo.py --appearance-capture ;;
  drone) exec "$ROBOTICS_HOME/venv/px4/bin/python" simulation/run_drone.py --gui --takeoff --seconds 180 ;;
  arm-view|hand-view|arm-train|hand-train)
    task="${1%%-*}"; mode="${1##*-}"
    export MUJOCO_GL=glfw
    unset PYTHONPATH PYTHONHOME
    exec "$ROBOTICS_HOME/venv/learning/bin/python" training/robot_learning.py --task "$task" --mode "$mode" --steps 4096 --output "$LAB_ROOT/results/learning" ;;
  verify) exec /usr/bin/python3 simulation/run_lab.py --seconds 120 ;;
  check) exec /usr/bin/python3 scripts/verify.py --smoke ;;
  *) echo "Unknown option: $1"; exit 2 ;;
esac
