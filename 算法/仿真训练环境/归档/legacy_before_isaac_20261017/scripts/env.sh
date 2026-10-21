#!/usr/bin/env bash
# Source this file in a fresh bash shell. Keep ROS Python separate from ML venvs.
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAB_ROOT
export ROBOTICS_HOME="${ROBOTICS_HOME:-$HOME/robotics-lab}"
mkdir -p "$ROBOTICS_HOME"
export PYTHONDONTWRITEBYTECODE=1
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_PARTITION="${GZ_PARTITION:-robotics_lab_$(id -u)}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$ROBOTICS_HOME/build/uwb:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export QT_QPA_PLATFORM=xcb
if [[ -e /dev/dxg && -n "${DISPLAY:-}" && -z "${GALLIUM_DRIVER:-}" ]]; then
  if /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1; then
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
  fi
  if GALLIUM_DRIVER=d3d12 glxinfo -B 2>/dev/null | grep -q 'Accelerated: yes'; then
    export GALLIUM_DRIVER=d3d12
  fi
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
