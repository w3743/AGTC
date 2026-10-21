#!/usr/bin/env bash
# Windows orchestrates root/user stages; native Ubuntu users run this directly.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROBOTICS_HOME="${ROBOTICS_HOME:-$HOME/robotics-lab}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
stage="${1:-all}"
fail() { echo "安装失败：$*" >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 && "$(uname -m)" == x86_64 ]] || fail '只支持 x86_64 Ubuntu 24.04。'
python3 -c 'import sys; assert sys.version_info[:2] == (3,12)' || fail '需要系统 Python 3.12。'
if [[ "$stage" == all ]]; then
  [[ $EUID -ne 0 ]] || fail '请用普通用户运行；系统安装阶段会请求 sudo。'
  sudo bash "$0" system
  exec bash "$0" user
fi
if [[ "$stage" == system ]]; then
  [[ $EUID -eq 0 ]] || fail 'system 阶段需要 root。'
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl gnupg
  # Download and check the official key before replacing the package-specific source.
  key=$(mktemp)
  trap 'rm -f "$key"' EXIT
  curl -fsSL --retry 3 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "$key"
  gpg --show-keys "$key" >/dev/null
  install -m 644 "$key" /usr/share/keyrings/robotics-lab-ros-keyring.gpg
  printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/robotics-lab-ros-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main' > /etc/apt/sources.list.d/robotics-lab-ros2.list
  apt-get update
  mapfile -t system_packages < <(grep -vE '^\s*(#|$)' "$ROOT/config/system-packages.txt")
  mapfile -t ros_packages < <(grep -vE '^\s*(#|$)' "$ROOT/config/ros-packages.lock")
  # Version availability is checked before any large install. Never silently upgrade pins.
  apt-get -s install --no-install-recommends "${system_packages[@]}" "${ros_packages[@]}" >/dev/null || fail '锁定软件版本不可用，或缺少 Ubuntu universe 源。请查看安装日志，不会自动换版。'
  apt-get install -y --no-install-recommends "${system_packages[@]}" "${ros_packages[@]}"
  echo 'SYSTEM_INSTALL_OK'
elif [[ "$stage" == user ]]; then
  [[ $EUID -ne 0 ]] || fail '请先在 Ubuntu 中创建普通用户，再重新运行安装。'
  [[ -f /opt/ros/jazzy/setup.bash ]] || fail '请先运行系统安装阶段。'
  [[ -z "${VIRTUAL_ENV:-}" ]] || fail '请退出已有 Python 虚拟环境后再安装。'
  mkdir -p "$ROBOTICS_HOME" "$ROOT/results"
  available=$(df -Pk "$ROBOTICS_HOME" | awk 'NR==2 {print $4}')
  [[ "$available" -gt 15728640 ]] || fail 'Linux 存储至少需要15GB可用空间；还要检查Windows所在盘的真实剩余空间。'
  [[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt 1048576 ]] || fail '项目目录所在磁盘至少需要1GB可用空间。'
  exec 9>"$ROBOTICS_HOME/install.lock"
  flock -n 9 || fail '已有安装程序在运行。'
  for kind in mapping drone; do
    flock -n "$ROBOTICS_HOME/$kind.lock" true || fail '请先停止运行中的仿真。'
  done
  # These two venvs are application-owned. Never inherit Ubuntu/ROS Python packages.
  for kind in learning px4; do
    python3 -m venv "$ROBOTICS_HOME/venv/$kind"
  done
  "$ROBOTICS_HOME/venv/learning/bin/python" -m pip install --disable-pip-version-check torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
  "$ROBOTICS_HOME/venv/learning/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/config/learning.lock"
  "$ROBOTICS_HOME/venv/px4/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/config/px4.lock"
  "$ROBOTICS_HOME/venv/learning/bin/python" -m pip check
  "$ROBOTICS_HOME/venv/px4/bin/python" -m pip check
  python3 "$ROOT/scripts/sources.py"
  source "$ROOT/scripts/env.sh"
  cmake --fresh -S "$ROOT/uwb_plugin" -B "$ROBOTICS_HOME/build/uwb" -G Ninja -DCMAKE_BUILD_TYPE=Release -DUWB_SOURCE_DIR="$ROBOTICS_HOME/src/UWBPX4Sim/uwb_gazebo_plugin"
  cmake --build "$ROBOTICS_HOME/build/uwb" -j4
  export PATH="$ROBOTICS_HOME/venv/px4/bin:$PATH"
  export GZ_DISTRO=harmonic
  unset PYTHONPATH PYTHONHOME
  make -C "$ROBOTICS_HOME/src/PX4-Autopilot" px4_sitl_default -j4
  /usr/bin/python3 "$ROOT/scripts/generate_world.py"
  /usr/bin/python3 "$ROOT/scripts/verify.py" --smoke
  echo 'DEPLOYMENT_OK：可以启动实验室。'
else
  fail '参数只能是 system、user 或 all。'
fi
