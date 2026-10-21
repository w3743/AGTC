#!/usr/bin/env bash
set -euo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="$FIELD_ROOT/external/IsaacLab"
ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="5870804e4f6aee57bdef68d59f67d328b990d1e1"
ISAACLAB_TOOLS_HOME="${ISAACLAB_TOOLS_HOME:-$HOME/robotics-lab/isaac-tools}"

export UV_PROJECT_ENVIRONMENT="/home/wangj/robotics-lab/venv/isaaclab-field"
export UV_CACHE_DIR="$FIELD_ROOT/.cache/uv"
export UV_LINK_MODE=copy
export WARP_CACHE_PATH="$FIELD_ROOT/.cache/warp"
export ISAACLAB_TOOLS_HOME
export PATH="$ISAACLAB_TOOLS_HOME/bin:$PATH"

fail() {
    printf 'Isaac Lab 安装失败：%s\n' "$*" >&2
    exit 1
}

[[ "${WSL_DISTRO_NAME:-}" == "Ubuntu-24.04" ]] \
    || fail '请从 Windows 的 安装IsaacLab.cmd 启动，或在 Ubuntu-24.04 WSL 中运行此脚本。'
[[ "$(uname -m)" == "x86_64" ]] || fail '仅支持 x86_64 WSL。'
[[ -x /usr/bin/python3.12 ]] || fail '缺少 /usr/bin/python3.12。'
command -v git >/dev/null 2>&1 || fail '缺少 git。'
[[ "$EUID" -ne 0 ]] || fail '请使用 Ubuntu 的普通用户运行，不要以 root 安装。'

UV_EXECUTABLE="$ISAACLAB_TOOLS_HOME/bin/uv"
if [[ ! -x "$UV_EXECUTABLE" ]]; then
    UV_EXECUTABLE="$(command -v uv || true)"
fi
if [[ -z "$UV_EXECUTABLE" || ! -x "$UV_EXECUTABLE" ]]; then
    mkdir -p "$ISAACLAB_TOOLS_HOME"
    /usr/bin/python3.12 -m pip install --disable-pip-version-check --upgrade \
        --target "$ISAACLAB_TOOLS_HOME" 'uv==0.13.1'
    UV_EXECUTABLE="$ISAACLAB_TOOLS_HOME/bin/uv"
fi
[[ -x "$UV_EXECUTABLE" ]] || fail "uv 安装后仍找不到可执行文件：$ISAACLAB_TOOLS_HOME/bin/uv"

if [[ -e "$ISAACLAB_ROOT" ]]; then
    git -C "$ISAACLAB_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || fail "$ISAACLAB_ROOT 已存在但不是 Git checkout；为避免覆盖文件，安装器不会替换它。"
else
    mkdir -p "$(dirname "$ISAACLAB_ROOT")"
    git clone --filter=blob:none --no-checkout "$ISAACLAB_URL" "$ISAACLAB_ROOT"
    git -C "$ISAACLAB_ROOT" checkout --detach "$ISAACLAB_COMMIT"
fi

actual_commit="$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$ISAACLAB_COMMIT" ]] \
    || fail "Isaac Lab checkout 是 $actual_commit，要求固定提交 $ISAACLAB_COMMIT；请先核对 checkout，安装器不会重置它。"

mkdir -p "$UV_CACHE_DIR" "$WARP_CACHE_PATH" "$(dirname "$UV_PROJECT_ENVIRONMENT")"
cd "$ISAACLAB_ROOT"
"$UV_EXECUTABLE" sync --locked --extra sb3 --extra viser --extra test --python /usr/bin/python3.12

# Isaac Lab's lock pins Torch and SB3. Install only the matching contrib package
# without dependency resolution so it cannot replace those upstream pins.
"$UV_EXECUTABLE" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps 'sb3-contrib==2.9.0'
"$UV_EXECUTABLE" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps 'vqf==2.1.2'
"$UV_EXECUTABLE" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$FIELD_ROOT"

"$UV_PROJECT_ENVIRONMENT/bin/python" -c \
    'import isaaclab, sb3_contrib, stable_baselines3, torch; print("Isaac Lab", isaaclab.__file__); print("PyTorch", torch.__version__); print("Stable-Baselines3", stable_baselines3.__version__); print("SB3-Contrib", sb3_contrib.__version__)'
printf 'Isaac Lab runtime 安装完成（提交 %s）。\n' "$ISAACLAB_COMMIT"
