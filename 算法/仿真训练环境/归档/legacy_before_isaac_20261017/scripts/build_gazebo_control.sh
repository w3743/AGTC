#!/usr/bin/env bash
set -eo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LAB_ROOT/scripts/env.sh"

# Jazzy vendors Gazebo dependencies in separate prefixes. Make their pkg-config
# metadata visible only to this local build; no system package is installed.
PKG_CONFIG_PATH="$(find /opt/ros/jazzy/opt -path '*/lib/pkgconfig' -type d -print | paste -sd: -)${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_PATH

output="$LAB_ROOT/.runtime-gazebo/libfield_gz_control.so"
mkdir -p "$(dirname "$output")"

"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared \
  $(pkg-config --cflags gz-transport13 gz-msgs10) \
  "$LAB_ROOT/simulation/native_gazebo_control.cc" \
  -o "$output" \
  $(pkg-config --libs gz-transport13 gz-msgs10)

printf 'Built %s\n' "$output"
