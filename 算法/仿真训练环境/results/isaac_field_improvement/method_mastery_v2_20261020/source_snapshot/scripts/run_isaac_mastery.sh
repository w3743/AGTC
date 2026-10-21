#!/usr/bin/env bash
# The parent shell records the child's actual exit, including interruptions.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash scripts/run_isaac_mastery.sh OUTPUT [--device cuda:0]\n' >&2
  exit 2
fi
OUT="$1"
shift
if [[ ! -f "$OUT/protocol.json" || ! -f "$OUT/run_status.json" ]]; then
  printf 'Prepare a new mastery experiment first.\n' >&2
  exit 2
fi
if [[ -e "$OUT/launch.log" || -e "$OUT/process_exit.json" ]]; then
  printf 'Refusing to overwrite launch or exit evidence.\n' >&2
  exit 2
fi
if ! mkdir "$OUT/.launch_lock"; then
  printf 'This trial has already been claimed by a launcher.\n' >&2
  exit 2
fi
bash scripts/isaaclab_python.sh scripts/train_isaac_mastery.py --launch --output "$OUT" "$@" > "$OUT/launch.log" 2>&1
rc=$?
printf '%d\n' "$rc" > "$OUT/python.exit_code"
printf '{"actual_exit_code":%d,"timestamp_utc":"%s","source":"parent_shell_waited_for_child"}\n' \
  "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/process_exit.json"
exit "$rc"
