#!/usr/bin/env bash
set -uo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OUT/../../../.." && pwd)"
cd "$ROOT"
unset CUDA_VISIBLE_DEVICES WARP_DISABLE_CUDA
export RUN_ISAAC_PHYSICS=1
scripts/isaaclab_python.sh -m pytest -q -s tests/test_front_guide_native.py > "$OUT/front_guide_native.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/front_guide_native.exit_code"
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >> "$OUT/front_guide_native.log"
printf '{"actual_exit_code":%s,"device":"cuda","suite":"tests/test_front_guide_native.py","log_path":"%s/front_guide_native.log"}\n' "$rc" "$OUT" > "$OUT/front_guide_native.exit.json"
exit "$rc"
