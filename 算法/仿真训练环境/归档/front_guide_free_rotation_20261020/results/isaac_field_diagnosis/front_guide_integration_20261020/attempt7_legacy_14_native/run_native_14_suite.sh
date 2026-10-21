#!/usr/bin/env bash
set -uo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OUT/../../../.." && pwd)"
cd "$ROOT"
unset CUDA_VISIBLE_DEVICES WARP_DISABLE_CUDA
export RUN_ISAAC_PHYSICS=1
scripts/isaaclab_python.sh -m pytest -q -s tests/test_env_integration.py > "$OUT/native_14_suite.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/native_14_suite.exit_code"
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >> "$OUT/native_14_suite.log"
printf '{"actual_exit_code":%s,"device":"cuda","suite":"tests/test_env_integration.py","test_functions":14,"log_path":"%s/native_14_suite.log"}\n' "$rc" "$OUT" > "$OUT/native_14_suite.exit.json"
exit "$rc"