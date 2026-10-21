#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/post_side_static_validation.log"
STATUS_PATH="$OUT_DIR/post_side_static_validation_exit.json"
if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite post-side static validation evidence\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
{
  bash -n "$OUT_DIR/run_cpu_math_tests_attempt2.sh"
  cpu_runner_rc=$?
  bash -n "$OUT_DIR/run_pi_probe.sh"
  probe_runner_rc=$?
  bash -n "$OUT_DIR/run_native_v5_regression_attempt1.sh"
  native_runner_rc=$?
  bash -n "$OUT_DIR/run_paired_capped_p_and_pi.sh"
  pair_runner_rc=$?
  bash -n "$OUT_DIR/run_post_side_static_checks.sh"
  static_runner_rc=$?
  python3 - "$OUT_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "runtime_pi.py",
    "test_pi_counterfactual_math.py",
    "pi_counterfactual_probe.py",
    "compare_pi_counterfactual_runs.py",
):
    path = root / name
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PYTHON_AST_OK {path}")
PY
  ast_rc=$?
  rc=0
  for item in "$cpu_runner_rc" "$probe_runner_rc" "$native_runner_rc" "$pair_runner_rc" "$static_runner_rc" "$ast_rc"; do
    if (( item != 0 )); then rc=$item; break; fi
  done
  printf 'bash_cpu=%s bash_probe=%s bash_native=%s bash_pair=%s bash_static=%s python_ast=%s\n' \
    "$cpu_runner_rc" "$probe_runner_rc" "$native_runner_rc" "$pair_runner_rc" "$static_runner_rc" "$ast_rc"
} >"$LOG_PATH" 2>&1
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s}\n' "$rc" >"$STATUS_PATH"
exit "$rc"
