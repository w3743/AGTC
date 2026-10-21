#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/review_static_validation.log"
STATUS_PATH="$OUT_DIR/review_static_validation_exit.json"
if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite review static validation evidence\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
{
  scripts_to_check=(
    "$OUT_DIR/run_cpu_math_tests_attempt2.sh"
    "$OUT_DIR/run_pi_probe.sh"
    "$OUT_DIR/run_native_v5_regression_attempt1.sh"
    "$OUT_DIR/run_paired_capped_p_and_pi.sh"
    "$OUT_DIR/run_post_side_static_checks.sh"
    "$OUT_DIR/run_final_static_validation.sh"
    "$OUT_DIR/run_review_static_validation.sh"
  )
  bash_rc=0
  for script in "${scripts_to_check[@]}"; do
    bash -n "$script"
    current_rc=$?
    printf 'BASH_SYNTAX %s exit=%s\n' "$script" "$current_rc"
    if (( current_rc != 0 && bash_rc == 0 )); then bash_rc=$current_rc; fi
  done
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
  if (( bash_rc != 0 )); then rc=$bash_rc; else rc=$ast_rc; fi
  printf 'bash_syntax_rc=%s python_ast_rc=%s\n' "$bash_rc" "$ast_rc"
} >"$LOG_PATH" 2>&1
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s}\n' "$rc" >"$STATUS_PATH"
exit "$rc"
