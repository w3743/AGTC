#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/preparation_static_validation.log"

if [[ -e "$LOG_PATH" ]]; then
  printf 'Refusing to overwrite %s\n' "$LOG_PATH" >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
{
  bash -n "$OUT_DIR/run_cpu_math_tests.sh"
  bash_cpu_rc=$?
  bash -n "$OUT_DIR/run_pi_probe.sh"
  bash_probe_rc=$?
  python3 - "$OUT_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "runtime_pi.py",
    "test_pi_counterfactual_math.py",
    "pi_counterfactual_probe.py",
):
    path = root / name
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PYTHON_AST_OK {path}")
PY
  ast_rc=$?
  rc=0
  if (( bash_cpu_rc != 0 )); then
    rc=$bash_cpu_rc
  elif (( bash_probe_rc != 0 )); then
    rc=$bash_probe_rc
  elif (( ast_rc != 0 )); then
    rc=$ast_rc
  fi
  printf 'bash_cpu_rc=%s bash_probe_rc=%s python_ast_rc=%s\n' \
    "$bash_cpu_rc" "$bash_probe_rc" "$ast_rc"
} >"$LOG_PATH" 2>&1
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
exit "$rc"
