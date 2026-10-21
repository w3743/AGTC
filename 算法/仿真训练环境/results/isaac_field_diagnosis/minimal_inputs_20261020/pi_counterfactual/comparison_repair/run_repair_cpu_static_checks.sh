#!/usr/bin/env bash
set -uo pipefail

REPAIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(cd "$REPAIR_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PI_DIR/../../../.." && pwd)"
LOG_PATH="$REPAIR_DIR/repair_cpu_static_checks.log"
STATUS_PATH="$REPAIR_DIR/repair_cpu_static_checks_exit.json"
if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite comparison repair validation evidence\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
{
  bash_rc=0
  for script in "$REPAIR_DIR/run_existing_raw_comparison.sh" "$REPAIR_DIR/run_repair_cpu_static_checks.sh"; do
    bash -n "$script"
    current_rc=$?
    printf 'BASH_SYNTAX %s exit=%s\n' "$script" "$current_rc"
    if (( current_rc != 0 && bash_rc == 0 )); then bash_rc=$current_rc; fi
  done
  python3 - "$PI_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ("compare_pi_counterfactual_runs.py", "test_confirmation_protocol_contract.py"):
    path = root / name
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PYTHON_AST_OK {path}")
PY
  ast_rc=$?
  if (( bash_rc != 0 )); then static_rc=$bash_rc; else static_rc=$ast_rc; fi
  printf 'bash_syntax_rc=%s python_ast_rc=%s\n' "$bash_rc" "$ast_rc"
  if (( static_rc == 0 )); then
    CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q \
      "$PI_DIR/test_confirmation_protocol_contract.py"
    cpu_rc=$?
  else
    cpu_rc=125
  fi
  printf 'cpu_test_rc=%s\n' "$cpu_rc"
  if (( static_rc != 0 )); then final_rc=$static_rc; else final_rc=$cpu_rc; fi
} >"$LOG_PATH" 2>&1
rc=$final_rc
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s}\n' "$rc" >"$STATUS_PATH"
exit "$rc"
