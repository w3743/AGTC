#!/usr/bin/env bash
set -uo pipefail

ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
DIAG_ROOT="$ROOT/results/isaac_field_diagnosis/steering_contact_repair_20261020"
SCRIPT_DIR="$DIAG_ROOT/cpu_validation"

if [[ "${RUN_STEERING_CONTACT_CPU_VALIDATION:-}" != "1" ]]; then
  printf 'Refusing to run until parent authorization sets RUN_STEERING_CONTACT_CPU_VALIDATION=1.\n' >&2
  exit 78
fi

ATTEMPT="${1:-attempt1}"
if [[ ! "$ATTEMPT" =~ ^attempt[0-9]+$ ]]; then
  printf 'Usage: %s [attemptN]\n' "$0" >&2
  exit 2
fi

OUT_DIR="$SCRIPT_DIR/$ATTEMPT"
if [[ -e "$OUT_DIR" || -L "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite existing validation output: %s\n' "$OUT_DIR" >&2
  exit 73
fi
if [[ ! -f "$ROOT/scripts/isaaclab_python.sh" || ! -d "$ROOT/tests" ]]; then
  printf 'Project checkout or official Python runtime is missing: %s\n' "$ROOT" >&2
  exit 2
fi

mkdir -p -- "$OUT_DIR"
cd -- "$ROOT"
unset RUN_ISAAC_PHYSICS
export CUDA_VISIBLE_DEVICES=

set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q >"$OUT_DIR/full_cpu_pytest.log" 2>&1
pytest_exit=$?

CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests >"$OUT_DIR/compileall.log" 2>&1
compileall_exit=$?

: >"$OUT_DIR/bash_syntax.log"
bash_syntax_exit=0
bash_target_count=0
while IFS= read -r -d '' script; do
  bash_target_count=$((bash_target_count + 1))
  bash -n "$script" >>"$OUT_DIR/bash_syntax.log" 2>&1
  script_exit=$?
  printf 'bash_n_exit_code=%d script=%s\n' "$script_exit" "$script" >>"$OUT_DIR/bash_syntax.log"
  if [[ "$script_exit" -ne 0 && "$bash_syntax_exit" -eq 0 ]]; then
    bash_syntax_exit=$script_exit
  fi
done < <(find "$ROOT/scripts" "$DIAG_ROOT" -type f -name '*.sh' -print0 | sort -z)
if [[ "$bash_target_count" -eq 0 && "$bash_syntax_exit" -eq 0 ]]; then
  bash_syntax_exit=2
  printf 'No saved .sh files found in validation scope.\n' >>"$OUT_DIR/bash_syntax.log"
fi

aggregate_exit=0
for code in "$pytest_exit" "$compileall_exit" "$bash_syntax_exit"; do
  if [[ "$code" -ne 0 && "$aggregate_exit" -eq 0 ]]; then
    aggregate_exit=$code
  fi
done
cat >"$OUT_DIR/validation_exit.json" <<EOF
{
  "schema": "steering-contact-repair-full-cpu-validation-v1",
  "attempt": "$ATTEMPT",
  "mode": "CPU only; RUN_ISAAC_PHYSICS unset; CUDA_VISIBLE_DEVICES empty",
  "pytest_exit_code": $pytest_exit,
  "compileall_exit_code": $compileall_exit,
  "bash_syntax_exit_code": $bash_syntax_exit,
  "bash_target_count": $bash_target_count,
  "aggregate_exit_code": $aggregate_exit
}
EOF
printf 'pytest_exit_code=%d\ncompileall_exit_code=%d\nbash_syntax_exit_code=%d\nbash_target_count=%d\naggregate_exit_code=%d\n' \
  "$pytest_exit" "$compileall_exit" "$bash_syntax_exit" "$bash_target_count" "$aggregate_exit"
exit "$aggregate_exit"
