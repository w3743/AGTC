#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)"
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [fresh-output-subdirectory]\n' "$0" >&2
  exit 2
fi
OUTPUT_NAME="${1:-.}"
if [[ "$OUTPUT_NAME" != . && ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Output name must be a simple fresh subdirectory name.\n' >&2
  exit 2
fi
OUTPUT_DIR="$SCRIPT_DIR"
if [[ "$OUTPUT_NAME" != . ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_NAME"
  mkdir -p -- "$OUTPUT_DIR"
fi
cd -- "$REPO_ROOT"

for name in full_cpu_pytest.log compileall.log bash_syntax.log validation_exit.json; do
  if [[ -e "$OUTPUT_DIR/$name" || -L "$OUTPUT_DIR/$name" ]]; then
    printf 'Refusing to overwrite existing validation output: %s\n' "$OUTPUT_DIR/$name" >&2
    exit 73
  fi
done

unset RUN_ISAAC_PHYSICS
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q >"$OUTPUT_DIR/full_cpu_pytest.log" 2>&1
pytest_exit=$?
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts >"$OUTPUT_DIR/compileall.log" 2>&1
compile_exit=$?
: >"$OUTPUT_DIR/bash_syntax.log"
syntax_exit=0
for script in scripts/install_isaaclab.sh scripts/isaaclab_python.sh "$SCRIPT_DIR/run_cpu_validation.sh"; do
  bash -n "$script" >>"$OUTPUT_DIR/bash_syntax.log" 2>&1
  script_exit=$?
  printf 'bash_n_exit_code=%d script=%s\n' "$script_exit" "$script" >>"$OUTPUT_DIR/bash_syntax.log"
  if [[ "$script_exit" -ne 0 && "$syntax_exit" -eq 0 ]]; then syntax_exit="$script_exit"; fi
done
bash_exit=$syntax_exit

aggregate_exit=0
for code in "$pytest_exit" "$compile_exit" "$bash_exit"; do
  if [[ "$code" -ne 0 && "$aggregate_exit" -eq 0 ]]; then aggregate_exit="$code"; fi
done
cat >"$OUTPUT_DIR/validation_exit.json" <<EOF
{
  "schema": "minimal-input-effort-limit-cpu-validation-v1",
  "pytest_exit_code": $pytest_exit,
  "compileall_exit_code": $compile_exit,
  "bash_syntax_exit_code": $bash_exit,
  "aggregate_exit_code": $aggregate_exit
}
EOF
printf 'pytest_exit_code=%d\ncompileall_exit_code=%d\nbash_syntax_exit_code=%d\naggregate_exit_code=%d\n' \
  "$pytest_exit" "$compile_exit" "$bash_exit" "$aggregate_exit"
exit "$aggregate_exit"
