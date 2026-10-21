#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIELD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTER_DIR="$SCRIPT_DIR/outer_$STAMP"
if [[ -e "$OUTER_DIR" ]]; then
  OUTER_DIR="$SCRIPT_DIR/outer_${STAMP}_$$"
fi
mkdir -p "$OUTER_DIR"
cd "$FIELD_ROOT"

cat > "$OUTER_DIR/outer_command.txt" <<'EOF'
Main exact command:
env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= bash results/project_history/cleanup_20261020_batch15/run_batch15_validation.sh

Additional exact checks:
bash -n results/isaac_field_viewer/front_guides_rough_20261020/run_viewer.sh
bash -n results/isaac_field_diagnosis/front_guide_integration_20261020/attempt6_native_after_solver_map_and_seed/run_front_guide_native.sh
env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= scripts/isaaclab_python.sh -m py_compile results/isaac_field_diagnosis/front_guide_step_probe_20261020/probe.py

The launcher runs checks only. It does not start a viewer, training job, or physics simulation.
EOF

FAILED=0
run_check() {
  local name="$1"
  shift
  local rc
  if "$@" > "$OUTER_DIR/$name.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$OUTER_DIR/$name.exit_code"
  if [[ "$rc" -ne 0 ]]; then
    FAILED=1
  fi
}

run_check main_runner env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
  bash results/project_history/cleanup_20261020_batch15/run_batch15_validation.sh
run_check bash_n_viewer bash -n \
  results/isaac_field_viewer/front_guides_rough_20261020/run_viewer.sh
run_check bash_n_native_wrapper bash -n \
  results/isaac_field_diagnosis/front_guide_integration_20261020/attempt6_native_after_solver_map_and_seed/run_front_guide_native.sh
run_check probe_py_compile env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
  scripts/isaaclab_python.sh -m py_compile \
  results/isaac_field_diagnosis/front_guide_step_probe_20261020/probe.py

if [[ "$FAILED" -eq 0 ]]; then
  printf '0\n' > "$OUTER_DIR/outer.exit_code"
  printf 'outer_result_dir=%s\noverall_exit=0\n' "$OUTER_DIR" > "$OUTER_DIR/outer_status.txt"
  exit 0
fi

printf '1\n' > "$OUTER_DIR/outer.exit_code"
printf 'outer_result_dir=%s\noverall_exit=1\n' "$OUTER_DIR" > "$OUTER_DIR/outer_status.txt"
exit 1
