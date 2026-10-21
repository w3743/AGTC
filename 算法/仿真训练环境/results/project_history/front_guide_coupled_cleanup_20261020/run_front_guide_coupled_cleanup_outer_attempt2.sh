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
env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= bash results/project_history/front_guide_coupled_cleanup_20261020/run_front_guide_coupled_validation.sh

Static-only additional checks:
bash -n results/project_history/front_guide_coupled_cleanup_20261020/run_front_guide_coupled_validation.sh
bash -n results/isaac_field_viewer/front_guides_coupled_rough_20261020/run_viewer.sh
bash -n results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt8_native_full_pass/run_native.sh
bash -n results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt9_native_legacy14/run_native.sh
bash -n results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/run_pair_attempt1.sh
env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= scripts/isaaclab_python.sh -m py_compile results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/probe.py results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/verify_pair_attempt1.py

The runner executes CPU pytest, compileall, project-shell syntax checks and hash stability only. It does not prepare a trial, run the paired physics probe, launch a viewer, or start training. Child checks unset RUN_ISAAC_PHYSICS and set CUDA_VISIBLE_DEVICES to the empty string.
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
  printf '%s=%s\n' "$name" "$rc" >> "$OUTER_DIR/outer_status.txt"
  if [[ "$rc" -ne 0 ]]; then
    FAILED=1
  fi
}

printf 'started_utc=%s\nRUN_ISAAC_PHYSICS=unset\nCUDA_VISIBLE_DEVICES=<empty>\n' "$STAMP" \
  > "$OUTER_DIR/outer_status.txt"

run_check full_cpu_static_runner env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
  bash results/project_history/front_guide_coupled_cleanup_20261020/run_front_guide_coupled_validation.sh
run_check bash_n_new_validation_runner bash -n \
  results/project_history/front_guide_coupled_cleanup_20261020/run_front_guide_coupled_validation.sh
run_check bash_n_new_outer bash -n \
  results/project_history/front_guide_coupled_cleanup_20261020/run_front_guide_coupled_cleanup_outer.sh
run_check bash_n_coupled_viewer bash -n \
  results/isaac_field_viewer/front_guides_coupled_rough_20261020/run_viewer.sh
run_check bash_n_native_attempt8 bash -n \
  results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt8_native_full_pass/run_native.sh
run_check bash_n_legacy_native_attempt9 bash -n \
  results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt9_native_legacy14/run_native.sh
run_check bash_n_coupled_pair_runner bash -n \
  results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/run_pair_attempt1.sh
run_check py_compile_coupled_pair_files env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
  scripts/isaaclab_python.sh -m py_compile \
  results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/probe.py \
  results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/verify_pair_attempt1.py

MAIN_RUN_DIR="$(find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'run_*' \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
printf 'main_validation_dir=%s\n' "$MAIN_RUN_DIR" >> "$OUTER_DIR/outer_status.txt"
if [[ -z "$MAIN_RUN_DIR" || ! -f "$MAIN_RUN_DIR/source_hashes_before.sha256" \
      || ! -f "$MAIN_RUN_DIR/source_hashes_after.sha256" ]]; then
  printf 'source_hash_count=unavailable\n' >> "$OUTER_DIR/outer_status.txt"
  FAILED=1
else
  SOURCE_HASH_COUNT="$(wc -l < "$MAIN_RUN_DIR/source_hashes_before.sha256" | tr -d ' ')"
  printf 'source_hash_count=%s\n' "$SOURCE_HASH_COUNT" >> "$OUTER_DIR/outer_status.txt"
  if [[ -f "$MAIN_RUN_DIR/cpu_pytest.log" ]]; then
    tail -n 1 "$MAIN_RUN_DIR/cpu_pytest.log" > "$OUTER_DIR/pytest_summary.txt"
  fi
fi

if [[ "$FAILED" -eq 0 ]]; then
  printf 'overall_exit=0\nfinished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUTER_DIR/outer_status.txt"
  printf '0\n' > "$OUTER_DIR/outer.exit_code"
  exit 0
fi

printf 'overall_exit=1\nfinished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$OUTER_DIR/outer_status.txt"
printf '1\n' > "$OUTER_DIR/outer.exit_code"
exit 1
