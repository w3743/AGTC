#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
EXPERIMENT_DIR="$ROOT_DIR/results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020"
PAIR_DIR="$EXPERIMENT_DIR/pair_attempt3"
PROBE_REL="results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/probe.py"
ANALYZER_REL="results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/verify_pair_attempt3.py"
RUNNER_REL="results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/run_pair_attempt3.sh"

if [[ "${RUN_FRONT_GUIDE_PAIR:-}" != "1" ]]; then
  printf 'Prepared only; set RUN_FRONT_GUIDE_PAIR=1 after root grants exclusive GPU permission.\n' >&2
  exit 64
fi
if [[ -e "$PAIR_DIR" ]]; then
  printf 'Refusing to overwrite existing paired attempt directory: %s\n' "$PAIR_DIR" >&2
  exit 73
fi

mkdir "$PAIR_DIR"
cd "$ROOT_DIR"

PROBE_SHA256="$(sha256sum "$PROBE_REL" | awk '{print $1}')"
ENV_SHA256="$(sha256sum isaac_field/env.py | awk '{print $1}')"
ASSETS_SHA256="$(sha256sum isaac_field/assets.py | awk '{print $1}')"
DRIVE_SHA256="$(sha256sum isaac_field/newton_drive.py | awk '{print $1}')"
TASK_SHA256="$(sha256sum isaac_field/task.py | awk '{print $1}')"
ANALYZER_SHA256="$(sha256sum "$ANALYZER_REL" | awk '{print $1}')"
RUNNER_SHA256="$(sha256sum "$RUNNER_REL" | awk '{print $1}')"
cp "$PROBE_REL" "$PAIR_DIR/probe.py.source"
cp "$ANALYZER_REL" "$PAIR_DIR/verify_pair_attempt3.py.source"
cp "$RUNNER_REL" "$PAIR_DIR/run_pair_attempt3.sh.source"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FREE_EXIT_CODE="null"
COUPLED_EXIT_CODE="null"
ANALYSIS_EXIT_CODE="null"

write_pair_manifest() {
  local status="$1"
  local updated_utc
  updated_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "$PAIR_DIR/pair_manifest.json.tmp" <<EOF
{
  "attempt": "pair_attempt3",
  "status": "$status",
  "started_utc": "$STARTED_UTC",
  "updated_utc": "$updated_utc",
  "working_directory": "$ROOT_DIR",
  "wsl_distribution": "Ubuntu-24.04",
  "gpu_gate": "RUN_FRONT_GUIDE_PAIR=1 (must be set only after explicit exclusive GPU permission)",
  "variant_order": ["free", "same_side_track_speed"],
  "run_directories": {
    "free": "$PAIR_DIR/free_run_003",
    "same_side_track_speed": "$PAIR_DIR/coupled_run_003"
  },
  "variant_runner_return_codes": {
    "free": $FREE_EXIT_CODE,
    "same_side_track_speed": $COUPLED_EXIT_CODE,
    "pair_cpu_analysis": $ANALYSIS_EXIT_CODE
  },
  "fixed_probe_contract": {
    "reset_seed": 87001,
    "environment_seed": 80003,
    "start_xy_yaw": [2.0, 2.0, 0.0],
    "goal_xy": [6.0, 2.0],
    "step_size_xyz_m": [0.8, 2.0, 0.06],
    "step_center_xyz_m": [3.0, 2.0, 0.03],
    "policy_steps": 200,
    "normalized_action": [0.05, 0.05, 0.0, 0.0],
    "physics_dt_s": 0.001,
    "decimation": 50,
    "policy_dt_s": 0.05,
    "trained_policy_comparison": false
  },
  "source_sha256": {
    "$PROBE_REL": "$PROBE_SHA256",
    "isaac_field/env.py": "$ENV_SHA256",
    "isaac_field/assets.py": "$ASSETS_SHA256",
    "isaac_field/newton_drive.py": "$DRIVE_SHA256",
    "isaac_field/task.py": "$TASK_SHA256",
    "$ANALYZER_REL": "$ANALYZER_SHA256",
    "$RUNNER_REL": "$RUNNER_SHA256"
  },
  "source_copies": {
    "probe": "probe.py.source",
    "pair_cpu_analyzer": "verify_pair_attempt3.py.source",
    "runner": "run_pair_attempt3.sh.source"
  },
  "analysis_outputs": ["pair_comparison.json", "paired_trajectory_deltas.jsonl"]
}
EOF
  mv "$PAIR_DIR/pair_manifest.json.tmp" "$PAIR_DIR/pair_manifest.json"
}

run_variant() {
  local mode="$1"
  local label="$2"
  local output_dir="$3"
  local capture_log="$PAIR_DIR/${label}.capture.log"
  local started_utc finished_utc command rc source_unchanged
  if [[ -e "$output_dir" || -e "$capture_log" ]]; then
    printf 'Refusing to overwrite existing %s run output.\n' "$label" >&2
    return 73
  fi
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  command="RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh $PROBE_REL --rotation-coupling $mode --output $output_dir"
  if RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh "$PROBE_REL" \
    --rotation-coupling "$mode" --output "$output_dir" > "$capture_log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi
  mv "$capture_log" "$output_dir/probe.log"
  cp "$PROBE_REL" "$output_dir/probe.py.source"
  source_unchanged=true
  [[ "$(sha256sum "$PROBE_REL" | awk '{print $1}')" == "$PROBE_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum isaac_field/env.py | awk '{print $1}')" == "$ENV_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum isaac_field/assets.py | awk '{print $1}')" == "$ASSETS_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum isaac_field/newton_drive.py | awk '{print $1}')" == "$DRIVE_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum isaac_field/task.py | awk '{print $1}')" == "$TASK_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum "$ANALYZER_REL" | awk '{print $1}')" == "$ANALYZER_SHA256" ]] || source_unchanged=false
  [[ "$(sha256sum "$RUNNER_REL" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || source_unchanged=false
  local run_probe_sha256
  run_probe_sha256="$(sha256sum "$output_dir/probe.py.source" | awk '{print $1}')"
  cat > "$output_dir/run_manifest.json" <<EOF
{
  "attempt": "$label",
  "rotation_coupling": "$mode",
  "started_utc": "$started_utc",
  "finished_utc": "$finished_utc",
  "working_directory": "$ROOT_DIR",
  "command": "$command",
  "argv": ["scripts/isaaclab_python.sh", "$PROBE_REL", "--rotation-coupling", "$mode", "--output", "$output_dir"],
  "environment": {"RUN_FRONT_GUIDE_PAIR": "1", "RUN_ISAAC_PHYSICS": "1", "wsl_distribution": "Ubuntu-24.04"},
  "source_sha256": {
    "$PROBE_REL": "$PROBE_SHA256",
    "probe.py.source": "$run_probe_sha256",
    "isaac_field/env.py": "$ENV_SHA256",
    "isaac_field/assets.py": "$ASSETS_SHA256",
    "isaac_field/newton_drive.py": "$DRIVE_SHA256",
    "isaac_field/task.py": "$TASK_SHA256",
    "$ANALYZER_REL": "$ANALYZER_SHA256",
    "$RUNNER_REL": "$RUNNER_SHA256"
  },
  "all_effective_sources_unchanged_through_run": $source_unchanged,
  "actual_exit_code": $rc,
  "log": "probe.log"
}
EOF
  printf '{"rotation_coupling":"%s","started_utc":"%s","finished_utc":"%s","actual_exit_code":%d,"source_unchanged":%s,"log":"probe.log","manifest":"run_manifest.json"}\n' \
    "$mode" "$started_utc" "$finished_utc" "$rc" "$source_unchanged" \
    > "$output_dir/probe.exit.json"
  if [[ "$source_unchanged" != true ]]; then
    return 97
  fi
  return "$rc"
}

write_pair_manifest "running_free" 
if run_variant "free" "free_run_003" "$PAIR_DIR/free_run_003"; then
  FREE_EXIT_CODE=0
else
  FREE_EXIT_CODE=$?
fi
write_pair_manifest "free_finished"
if [[ "$FREE_EXIT_CODE" -ne 0 ]]; then
  write_pair_manifest "free_probe_failed"
  exit "$FREE_EXIT_CODE"
fi

if run_variant "same_side_track_speed" "coupled_run_003" "$PAIR_DIR/coupled_run_003"; then
  COUPLED_EXIT_CODE=0
else
  COUPLED_EXIT_CODE=$?
fi
write_pair_manifest "coupled_finished"
if [[ "$COUPLED_EXIT_CODE" -ne 0 ]]; then
  write_pair_manifest "coupled_probe_failed"
  exit "$COUPLED_EXIT_CODE"
fi

if python3 "$ANALYZER_REL" "$PAIR_DIR" > "$PAIR_DIR/pair_analysis.capture.log" 2>&1; then
  ANALYSIS_EXIT_CODE=0
else
  ANALYSIS_EXIT_CODE=$?
fi
mv "$PAIR_DIR/pair_analysis.capture.log" "$PAIR_DIR/pair_analysis.log"
printf '{"started_utc":"%s","finished_utc":"%s","actual_exit_code":%d,"status":"%s","log":"pair_analysis.log","comparison":"pair_comparison.json"}\n' \
  "$STARTED_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ANALYSIS_EXIT_CODE" \
  "$(if [[ "$ANALYSIS_EXIT_CODE" -eq 0 ]]; then printf passed; else printf failed; fi)" \
  > "$PAIR_DIR/pair_analysis.exit.json"

if [[ "$ANALYSIS_EXIT_CODE" -eq 0 ]]; then
  write_pair_manifest "paired_analysis_passed"
else
  write_pair_manifest "paired_analysis_failed"
fi
exit "$ANALYSIS_EXIT_CODE"

