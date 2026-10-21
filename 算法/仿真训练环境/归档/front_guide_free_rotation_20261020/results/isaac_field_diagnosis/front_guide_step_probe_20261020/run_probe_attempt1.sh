#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
EXPERIMENT_DIR="$ROOT_DIR/results/isaac_field_diagnosis/front_guide_step_probe_20261020"
OUTPUT_DIR="$EXPERIMENT_DIR/run_001"
CAPTURE_LOG="$EXPERIMENT_DIR/run_001.capture.log"
PROBE_REL="results/isaac_field_diagnosis/front_guide_step_probe_20261020/probe.py"

if [[ -e "$OUTPUT_DIR" || -e "$CAPTURE_LOG" ]]; then
  printf 'Refusing to overwrite an existing attempt-1 output or capture log.\n' >&2
  exit 73
fi
cd "$ROOT_DIR"

PROBE_SHA256="$(sha256sum "$PROBE_REL" | awk '{print $1}')"
ENV_SHA256="$(sha256sum isaac_field/env.py | awk '{print $1}')"
ASSETS_SHA256="$(sha256sum isaac_field/assets.py | awk '{print $1}')"
DRIVE_SHA256="$(sha256sum isaac_field/newton_drive.py | awk '{print $1}')"
TASK_SHA256="$(sha256sum isaac_field/task.py | awk '{print $1}')"
LAUNCHER_SHA256="$(sha256sum "$BASH_SOURCE" | awk '{print $1}')"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAUNCH_COMMAND="RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh $PROBE_REL"

set +e
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh "$PROBE_REL" > "$CAPTURE_LOG" 2>&1
ACTUAL_EXIT_CODE=$?
set -e
FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir "$OUTPUT_DIR"
fi
mv "$CAPTURE_LOG" "$OUTPUT_DIR/probe.log"
cat > "$OUTPUT_DIR/run_manifest.json" <<EOF
{
  "attempt": "run_001",
  "started_utc": "$STARTED_UTC",
  "finished_utc": "$FINISHED_UTC",
  "working_directory": "$ROOT_DIR",
  "command": "$LAUNCH_COMMAND",
  "argv": ["scripts/isaaclab_python.sh", "$PROBE_REL"],
  "environment": {"RUN_ISAAC_PHYSICS": "1", "wsl_distribution": "Ubuntu-24.04"},
  "source_sha256": {
    "$PROBE_REL": "$PROBE_SHA256",
    "isaac_field/env.py": "$ENV_SHA256",
    "isaac_field/assets.py": "$ASSETS_SHA256",
    "isaac_field/newton_drive.py": "$DRIVE_SHA256",
    "isaac_field/task.py": "$TASK_SHA256"
  },
  "launcher_sha256": "$LAUNCHER_SHA256",
  "actual_exit_code": $ACTUAL_EXIT_CODE
}
EOF
printf '{"started_utc":"%s","finished_utc":"%s","actual_exit_code":%d,"status":"%s","log":"probe.log","manifest":"run_manifest.json"}\n' \
  "$STARTED_UTC" "$FINISHED_UTC" "$ACTUAL_EXIT_CODE" \
  "$(if [[ $ACTUAL_EXIT_CODE -eq 0 ]]; then printf passed; else printf failed; fi)" \
  > "$OUTPUT_DIR/probe.exit.json"
exit "$ACTUAL_EXIT_CODE"
