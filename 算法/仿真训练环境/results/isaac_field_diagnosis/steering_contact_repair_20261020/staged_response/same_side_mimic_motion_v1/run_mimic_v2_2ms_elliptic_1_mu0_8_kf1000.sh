#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/../../../../.." && pwd)"
RUNNER="$HERE/response_runner.py"
OUTPUT_ROOT="$HERE/runs"
CASE_NAME="mimic_v2_2ms_elliptic_1_mu0_8_kf1000"
OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
LOG_PATH="$OUTPUT_ROOT/$CASE_NAME.log"
EXIT_PATH="$OUTPUT_ROOT/$CASE_NAME.exit.json"

if [[ "${RUN_STAGED_MIMIC_RESPONSE:-}" != "1" ]]; then
  printf 'Requires RUN_STAGED_MIMIC_RESPONSE=1 after parent confirms GPU is idle.\n' >&2
  exit 2
fi
if [[ $# -ne 0 ]]; then
  printf 'Pinned mimic candidate runner accepts no arguments.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to reuse prior output/log/exit: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
if ! mkdir -- "$OUTPUT_DIR"; then
  printf 'Could not reserve fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior log: %s\n' "$LOG_PATH" >&2
  rmdir "$OUTPUT_DIR" 2>/dev/null || true
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_STAGED_MIMIC_RESPONSE=1 scripts/isaaclab_python.sh "$RUNNER" --mode mimic >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
runner_sha="$(sha256sum "$RUNNER" | cut -d' ' -f1)"
probe_sha="$(sha256sum "$ROOT_DIR/results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/staged_motion_response_probe.py" | cut -d' ' -f1)"
(set -o noclobber; printf '{"actual_exit_code":%s,"mode":"mimic","physics_dt_s":0.002,"policy_dt_s":0.05,"decimation":25,"cone":"elliptic","impratio":1,"mu":0.8,"kf":1000,"effort_cap_nm":2,"runner_sha256":"%s","collector_sha256":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$runner_sha" "$probe_sha" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
