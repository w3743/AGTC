#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
if [[ $# -ne 0 ]]; then
  printf 'This one-case runner is pinned to flat_mesh_to_native_plane, dt=0.002, cone=elliptic, impratio=1 and accepts no arguments.\n' >&2
  exit 2
fi
DT="0.002"
CONE="elliptic"
IMPRATIO="1"
RUN_LABEL="flat-geometry-diagnostic"
CASE_NAME="flat-mesh-to-native-plane__dt-2ms__cone-elliptic__impratio-1"

if [[ "${RUN_CONTACT_GROUND_PLANE_V6:-}" != "1" ]]; then
  printf 'Runner requires RUN_CONTACT_GROUND_PLANE_V6=1 after parent authorization.\n' >&2
  exit 2
fi

OUTPUT_ROOT="$PROBE_DIR/contact_probe_runs_v6"
OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
LOG_PATH="$OUTPUT_ROOT/$CASE_NAME.log"
EXIT_PATH="$OUTPUT_ROOT/$CASE_NAME.exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior output/log/exit status: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
if ! mkdir -- "$OUTPUT_DIR"; then
  printf 'Could not create fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior log: %s\n' "$LOG_PATH" >&2
  rmdir "$OUTPUT_DIR" 2>/dev/null || true
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_CONTACT_GROUND_PLANE_V6=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_ground_plane_v6.py" \
  --physics-dt "$DT" \
  --cone "$CONE" \
  --impratio "$IMPRATIO" \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"physics_dt_s":%s,"cone":"%s","impratio":%s,"geometry_treatment":"flat_mesh_to_native_plane","run_label":"%s","case":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$DT" "$CONE" "$IMPRATIO" "$RUN_LABEL" "$CASE_NAME" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
