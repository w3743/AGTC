#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
P_OUT="$OUT_DIR/confirmation_p"
PI_OUT="$OUT_DIR/confirmation_pi"
PROTOCOL="$OUT_DIR/confirmation_protocol.json"
AUDIT="$OUT_DIR/analysis/initial_state_differences.json"
SOURCE_SNAPSHOT="$OUT_DIR/source_snapshot/source_snapshot_manifest.json"
FREEZE="$OUT_DIR/confirmation_source_freeze_sha256.txt"
PAIR_LOG="$OUT_DIR/confirmation_pair.log"
PAIR_STATUS="$OUT_DIR/confirmation_pair_exit.json"
PAIR_RESULT="$OUT_DIR/confirmation_pair_summary.json"

if [[ "${RUN_PI_CONFIRMATION:-}" != "1" ]]; then
  printf 'Prepared only; set RUN_PI_CONFIRMATION=1 after explicit parent authorization\n' >&2
  exit 2
fi
for path in "$FREEZE" "$PAIR_LOG" "$PAIR_STATUS" "$PAIR_RESULT" \
            "$OUT_DIR/confirmation_p.log" "$OUT_DIR/confirmation_p_exit.json" \
            "$OUT_DIR/confirmation_pi.log" "$OUT_DIR/confirmation_pi_exit.json"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to overwrite confirmation evidence: %s\n' "$path" >&2
    exit 2
  fi
done
for output_dir in "$P_OUT" "$PI_OUT"; do
  if [[ -e "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing non-empty confirmation output directory: %s\n' "$output_dir" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
done

cd "$ROOT_DIR"
python3 - "$PROTOCOL" "$AUDIT" "$SOURCE_SNAPSHOT" "$OUT_DIR/native_v5_regression_attempt1_exit.json" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol_path, audit_path, snapshot_path, native_status_path = map(pathlib.Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
native = json.loads(native_status_path.read_text(encoding="utf-8"))
if protocol.get("status") != "prepared_not_run":
    raise SystemExit("confirmation protocol is not in prepared state")
if native.get("actual_exit_code") != 0:
    raise SystemExit("native v5 effort regression must have exit 0 first")
if audit.get("source_runs", {}).get("pair_attempt_exit", {}).get("actual_exit_code") != 1:
    raise SystemExit("preserved exact-pair failure evidence is missing")
pairing = audit.get("pairing_metadata", {})
if not all(pairing.get(key) is True for key in (
    "source_sha256_equal", "base_drive_config_equal", "environment_and_reset_config_equal",
    "all_actions_equal", "settle_control_strategy_equal",
)):
    raise SystemExit("old P/PI configuration or source pairing audit is incomplete")
if not all(row.get("source_matches_snapshot") is True for row in snapshot.get("original_sources", [])):
    raise SystemExit("source snapshot contains an unverified copy")
if snapshot.get("original_source_count") != len(snapshot.get("original_sources", [])):
    raise SystemExit("source snapshot manifest count mismatch")

tolerances = protocol["initial_state_pairing_tolerances"]
observed = audit.get("initial_state_audit", {}).get("max_abs_difference_by_field", {})
field_prefixes = {
    "position_xy_m": ("position_xy_m[",),
    "root_z_m": ("root_z_m",),
    "roll_pitch_yaw_rad": ("roll_pitch_yaw_rad[",),
    "actual_v_w": ("actual_v_w[",),
    "wheel_actual_qd_rad_s": ("wheel_actual_qd_rad_s[",),
    "track_command_mps": ("track_command_mps[",),
    "wheel_target_rad_s": ("wheel_target_rad_s[",),
}
for field, tolerance in tolerances.items():
    if field == "comparison_rule":
        continue
    matching = [
        float(value) for key, value in observed.items()
        if key == field or any(key.startswith(prefix) for prefix in field_prefixes.get(field, ()))
    ]
    maximum = max(matching, default=0.0)
    if maximum > float(tolerance):
        raise SystemExit(f"old initial-state difference exceeds confirmation tolerance: {field}={maximum}>{tolerance}")
    print(f"OLD_STATE_DELTA_OK {field} max_abs={maximum} tolerance={tolerance}")

snapshot_root = snapshot_path.parent
for entry in snapshot.get("original_sources", []):
    relative = entry["snapshot_relative_path"].replace("\\", "/")
    copied = snapshot_root / relative
    actual = hashlib.sha256(copied.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise SystemExit(f"source snapshot copy changed: {relative}")
print("PREFLIGHT_OK no GPU started")
PY
preflight_rc=$?
if (( preflight_rc != 0 )); then
  printf 'PRECHECK_EXIT_CODE=%s\n' "$preflight_rc" >&2
  exit "$preflight_rc"
fi

freeze_files=()
while read -r expected relative; do
  [[ -n "${relative:-}" ]] && freeze_files+=("$relative")
done <"$OUT_DIR/source_freeze_sha256.txt"
freeze_files+=(
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_protocol.json
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/source_snapshot/capture_source_snapshot.ps1
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/source_snapshot/source_snapshot_manifest.json
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/analysis/initial_state_differences.json
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_confirmation_pair.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/test_pi_counterfactual_math.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_confirmation_cpu_tests.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_confirmation_static_checks.sh
)
while IFS= read -r snapshot_relative; do
  freeze_files+=("results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/source_snapshot/$snapshot_relative")
done < <(python3 - "$SOURCE_SNAPSHOT" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in data["original_sources"]:
    print(row["snapshot_relative_path"].replace("\\", "/"))
PY
)
sha256sum "${freeze_files[@]}" >"$FREEZE"

set +e
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh \
  "$OUT_DIR/pi_counterfactual_probe.py" \
  --ki 0 --output-dir "$P_OUT" --brake-seconds 5 \
  --case diff_left_plus_0.05,diff_left_plus_0.10,brake_straight_0.20_to_zero,brake_diff_left_plus_0.80_to_zero \
  >"$OUT_DIR/confirmation_p.log" 2>&1
p_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$p_rc" >>"$OUT_DIR/confirmation_p.log"
printf '{"run_label":"confirmation_p","ki":0,"actual_exit_code":%s}\n' "$p_rc" \
  >"$OUT_DIR/confirmation_p_exit.json"

pi_rc=125
compare_rc=125
if (( p_rc == 0 )); then
  set +e
  RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh \
    "$OUT_DIR/pi_counterfactual_probe.py" \
    --ki 0.1 --output-dir "$PI_OUT" --brake-seconds 5 \
    --case diff_left_plus_0.05,diff_left_plus_0.10,brake_straight_0.20_to_zero,brake_diff_left_plus_0.80_to_zero \
    >"$OUT_DIR/confirmation_pi.log" 2>&1
  pi_rc=$?
  set -e
  printf 'ACTUAL_EXIT_CODE=%s\n' "$pi_rc" >>"$OUT_DIR/confirmation_pi.log"
  printf '{"run_label":"confirmation_pi","ki":0.1,"actual_exit_code":%s}\n' "$pi_rc" \
    >"$OUT_DIR/confirmation_pi_exit.json"
fi

if (( p_rc == 0 && pi_rc == 0 )); then
  set +e
  python3 "$OUT_DIR/compare_pi_counterfactual_runs.py" \
    --root "$OUT_DIR" \
    --output "$PAIR_RESULT" \
    --protocol-mode confirmation-v1 \
    --protocol "$PROTOCOL" \
    --source-freeze "$FREEZE"
  compare_rc=$?
  set -e
fi
rc=0
if (( p_rc != 0 )); then
  rc=$p_rc
elif (( pi_rc != 0 )); then
  rc=$pi_rc
elif (( compare_rc != 0 )); then
  rc=$compare_rc
fi
printf 'p_exit=%s pi_exit=%s comparison_exit=%s\n' "$p_rc" "$pi_rc" "$compare_rc" >"$PAIR_LOG"
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$PAIR_LOG"
printf '{"p_exit":%s,"pi_exit":%s,"comparison_exit":%s,"actual_exit_code":%s}\n' \
  "$p_rc" "$pi_rc" "$compare_rc" "$rc" >"$PAIR_STATUS"
exit "$rc"
