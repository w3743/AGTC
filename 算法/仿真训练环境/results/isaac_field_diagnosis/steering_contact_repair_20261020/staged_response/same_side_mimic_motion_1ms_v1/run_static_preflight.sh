#!/usr/bin/env bash
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$HERE/../../../../.." && pwd)"
LOG="$HERE/static_cpu_preflight.log"
EXIT_JSON="$HERE/static_cpu_preflight.exit.json"
if [[ -e "$LOG" || -e "$EXIT_JSON" ]]; then
  printf 'Refusing to overwrite prior static preflight records.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
set +e
bash -n "$HERE/run_baseline_1ms_elliptic_1_mu0_8_kf1000.sh" && \
  bash -n "$HERE/run_mimic_v2_1ms_elliptic_1_mu0_8_kf1000.sh" && \
  scripts/isaaclab_python.sh "$HERE/response_runner.py" --static-check >"$LOG" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG"
runner_sha="$(sha256sum "$HERE/response_runner.py" | cut -d' ' -f1)"
probe_sha="$(sha256sum "$ROOT_DIR/results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/staged_motion_response_probe.py" | cut -d' ' -f1)"
hook_sha="$(sha256sum "$ROOT_DIR/results/isaac_field_diagnosis/steering_contact_repair_20261020/same_side_mimic_v2/same_side_mimic_hook.py" | cut -d' ' -f1)"
protocol_sha="$(sha256sum "$ROOT_DIR/results/isaac_field_diagnosis/steering_contact_repair_20261020/same_side_mimic_v2/protocol.json" | cut -d' ' -f1)"
(set -o noclobber; printf '{"actual_exit_code":%s,"gpu_used":false,"physics_dt_s":0.001,"policy_dt_s":0.05,"decimation":50,"expected_policy_rows_per_run":1280,"expected_native_qd_rows_per_run":64000,"expected_last_1s_native_qd_rows_per_case":1000,"expected_last_1s_qd_pair_values_per_case":12000,"runner_sha256":"%s","collector_sha256":"%s","hook_sha256":"%s","protocol_sha256":"%s"}\n' \
  "$rc" "$runner_sha" "$probe_sha" "$hook_sha" "$protocol_sha" >"$EXIT_JSON")
exit "$rc"
