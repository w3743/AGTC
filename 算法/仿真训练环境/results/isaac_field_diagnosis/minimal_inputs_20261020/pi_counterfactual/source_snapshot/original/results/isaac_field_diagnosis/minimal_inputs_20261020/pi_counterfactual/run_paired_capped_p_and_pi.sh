#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
NATIVE_STATUS="$OUT_DIR/native_v5_regression_attempt1_exit.json"
FREEZE_MANIFEST="$OUT_DIR/source_freeze_sha256.txt"
PAIR_LOG="$OUT_DIR/paired_counterfactual_attempt1.log"
PAIR_STATUS="$OUT_DIR/paired_counterfactual_attempt1_exit.json"
COMPARISON_PATH="$OUT_DIR/pair_comparison.json"

if [[ ! -f "$NATIVE_STATUS" ]] || ! grep -q '"actual_exit_code":0' "$NATIVE_STATUS"; then
  printf 'The native v5 effort regression must finish with exit 0 first\n' >&2
  exit 2
fi
for path in "$FREEZE_MANIFEST" "$PAIR_LOG" "$PAIR_STATUS" "$COMPARISON_PATH" \
            "$OUT_DIR/capped_p_reference" "$OUT_DIR/pi_candidate" \
            "$OUT_DIR/capped_p_reference.log" "$OUT_DIR/pi_candidate.log" \
            "$OUT_DIR/capped_p_reference_exit.json" "$OUT_DIR/pi_candidate_exit.json"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to overwrite paired counterfactual evidence: %s\n' "$path" >&2
    exit 2
  fi
done

cd "$ROOT_DIR"
freeze_files=(
  isaac_field/env.py
  isaac_field/task.py
  isaac_field/assets.py
  isaac_field/action_contract.py
  scripts/train_isaac_foundation_fresh.py
  tests/test_env_integration.py
  tests/newton_effort_capture.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/turn_response/turn_response_probe.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/turn_response/turn_response_summary.json
  results/isaac_field_diagnosis/minimal_inputs_20261020/effort_limit_smoke/effort_limit_smoke.py
  external/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py
  external/IsaacLab/source/isaaclab/isaaclab/actuators/newton/kernels.py
  external/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py
  external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py
  external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/runtime_pi.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/test_pi_counterfactual_math.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/pi_counterfactual_probe.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/compare_pi_counterfactual_runs.py
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_pi_probe.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_native_v5_regression_attempt1.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_paired_capped_p_and_pi.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_final_static_validation.sh
  results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_review_static_validation.sh
)
sha256sum "${freeze_files[@]}" >"$FREEZE_MANIFEST"

set +e
bash "$OUT_DIR/run_pi_probe.sh" capped_p_reference 0
p_rc=$?
if (( p_rc == 0 )); then
  bash "$OUT_DIR/run_pi_probe.sh" pi_candidate 0.1
  pi_rc=$?
else
  pi_rc=125
fi
if (( p_rc == 0 && pi_rc == 0 )); then
  python3 "$OUT_DIR/compare_pi_counterfactual_runs.py" \
    --root "$OUT_DIR" --output "$COMPARISON_PATH"
  compare_rc=$?
else
  compare_rc=125
fi
set -e
rc=0
if (( p_rc != 0 )); then
  rc=$p_rc
elif (( pi_rc != 0 )); then
  rc=$pi_rc
elif (( compare_rc != 0 )); then
  rc=$compare_rc
fi
printf 'capped_p_exit=%s pi_exit=%s comparison_exit=%s\n' \
  "$p_rc" "$pi_rc" "$compare_rc" >"$PAIR_LOG"
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$PAIR_LOG"
printf '{"capped_p_exit":%s,"pi_exit":%s,"comparison_exit":%s,"actual_exit_code":%s}\n' \
  "$p_rc" "$pi_rc" "$compare_rc" "$rc" >"$PAIR_STATUS"
exit "$rc"
