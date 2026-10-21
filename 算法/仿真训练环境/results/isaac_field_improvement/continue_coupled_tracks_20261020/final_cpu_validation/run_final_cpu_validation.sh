#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
TRIAL="$ROOT/results/isaac_field_improvement/continue_coupled_tracks_20261020"
OUT="$TRIAL/final_cpu_validation"
cd "$ROOT" || exit 2

for name in final_cpu_validation.json final_cpu_validation.log final_cpu_validation.exit.json; do
  if [[ -e "$OUT/$name" ]]; then
    printf 'Refusing to overwrite existing validation evidence: %s\n' "$OUT/$name" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=""
bash scripts/isaaclab_python.sh "$OUT/validate_final_checkpoint_cpu.py" > "$OUT/final_cpu_validation.log" 2>&1
rc=$?
report_sha=""
if [[ -f "$OUT/final_cpu_validation.json" ]]; then report_sha="$(sha256sum "$OUT/final_cpu_validation.json" | awk '{print $1}')"; fi
validator_sha="$(sha256sum "$OUT/validate_final_checkpoint_cpu.py" | awk '{print $1}')"
checkpoint_sha=""
if [[ -f "$TRIAL/stage_04_turn90.zip" ]]; then checkpoint_sha="$(sha256sum "$TRIAL/stage_04_turn90.zip" | awk '{print $1}')"; fi
protocol_sha="$(sha256sum "$TRIAL/protocol.json" | awk '{print $1}')"
manifest_sha=""
if [[ -f "$TRIAL/checkpoint_manifest.json" ]]; then manifest_sha="$(sha256sum "$TRIAL/checkpoint_manifest.json" | awk '{print $1}')"; fi
baseline_sha="$(sha256sum "$TRIAL/parent_final_baseline.zip" | awk '{print $1}')"
printf '{"timestamp_utc":"%s","validation_command":"CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh results/isaac_field_improvement/continue_coupled_tracks_20261020/final_cpu_validation/validate_final_checkpoint_cpu.py","actual_exit_code":%d,"status":"%s","cpu_only":true,"gpu_used":false,"report_sha256":"%s","validator_sha256":"%s","checkpoint_sha256":"%s","protocol_sha256":"%s","manifest_sha256":"%s","baseline_sha256":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$(if [[ $rc -eq 0 ]]; then printf passed; else printf failed; fi)" \
  "$report_sha" "$validator_sha" "$checkpoint_sha" "$protocol_sha" "$manifest_sha" "$baseline_sha" \
  > "$OUT/final_cpu_validation.exit.json"
cat "$OUT/final_cpu_validation.exit.json"
exit "$rc"
