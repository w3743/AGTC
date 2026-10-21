#!/usr/bin/env bash
set +e
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_improvement/fresh_coupled_tracks_20261020/final_cpu_validation"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
set +e
bash scripts/isaaclab_python.sh "$OUT/validate_final_checkpoint_cpu.py" > "$OUT/final_cpu_validation.log" 2>&1
rc=$?
report_sha=""
if [[ -f "$OUT/final_cpu_validation.json" ]]; then report_sha="$(sha256sum "$OUT/final_cpu_validation.json" | awk '{print $1}')"; fi
validator_sha="$(sha256sum "$OUT/validate_final_checkpoint_cpu.py" | awk '{print $1}')"
checkpoint_sha="$(sha256sum "$ROOT/results/isaac_field_improvement/fresh_coupled_tracks_20261020/stage_04_turn90.zip" | awk '{print $1}')"
protocol_sha="$(sha256sum "$ROOT/results/isaac_field_improvement/fresh_coupled_tracks_20261020/protocol.json" | awk '{print $1}')"
baseline_sha="$(sha256sum "$ROOT/results/isaac_field_improvement/fresh_coupled_tracks_20261020/untrained_baseline.zip" | awk '{print $1}')"
printf '{"timestamp_utc":"%s","validation_command":"CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh results/isaac_field_improvement/fresh_coupled_tracks_20261020/final_cpu_validation/validate_final_checkpoint_cpu.py","actual_exit_code":%d,"status":"%s","cpu_only":true,"gpu_used":false,"report_sha256":"%s","validator_sha256":"%s","checkpoint_sha256":"%s","protocol_sha256":"%s","baseline_sha256":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$(if [[ $rc -eq 0 ]]; then printf passed; else printf failed; fi)" "$report_sha" "$validator_sha" "$checkpoint_sha" "$protocol_sha" "$baseline_sha" > "$OUT/final_cpu_validation.exit.json"
cat "$OUT/final_cpu_validation.exit.json"
exit "$rc"