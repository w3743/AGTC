#!/usr/bin/env bash
set +e
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
BEFORE_JSON_SHA="$(sha256sum "$OUT/before_frame.json" | awk '{print $1}')"
AFTER_JSON_SHA="$(sha256sum "$OUT/after_frame.json" | awk '{print $1}')"
bash scripts/isaaclab_python.sh results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/compare_raw_exports_cpu.py > "$OUT/raw_comparison_cpu.log" 2>&1
rc=$?
REPORT_SHA=""
if [[ -f "$OUT/raw_comparison.json" ]]; then REPORT_SHA="$(sha256sum "$OUT/raw_comparison.json" | awk '{print $1}')"; fi
SCRIPT_SHA="$(sha256sum "$OUT/compare_raw_exports_cpu.py" | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum "$OUT/raw_export_copy_manifest.json" | awk '{print $1}')"
BEFORE_JSON_AFTER="$(sha256sum "$OUT/before_frame.json" | awk '{print $1}')"
AFTER_JSON_AFTER="$(sha256sum "$OUT/after_frame.json" | awk '{print $1}')"
printf '{"timestamp_utc":"%s","command":"CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/compare_raw_exports_cpu.py","actual_exit_code":%d,"status":"%s","cpu_only":true,"gpu_used":false,"raw_comparison_sha256":"%s","script_sha256":"%s","copy_manifest_sha256":"%s","preexisting_before_frame_json_sha256_before":"%s","preexisting_before_frame_json_sha256_after":"%s","preexisting_after_frame_json_sha256_before":"%s","preexisting_after_frame_json_sha256_after":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$(if [[ $rc -eq 0 ]]; then printf passed; else printf failed; fi)" "$REPORT_SHA" "$SCRIPT_SHA" "$MANIFEST_SHA" "$BEFORE_JSON_SHA" "$BEFORE_JSON_AFTER" "$AFTER_JSON_SHA" "$AFTER_JSON_AFTER" > "$OUT/raw_comparison_cpu.exit.json"
cat "$OUT/raw_comparison_cpu.exit.json"
exit "$rc"