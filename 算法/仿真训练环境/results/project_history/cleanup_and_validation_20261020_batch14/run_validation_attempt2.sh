#!/usr/bin/env bash
set -uo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OUT/../../.." && pwd)"
cd "$ROOT"
unset RUN_ISAAC_PHYSICS
export CUDA_VISIBLE_DEVICES=
bash scripts/isaaclab_python.sh -m pytest -q > "$OUT/full_cpu_pytest_attempt2.log" 2>&1
pytest_rc=$?
printf '%s\n' "$pytest_rc" > "$OUT/pytest_attempt2.exit_code"
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests > "$OUT/compileall.log" 2>&1
compile_rc=$?
printf '%s\n' "$compile_rc" > "$OUT/compileall.exit_code"
bash_rc=0
count=0
while IFS= read -r -d '' filename; do
  bash -n "$filename" >> "$OUT/bash_syntax.log" 2>&1 || bash_rc=1
  count=$((count + 1))
done < <(find scripts results/isaac_field_diagnosis/steering_contact_repair_20261020 -type f -name '*.sh' -print0)
for filename in results/isaac_field_viewer/detail_local_scale_20261020/run_viewer.sh results/isaac_field_viewer/remove_sidebar_20261020/run_viewer.sh results/isaac_field_improvement/continue_coupled_tracks_20261020_validation/launch_continue.sh results/project_history/cleanup_and_validation_20261020_batch14/run_validation_attempt2.sh; do
  bash -n "$filename" >> "$OUT/bash_syntax.log" 2>&1 || bash_rc=1
  count=$((count + 1))
done
printf '%s\n' "$bash_rc" > "$OUT/bash_syntax.exit_code"
printf '%s\n' "$count" > "$OUT/bash_syntax.count"
sha256sum -c "$OUT/source_hashes_before.sha256" > "$OUT/source_hashes_after.log" 2>&1
hash_rc=$?
printf '%s\n' "$hash_rc" > "$OUT/source_hashes.exit_code"
printf '{"pytest":%s,"compileall":%s,"bash_syntax":%s,"bash_count":%s,"source_hashes":%s}\n' "$pytest_rc" "$compile_rc" "$bash_rc" "$count" "$hash_rc" > "$OUT/validation_attempt2.exit.json"
if [[ "$pytest_rc" -ne 0 || "$compile_rc" -ne 0 || "$bash_rc" -ne 0 || "$hash_rc" -ne 0 ]]; then exit 1; fi
exit 0
