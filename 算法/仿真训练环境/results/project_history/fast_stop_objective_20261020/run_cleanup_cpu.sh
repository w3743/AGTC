#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020"
LABEL="${1:-attempt1}"
case "$LABEL" in attempt[0-9]*) ;; *) exit 79 ;; esac
cd "$ROOT" || exit 80
if [[ -e "$OUT/cpu_${LABEL}.log" ]]; then exit 81; fi
export CUDA_VISIBLE_DEVICES=""
unset RUN_ISAAC_PHYSICS
find isaac_field scripts tests -type f \( -name '*.py' -o -name '*.html' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum > "$OUT/source_test_hashes_before_${LABEL}.sha256"
bash scripts/isaaclab_python.sh -m pytest -q tests > "$OUT/cpu_${LABEL}.log" 2>&1
tests_rc=$?
printf '%s\n' "$tests_rc" > "$OUT/cpu_${LABEL}.exit_code"
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts > "$OUT/compileall_${LABEL}.log" 2>&1
compile_rc=$?
printf '%s\n' "$compile_rc" > "$OUT/compileall_${LABEL}.exit_code"
bash_rc=0
while IFS= read -r -d '' script; do
  bash -n "$script" >> "$OUT/bash_syntax_${LABEL}.log" 2>&1 || bash_rc=1
done < <(find scripts "$OUT" -maxdepth 1 -type f -name '*.sh' -print0)
printf '%s\n' "$bash_rc" > "$OUT/bash_syntax_${LABEL}.exit_code"
find isaac_field scripts tests -type f \( -name '*.py' -o -name '*.html' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum > "$OUT/source_test_hashes_after_${LABEL}.sha256"
cmp "$OUT/source_test_hashes_before_${LABEL}.sha256" "$OUT/source_test_hashes_after_${LABEL}.sha256" > "$OUT/source_stability_${LABEL}.log" 2>&1
hash_rc=$?
printf '{"pytest_exit":%s,"compileall_exit":%s,"bash_syntax_exit":%s,"source_test_stability_exit":%s,"source":"parent_shell_waited_for_children"}\n' "$tests_rc" "$compile_rc" "$bash_rc" "$hash_rc" > "$OUT/cpu_${LABEL}_actual_exit.json"
if [[ "$tests_rc" -ne 0 ]]; then exit "$tests_rc"; fi
if [[ "$compile_rc" -ne 0 ]]; then exit "$compile_rc"; fi
if [[ "$hash_rc" -ne 0 ]]; then exit "$hash_rc"; fi
exit "$bash_rc"
