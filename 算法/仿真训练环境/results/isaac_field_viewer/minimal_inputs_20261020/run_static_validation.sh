#!/usr/bin/env bash
set -uo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$FIELD_ROOT"

attempt=1
while [[ -e "$EVIDENCE_DIR/static_validation_attempt${attempt}.log" ]]; do
    attempt=$((attempt + 1))
done
log="$EVIDENCE_DIR/static_validation_attempt${attempt}.log"

set +e
{
    CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts
    compile_exit=$?
    printf 'compileall_exit_code=%d\n' "$compile_exit"

    aggregate_exit=0
    for script in scripts/install_isaaclab.sh scripts/isaaclab_python.sh "$EVIDENCE_DIR"/*.sh; do
        bash -n "$script"
        script_exit=$?
        printf 'bash_n_exit_code=%d script=%s\n' "$script_exit" "$script"
        if [[ $script_exit -ne 0 ]]; then
            aggregate_exit=$script_exit
        fi
    done
    if [[ $compile_exit -ne 0 ]]; then
        aggregate_exit=$compile_exit
    fi
    printf 'static_validation_exit_code=%d\n' "$aggregate_exit"
    exit "$aggregate_exit"
} >"$log" 2>&1
result=$?
set -e
printf 'static_validation_exit_code=%d log=%s\n' "$result" "$log"
exit "$result"
