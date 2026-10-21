#!/usr/bin/env bash
set -uo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 || ( "$1" != focused && "$1" != full ) ]]; then
    printf 'usage: %s focused|full\n' "$0" >&2
    exit 2
fi

cd "$FIELD_ROOT"
suite="$1"
attempt=1
while [[ -e "$EVIDENCE_DIR/${suite}_cpu_pytest_attempt${attempt}.log" ]]; do
    attempt=$((attempt + 1))
done
log="$EVIDENCE_DIR/${suite}_cpu_pytest_attempt${attempt}.log"

set +e
if [[ "$suite" == focused ]]; then
    CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q \
        tests/test_minimal_input_contract.py \
        tests/test_task.py \
        tests/test_isaac_field_policy.py \
        tests/test_policy_inspection.py \
        tests/test_fresh_foundation.py \
        tests/test_isaac_field_learning.py \
        tests/test_activation_maps.py \
        tests/test_network_2d.py \
        tests/test_network_panel.py \
        tests/test_isaac_field_viewer.py \
        tests/test_env_integration.py >"$log" 2>&1
else
    CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q >"$log" 2>&1
fi
test_exit=$?
set -e

printf '\npytest_exit_code=%d\n' "$test_exit" >>"$log"
printf 'suite=%s pytest_exit_code=%d log=%s\n' "$suite" "$test_exit" "$log"
exit "$test_exit"
