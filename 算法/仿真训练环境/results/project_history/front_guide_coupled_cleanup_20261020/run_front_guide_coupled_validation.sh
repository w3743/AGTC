#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIELD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$SCRIPT_DIR/run_$STAMP"
if [[ -e "$RUN_DIR" ]]; then
    RUN_DIR="$SCRIPT_DIR/run_${STAMP}_$$"
fi
mkdir -p "$RUN_DIR"
cd "$FIELD_ROOT"

FAILED=0
SUMMARY="$RUN_DIR/validation_summary.txt"
printf 'field_root=%s\nstarted_utc=%s\nenvironment_variables_modified=none\n' \
    "$FIELD_ROOT" "$STAMP" > "$SUMMARY"

write_source_manifest() {
    local target="$1"
    : > "$target"
    find isaac_field scripts tests -type f \
        \( -name '*.py' -o -name '*.sh' \) -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum >> "$target"
    if [[ -f pyproject.toml ]]; then
        sha256sum pyproject.toml >> "$target"
    fi
}

run_step() {
    local name="$1"
    shift
    local log="$RUN_DIR/${name}.log"
    local rc
    if "$@" > "$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" > "$RUN_DIR/${name}.exit_code"
    printf '%s_exit=%s\n' "$name" "$rc" >> "$SUMMARY"
    if [[ "$rc" -ne 0 ]]; then
        FAILED=1
    fi
}

write_source_manifest "$RUN_DIR/source_hashes_before.sha256"

if [[ "${RUN_ISAAC_PHYSICS:-}" == "1" ]]; then
    printf 'refused=RUN_ISAAC_PHYSICS is already 1; runner did not change it\n' >> "$SUMMARY"
    printf 'guard_exit=1\n' >> "$SUMMARY"
    printf '1\n' > "$RUN_DIR/guard.exit_code"
    exit 1
fi

run_step cpu_pytest "$FIELD_ROOT/scripts/isaaclab_python.sh" -m pytest -q
run_step compileall "$FIELD_ROOT/scripts/isaaclab_python.sh" -m compileall -q isaac_field scripts tests

while IFS= read -r -d '' shell_script; do
    script_name="$(basename "$shell_script" .sh)"
    run_step "bash_n_${script_name}" bash -n "$shell_script"
done < <(find scripts -maxdepth 1 -type f -name '*.sh' -print0 | LC_ALL=C sort -z)

write_source_manifest "$RUN_DIR/source_hashes_after.sha256"
if cmp -s "$RUN_DIR/source_hashes_before.sha256" "$RUN_DIR/source_hashes_after.sha256"; then
    printf 'source_test_hash_stability_exit=0\n' >> "$SUMMARY"
    printf '0\n' > "$RUN_DIR/source_test_hash_stability.exit_code"
else
    printf 'source_test_hash_stability_exit=1\n' >> "$SUMMARY"
    printf '1\n' > "$RUN_DIR/source_test_hash_stability.exit_code"
    diff -u "$RUN_DIR/source_hashes_before.sha256" "$RUN_DIR/source_hashes_after.sha256" \
        > "$RUN_DIR/source_hash_diff.log" || true
    FAILED=1
fi

if [[ "$FAILED" -eq 0 ]]; then
    printf 'overall_exit=0\n' >> "$SUMMARY"
    printf '0\n' > "$RUN_DIR/overall.exit_code"
    exit 0
fi

printf 'overall_exit=1\n' >> "$SUMMARY"
printf '1\n' > "$RUN_DIR/overall.exit_code"
exit 1
