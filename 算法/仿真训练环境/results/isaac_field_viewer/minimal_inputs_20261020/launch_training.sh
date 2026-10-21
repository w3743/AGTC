#!/usr/bin/env bash
set -Eeuo pipefail

# This script deliberately requires an explicit acknowledgement after the root
# agent freezes the source set. It only creates the fresh, fixed trial below.
if [[ "${1:-}" != "--after-root-source-freeze" || "$#" -ne 1 ]]; then
    printf 'Refusing to run. After the root agent confirms the source freeze, invoke with: %s --after-root-source-freeze\n' "$0" >&2
    exit 64
fi

if [[ "${WSL_DISTRO_NAME:-}" != "Ubuntu-24.04" ]]; then
    printf 'Run this launcher from the Ubuntu-24.04 WSL distribution.\n' >&2
    exit 65
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    printf 'This training protocol expects x86_64 WSL.\n' >&2
    exit 65
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
TRIAL_DIR="$REPO_ROOT/results/isaac_field_improvement/fresh_minimal_inputs_20261020"
PYTHON_WRAPPER="$REPO_ROOT/scripts/isaaclab_python.sh"
TRAINER="$REPO_ROOT/scripts/train_isaac_foundation_fresh.py"
EXIT_FILE="$TRIAL_DIR/process_exit.json"

if [[ -e "$TRIAL_DIR" || -L "$TRIAL_DIR" ]]; then
    printf 'Refusing to reuse or overwrite an existing trial: %s\n' "$TRIAL_DIR" >&2
    exit 73
fi
if [[ ! -x "$PYTHON_WRAPPER" || ! -f "$TRAINER" ]]; then
    printf 'Required runtime wrapper or fresh runner is missing.\n' >&2
    exit 66
fi

cd -- "$REPO_ROOT"
# prepare creates the output directory itself and refuses an existing path.
# If prepare fails, set -e stops here and training is not launched.
"$PYTHON_WRAPPER" "$TRAINER" --prepare --output "$TRIAL_DIR"

# The runner's wrapper uses exec, so this child PID is the actual Python runner PID.
# Save output only after prepare has created the fresh trial directory.
if [[ -e "$EXIT_FILE" || -L "$EXIT_FILE" ]]; then
    printf 'Refusing to overwrite an existing process-exit record: %s\n' "$EXIT_FILE" >&2
    exit 73
fi
"$PYTHON_WRAPPER" "$TRAINER" --launch --output "$TRIAL_DIR" \
    >"$TRIAL_DIR/launch_stdout.log" 2>"$TRIAL_DIR/launch_stderr.log" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_exit_code=$?
set -e

completed_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
completed_at_beijing="$(TZ=Asia/Shanghai date +'%Y-%m-%dT%H:%M:%S%:z')"
# noclobber provides a final guard against replacing another process's record.
set -o noclobber
printf '{\n  "schema": "isaac-field-process-exit-v1",\n  "trial": "fresh_minimal_inputs_20261020",\n  "runner_pid": %s,\n  "exit_code": %s,\n  "completed_at_utc": "%s",\n  "completed_at_beijing": "%s"\n}\n' \
    "$runner_pid" "$runner_exit_code" "$completed_at_utc" "$completed_at_beijing" \
    >"$EXIT_FILE"
set +o noclobber
printf 'Fresh training runner exited with code %s; process record: %s\n' "$runner_exit_code" "$EXIT_FILE"
exit "$runner_exit_code"
