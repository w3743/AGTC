#!/usr/bin/env bash
set -Eeuo pipefail

# This launcher requires the root agent's explicit source-freeze acknowledgement.
if [[ "${1:-}" != "--after-root-source-freeze" || "$#" -ne 1 ]]; then
    printf 'Refusing to run. After the root agent freezes the source set, invoke with: %s --after-root-source-freeze\n' "$0" >&2
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
TRIAL_DIR="$REPO_ROOT/results/isaac_field_improvement/fresh_side_pi_20261020"
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

# Prepare creates the trial directory itself; never pre-create or reuse it.
"$PYTHON_WRAPPER" "$TRAINER" --prepare --output "$TRIAL_DIR"

# Run the same frozen runner preflight explicitly before launch. The launch path
# runs it once more immediately before training.
if [[ -e "$TRIAL_DIR/preflight.log" || -e "$TRIAL_DIR/notes.md" ]]; then
    printf 'Refusing to overwrite a trial note or preflight record.\n' >&2
    exit 73
fi
"$PYTHON_WRAPPER" -c 'import json,sys; from pathlib import Path; from scripts.train_isaac_foundation_fresh import _sha256,preflight; p=Path(sys.argv[1]); protocol=preflight(p); print(json.dumps({"preflight":"passed","protocol_sha256":_sha256(p/"protocol.json"),"source_files":len(protocol["source_snapshot"]["files"]),"timesteps_total":protocol["training"]["timesteps_total"]},ensure_ascii=False,indent=2))' "$TRIAL_DIR" >"$TRIAL_DIR/preflight.log" 2>&1
cat >"$TRIAL_DIR/notes.md" <<'NOTES'
# Fresh side-PI training notes

This trial uses a new random RecurrentPPO initialization and the current frozen sensor-minimal-v6 policy contract. It does not resume or transfer any earlier checkpoint. The four-stage 196,608-step curriculum, formal zero-step baseline, 29-checkpoint schedule, and paired development evaluation are unchanged.

The simulation drive uses native per-wheel P=0.04 and a default side-integral Ki=0.1, updated every policy step using the actual `step_dt`; Newton's combined per-wheel effort clamp remains 2 Nm. Ki, the side-mean aggregation, and all proxy motor parameters are uncalibrated simulation assumptions, not real-vehicle controller settings.

The isolated five-second braking probe passed the final zero-command and terminal velocity gates, but the PI run showed about 0.0729 m reverse-speed integral and reached about -0.130 m/s, compared with near-zero reverse integral for P-only. This trial tests learning under the changed execution loop; it does not imply improved braking or real-vehicle safety.

Training uses an empty field (`plants=0`). Its free-space and 16-plant paired development results are not blind validation and cannot be described as a safety or field-capability improvement. The 830xx seed range remains unused and unobserved.
NOTES

if [[ -e "$EXIT_FILE" || -L "$EXIT_FILE" || -e "$TRIAL_DIR/launch_stdout.log" || -e "$TRIAL_DIR/launch_stderr.log" ]]; then
    printf 'Refusing to overwrite a process record or launch log.\n' >&2
    exit 73
fi
"$PYTHON_WRAPPER" "$TRAINER" --launch --output "$TRIAL_DIR" --device cuda:0 \
    >"$TRIAL_DIR/launch_stdout.log" 2>"$TRIAL_DIR/launch_stderr.log" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_exit_code=$?
set -e
completed_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
completed_at_beijing="$(TZ=Asia/Shanghai date +'%Y-%m-%dT%H:%M:%S%:z')"
set -o noclobber
printf '{\n  "schema": "isaac-field-process-exit-v1",\n  "trial": "fresh_side_pi_20261020",\n  "runner_pid": %s,\n  "exit_code": %s,\n  "completed_at_utc": "%s",\n  "completed_at_beijing": "%s"\n}\n' \
    "$runner_pid" "$runner_exit_code" "$completed_at_utc" "$completed_at_beijing" \
    >"$EXIT_FILE"
set +o noclobber
printf 'Fresh PI training runner exited with code %s; process record: %s\n' "$runner_exit_code" "$EXIT_FILE"
exit "$runner_exit_code"