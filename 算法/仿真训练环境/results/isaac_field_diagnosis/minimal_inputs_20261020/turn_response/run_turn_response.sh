#!/usr/bin/env bash
set -o pipefail

OUT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
FIELD_ROOT="$(cd -- "$OUT_DIR/../../../../" && pwd)"
LOG_PATH="$OUT_DIR/turn_response_run.log"
STATUS_PATH="$OUT_DIR/process_exit.json"

if [[ "$WSL_DISTRO_NAME" != "Ubuntu-24.04" ]]; then
    printf 'Run this diagnostic from the Ubuntu-24.04 WSL distribution.\n' >&2
    exit 2
fi
if [[ "$RUN_ISAAC_PHYSICS" != "1" ]]; then
    printf 'Refusing native Newton diagnostic without RUN_ISAAC_PHYSICS=1.\n' >&2
    exit 2
fi
for output in "$LOG_PATH" "$STATUS_PATH" "$OUT_DIR/turn_response_samples.jsonl" "$OUT_DIR/turn_response_summary.json"; do
    if [[ -e "$output" ]]; then
        printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
        exit 2
    fi
done

cd -- "$FIELD_ROOT"
printf 'Native Newton turn-response diagnostic started: %s\n' "$(date -Is)" > "$LOG_PATH"
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh \
    "$OUT_DIR/turn_response_probe.py" >> "$LOG_PATH" 2>&1
probe_exit=$?
printf '\nprobe_exit=%d\n' "$probe_exit" >> "$LOG_PATH"
printf '{"schema":"isaac-field-native-turn-response-process-v1","finished_at":"%s","exit_code":%d}\n' \
    "$(date -Is)" "$probe_exit" > "$STATUS_PATH"
tail -n 18 "$LOG_PATH"
exit "$probe_exit"
