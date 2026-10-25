#!/usr/bin/env bash
# Frozen native V1.1 training recipe. Runtime paths are relative to this script.
# Optional: PYTHON, SIM_DT (must be 0.001), DECIMATION (must be 50).
set -u
set -o pipefail

CALLER_PWD=$PWD
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif [[ -x "$SCRIPT_DIR/../../.venv/bin/python" ]]; then
    PYTHON_CMD=("$SCRIPT_DIR/../../.venv/bin/python")
elif [[ -x /opt/venv/bin/python ]]; then
    PYTHON_CMD=(/opt/venv/bin/python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python3)")
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python)")
else
    echo "ERROR: set PYTHON or install python3" >&2
    exit 2
fi
if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON is not executable: ${PYTHON_CMD[0]}" >&2
    exit 2
fi

if [[ $# -lt 1 ]]; then
    echo "usage: run_native_v1_trial.sh <tag> [--live-state-file path] [--live-state-interval n] [--live-state-spacing m]" >&2
    exit 2
fi
TAG=$1
shift
SIM_DT=${SIM_DT:-0.001}
DECIMATION=${DECIMATION:-50}
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" require-official-physics \
    --sim-dt "$SIM_DT" --decimation "$DECIMATION" || exit 2

EXTRA_ARGS=()
declare -A EXTRA_SEEN=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live-state-file|--live-state-interval|--live-state-spacing)
            flag=$1
            shift
            if [[ $# -eq 0 || -z "$1" ]]; then
                echo "ERROR: $flag requires a value" >&2
                exit 2
            fi
            if [[ -n "${EXTRA_SEEN[$flag]+x}" ]]; then
                echo "ERROR: duplicate viewer option $flag" >&2
                exit 2
            fi
            EXTRA_SEEN[$flag]=1
            EXTRA_ARGS+=("$flag" "$1")
            shift
            ;;
        *)
            echo "ERROR: unsupported extra argument: $1" >&2
            echo "Only live-state viewer options may be added to the frozen recipe." >&2
            exit 2
            ;;
    esac
done

cd -- "$SCRIPT_DIR" || exit 1
export PATH="/usr/lib/wsl/lib:$PATH"
unset CUDA_VISIBLE_DEVICES
STAMP=$(date +%Y%m%d_%H%M%S)
TRIAL="runs/native_v1_1_${TAG}_$STAMP"
mkdir -p -- "$TRIAL" || exit 1

echo "=== protocol SHA check (V1.1) ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/protocol_sha.py" --check || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}
cp -- PROTOCOL_NATIVE_V1.1.md PROTOCOL_NATIVE_V1.1.sha256 PROTOCOL_NATIVE_V1.md PROTOCOL_NATIVE_V1.sha256 "$TRIAL/" || exit 1

ARGS=(--policy lstm --lstm-hidden 512 --num-envs 64 --num-steps 32 --iterations 300
      --seed 0 --threads 8 --device cuda --physics newton_xpbd --task-backend fast_stop_v2
      --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp
      --sim-dt 0.001 --decimation 50 --save-interval 20 --nonfinite-limit 0 --out "$TRIAL")
printf 'PHYSICS sim_dt=0.001 decimation=50\n' | tee "$TRIAL/physics.txt"
printf '%q ' "${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/train_wsl.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}" > "$TRIAL/cli_args.txt"
printf '\n' >> "$TRIAL/cli_args.txt"

echo "=== formal trial train: $TRIAL ==="
"${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/train_wsl.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$TRIAL/train.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
TRAIN_RC=${PIPE_STATUS[0]}
TEE_RC=${PIPE_STATUS[1]}

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" write-manifest \
    --trial-dir "$TRIAL" --train-exit "$TRAIN_RC" \
    --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V1.1.sha256"
MANIFEST_RC=$?
echo "TRIAL_DIR=$TRIAL train_exit=$TRAIN_RC tee_exit=$TEE_RC manifest_check=$MANIFEST_RC"

if [[ $TRAIN_RC -ne 0 ]]; then
    exit "$TRAIN_RC"
fi
if [[ $TEE_RC -ne 0 ]]; then
    exit 5
fi
exit "$MANIFEST_RC"
