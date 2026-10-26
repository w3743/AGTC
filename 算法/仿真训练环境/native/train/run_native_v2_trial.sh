#!/usr/bin/env bash
# Fixed native V2 training recipe; an unfrozen protocol is rejected. Runtime paths are relative to this script.
# Optional: PYTHON, SIM_DT (must be 0.001), DECIMATION (must be 50).
set -u
set -o pipefail

CALLER_PWD=$PWD
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
PYTHON_CMD=("${PYTHON:-/opt/venv_mujoco315_smoke/bin/python}")
if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
    echo "ERROR: set PYTHON to the validated native V2 ARM64 environment" >&2
    exit 2
fi
"${PYTHON_CMD[@]}" -c 'from importlib.metadata import version; expected={"newton":"1.7.0","warp-lang":"1.18.0","mujoco":"3.15.0","mujoco-warp":"3.15.0"}; actual={k:version(k) for k in expected}; print(actual); assert actual==expected, "native V2 dependency versions differ"' || exit 2

if [[ $# -lt 1 ]]; then
    echo "usage: run_native_v2_trial.sh <tag> [--live-state-file path] [--live-state-interval n] [--live-state-spacing m]" >&2
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
TRIAL="runs/native_v2_${TAG}_$STAMP"
mkdir -p -- runs || exit 1
mkdir -- "$TRIAL" || exit 1

echo "=== protocol SHA check (V2) ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/protocol_sha.py" --protocol PROTOCOL_NATIVE_V2.md --check || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" check-protocol --protocol-version V2 --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V2.sha256" || exit $?
cp -- PROTOCOL_NATIVE_V2.md PROTOCOL_NATIVE_V2.sha256 "$TRIAL/" || exit 1

ARGS=(--policy lstm --lstm-hidden 512 --num-envs 64 --num-steps 32 --iterations 300
      --seed 0 --threads 8 --epochs 4 --lr 0.0003 --device cuda --physics newton_mujoco_warp --task-backend fast_stop_v2
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
    --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V2.sha256" --protocol-version V2
MANIFEST_RC=$?
echo "TRIAL_DIR=$TRIAL train_exit=$TRAIN_RC tee_exit=$TEE_RC manifest_check=$MANIFEST_RC"

if [[ $TRAIN_RC -ne 0 ]]; then
    exit "$TRAIN_RC"
fi
if [[ $TEE_RC -ne 0 ]]; then
    exit 5
fi
exit "$MANIFEST_RC"
