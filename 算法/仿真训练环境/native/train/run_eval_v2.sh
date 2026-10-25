#!/usr/bin/env bash
# Pilot or 16-seed development evaluation of a validated native V2 checkpoint.
# Usage: bash run_eval_v2.sh <checkpoint.pt> [out_dir] [pilot|formal]
set -u
set -o pipefail

CALLER_PWD=$PWD
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
PYTHON_CMD=("${PYTHON:-/opt/venv_mujoco315_smoke/bin/python}")
if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
    echo "ERROR: set PYTHON to the validated native V2 ARM64 environment" >&2
    exit 2
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: run_eval_v2.sh <checkpoint.pt> [out_dir] [pilot|formal]" >&2
    exit 2
fi
CKPT=$1
OUT=${2:-"$SCRIPT_DIR/runs/eval_native_v2_$(date +%Y%m%d_%H%M%S)"}
MODE=${3:-pilot}
case "$MODE" in
    pilot) SEEDS=87010-87013; SEED_LIST=87010,87011,87012,87013 ;;
    formal) SEEDS=87010-87025; SEED_LIST=87010,87011,87012,87013,87014,87015,87016,87017,87018,87019,87020,87021,87022,87023,87024,87025 ;;
    *) echo "ERROR: evaluation mode must be pilot or formal" >&2; exit 2 ;;
esac
case "$CKPT" in /*) ;; *) CKPT="$CALLER_PWD/$CKPT" ;; esac
case "$OUT" in /*) ;; *) OUT="$CALLER_PWD/$OUT" ;; esac

cd -- "$SCRIPT_DIR" || exit 1
export PATH="/usr/lib/wsl/lib:$PATH"
unset CUDA_VISIBLE_DEVICES
mkdir -p -- "$(dirname -- "$OUT")" || exit 1
mkdir -- "$OUT" || exit 1

echo "=== protocol SHA check (V2) ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/protocol_sha.py" --protocol PROTOCOL_NATIVE_V2.md --check || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" check-protocol --protocol-version V2 --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V2.sha256" || exit $?
echo "=== checkpoint contract preflight ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" preflight-checkpoint \
    --checkpoint "$CKPT" --out "$OUT" \
    --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V2.sha256" --protocol-version V2 || exit $?

echo "=== evaluation ($MODE, 1200 steps, official 1ms/50) ==="
"${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/eval_native.py" \
    --checkpoint "$CKPT" --seeds "$SEEDS" --max-steps 1200 \
    --device cuda --physics newton_mujoco_warp --task-backend fast_stop_v2 \
    --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.001 --decimation 50 --out "$OUT" 2>&1 | tee "$OUT/eval.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
EVAL_RC=${PIPE_STATUS[0]}
TEE_RC=${PIPE_STATUS[1]}
echo "EVAL_DIR=$OUT eval_exit=$EVAL_RC tee_exit=$TEE_RC"

if [[ $EVAL_RC -ne 0 ]]; then
    exit "$EVAL_RC"
fi
if [[ $TEE_RC -ne 0 ]]; then
    exit 5
fi
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" check-evaluation \
    --out "$OUT" --seeds "$SEED_LIST"