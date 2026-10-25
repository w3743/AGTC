#!/usr/bin/env bash
# Sixteen-seed native V1.1 evaluation of a validated formal checkpoint.
# Usage: bash run_eval_16seeds.sh <checkpoint.pt> [out_dir] [87010-87025]
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

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: run_eval_16seeds.sh <checkpoint.pt> [out_dir] [87010-87025]" >&2
    exit 2
fi
CKPT=$1
OUT=${2:-"$SCRIPT_DIR/runs/eval_native_v1_1_$(date +%Y%m%d_%H%M%S)"}
SEEDS=${3:-87010-87025}
if [[ "$SEEDS" != "87010-87025" ]]; then
    echo "ERROR: formal evaluation seeds are frozen at 87010-87025" >&2
    exit 2
fi
case "$CKPT" in /*) ;; *) CKPT="$CALLER_PWD/$CKPT" ;; esac
case "$OUT" in /*) ;; *) OUT="$CALLER_PWD/$OUT" ;; esac

cd -- "$SCRIPT_DIR" || exit 1
export PATH="/usr/lib/wsl/lib:$PATH"
unset CUDA_VISIBLE_DEVICES
mkdir -p -- "$OUT" || exit 1

echo "=== protocol SHA check (V1.1) ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/protocol_sha.py" --check || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}
echo "=== checkpoint contract preflight ==="
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/trial_checks.py" preflight-checkpoint \
    --checkpoint "$CKPT" --out "$OUT" \
    --protocol-sha "$SCRIPT_DIR/PROTOCOL_NATIVE_V1.1.sha256" || exit $?

echo "=== evaluation (16 seeds, 1200 steps each, official 1ms/50) ==="
"${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/eval_native.py" \
    --checkpoint "$CKPT" --seeds "$SEEDS" --max-steps 1200 \
    --device cuda --physics newton_xpbd --task-backend fast_stop_v2 \
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
    --out "$OUT" --seeds 87010,87011,87012,87013,87014,87015,87016,87017,87018,87019,87020,87021,87022,87023,87024,87025