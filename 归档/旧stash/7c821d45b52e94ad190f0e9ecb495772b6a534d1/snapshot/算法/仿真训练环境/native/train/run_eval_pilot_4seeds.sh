#!/bin/bash
# V1.1 4-seed development evaluation on the formal 1 ms / 50 fast_stop_v2 stack.
# Usage: bash run_eval_pilot_4seeds.sh <checkpoint.pt> [out_dir] [seeds]
set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
CALLER_DIR=$PWD
cd "$SCRIPT_DIR" || exit 1
export PATH="/usr/lib/wsl/lib:$PATH"
unset CUDA_VISIBLE_DEVICES

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("${PYTHON}")
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_CMD=("$VIRTUAL_ENV/bin/python")
elif [[ -x /opt/venv/bin/python ]]; then
    PYTHON_CMD=(/opt/venv/bin/python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python3)")
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python)")
else
    echo "Python not found; set PYTHON to the native-stack interpreter." >&2
    exit 2
fi
run_python() { "${PYTHON_CMD[@]}" "$@"; }

if (($# < 1)); then
    echo "Usage: bash run_eval_pilot_4seeds.sh <checkpoint.pt> [out_dir] [seeds]" >&2
    exit 2
fi
CKPT_ARG=$1
OUT_ARG=${2:-}
SEEDS=${3:-87010-87013}
if [[ "$SEEDS" != "87010-87013" ]]; then
    echo "Pilot V1.1 evaluation is fixed to seeds 87010-87013." >&2
    exit 2
fi
if [[ "$CKPT_ARG" == /* ]]; then CKPT_INPUT=$CKPT_ARG; else CKPT_INPUT="$CALLER_DIR/$CKPT_ARG"; fi
CKPT=$(realpath -e -- "$CKPT_INPUT") || { echo "Checkpoint not found: $CKPT_ARG" >&2; exit 2; }
if [[ -n "$OUT_ARG" ]]; then
    if [[ "$OUT_ARG" == /* ]]; then OUT=$(realpath -m -- "$OUT_ARG"); else OUT=$(realpath -m -- "$CALLER_DIR/$OUT_ARG"); fi
else
    OUT="$SCRIPT_DIR/runs/eval_native_pilot_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT" || exit 1

echo "=== protocol SHA check (V1.1) ==="
run_python protocol_sha.py --check --protocol PROTOCOL_NATIVE_V1.1.md || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}

echo "=== checkpoint contract preflight ==="
run_python - "$CKPT" "$OUT" <<'PY'
import json, math, sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
out = Path(sys.argv[2])
trial = checkpoint.parent
contract_path = trial / "observation_contract.json"
identity_path = trial / "identity.json"
if not checkpoint.is_file() or not contract_path.is_file() or not identity_path.is_file():
    raise SystemExit("checkpoint and same-trial observation_contract.json/identity.json are required")
contract = json.loads(contract_path.read_text(encoding="utf-8"))
identity = json.loads(identity_path.read_text(encoding="utf-8"))
env_identity = identity.get("env", {})
task_backend = contract.get("task_backend")
identity_task_backend = env_identity.get("task_backend")
task_sha = contract.get("task_identity_sha256")
identity_task_sha = env_identity.get("task_identity_sha256")
dt = contract.get("physics_dt_s")
decimation = contract.get("decimation")
experimental = contract.get("physics_experimental")
channels = contract.get("channels", {})

try:
    physics_ok = (math.isclose(float(dt), 0.001, rel_tol=0.0, abs_tol=1e-12)
                  and int(decimation) == 50 and experimental is False)
except (TypeError, ValueError):
    physics_ok = False
identity_ok = (math.isclose(float(env_identity.get("sim_dt_s", -1)), 0.001,
                            rel_tol=0.0, abs_tol=1e-12)
               and int(env_identity.get("decimation", -1)) == 50
               and env_identity.get("experimental_physics") is False)
channels_ok = all(channels.get(name, {}).get("mode") == "real"
                  for name in ("lidar", "depth", "detections", "state", "feedback"))
channels_ok = (channels_ok
               and channels.get("lidar", {}).get("backend") == "warp_kernel"
               and channels.get("depth", {}).get("backend") == "analytic_warp"
               and channels.get("detections", {}).get("backend") == "analytic_detector_warp"
               and channels.get("feedback", {}).get("backend") == "vqf_reference")
if not (physics_ok and identity_ok and task_backend == "fast_stop_v2"
        and identity_task_backend == task_backend and task_sha
        and task_sha == identity_task_sha and channels_ok
        and identity.get("policy", {}).get("policy_type") == "lstm"
        and identity.get("policy", {}).get("hidden_size") == 512):
    raise SystemExit("checkpoint trial is not a matching V1.1 1ms/50 fast_stop_v2 all-real-observation run")
preflight = {
    "checkpoint": str(checkpoint),
    "contract_sha256": contract.get("observation_contract_sha256"),
    "task_identity_sha256": task_sha,
    "task_backend": task_backend,
    "sim_dt_s": dt,
    "decimation": decimation,
    "channels_all_real": channels_ok,
    "checkpoint_contract_check": "eval_native must also match the SHA stored inside the checkpoint",
}
(out / "checkpoint_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
print(json.dumps(preflight))
PY
PREFLIGHT_RC=$?
if ((PREFLIGHT_RC != 0)); then
    echo "CHECKPOINT_PREFLIGHT_FAILED" >&2
    exit "$PREFLIGHT_RC"
fi

echo "=== evaluation (4 seeds, 1200 steps, 1 ms/50) ==="
run_python -u eval_native.py --checkpoint "$CKPT" --seeds "$SEEDS" --max-steps 1200 +    --device cuda --physics newton_xpbd --task-backend fast_stop_v2 +    --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp +    --sim-dt 0.001 --decimation 50 --out "$OUT" 2>&1 | tee "$OUT/eval.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
EVAL_RC=${PIPE_STATUS[0]}
TEE_RC=${PIPE_STATUS[1]:-0}
if ((EVAL_RC == 0 && TEE_RC != 0)); then EVAL_RC=$TEE_RC; fi
if ((EVAL_RC != 0)); then
    echo "EVALUATION_FAILED exit=$EVAL_RC" >&2
    echo "EVAL_DIR=$OUT exit=$EVAL_RC"
    exit "$EVAL_RC"
fi

run_python - "$CKPT" "$OUT" <<'PY'
import json, sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
out = Path(sys.argv[2])
contract = json.loads((checkpoint.parent / "observation_contract.json").read_text(encoding="utf-8"))
evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
expected_sha = contract.get("observation_contract_sha256")
evaluation_sha = evaluation.get("policy", {}).get("observation_contract_sha256")
physics = evaluation.get("physics", {})
task_backend = evaluation.get("task_backend") or evaluation.get("config", {}).get("task_backend")
match = (expected_sha is not None and expected_sha == evaluation_sha
         and physics.get("dt_s") == 0.001 and physics.get("decimation") == 50
         and physics.get("experimental") is False and task_backend == "fast_stop_v2"
         and evaluation.get("config", {}).get("seeds") == list(range(87010, 87014)))
result = {
    "trial_contract_sha256": expected_sha,
    "evaluation_contract_sha256": evaluation_sha,
    "match": bool(match),
    "task_backend": task_backend,
    "physics": physics,
    "capability_metrics_allowed": evaluation.get("capability_metrics_allowed", False),
}
(out / "sha_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
if not match:
    raise SystemExit(4)
PY
CHECK_RC=$?
echo "EVAL_DIR=$OUT exit=$CHECK_RC"
exit "$CHECK_RC"
