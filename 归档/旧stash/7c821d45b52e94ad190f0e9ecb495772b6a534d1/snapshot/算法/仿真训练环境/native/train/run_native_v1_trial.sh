#!/bin/bash
# Formal native V1.1 trial runner: verifies the frozen protocol SHA, creates the
# trial directory, and runs the fixed 1 ms / 50, fast_stop_v2 LSTM recipe.
# Usage: bash run_native_v1_trial.sh <tag> [--live-state-file PATH] [--live-state-interval N]
# Optional env: PYTHON, SIM_DT (must remain 0.001), DECIMATION (must remain 50).
set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
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

SIM_DT=${SIM_DT:-0.001}
DECIMATION=${DECIMATION:-50}
if ! run_python - "$SIM_DT" "$DECIMATION" <<'PY'
import math
import sys
try:
    dt = float(sys.argv[1])
    decimation = int(sys.argv[2])
except (IndexError, TypeError, ValueError):
    raise SystemExit("SIM_DT/DECIMATION must be numeric")
if not math.isclose(dt, 0.001, rel_tol=0.0, abs_tol=1e-12) or decimation != 50:
    raise SystemExit("formal V1.1 trials require SIM_DT=0.001 and DECIMATION=50")
PY
then
    echo "FORMAL_PHYSICS_REJECTED" >&2
    exit 2
fi

TAG=${1:?usage: run_native_v1_trial.sh <tag> [viewer options]}
shift
EXTRA_ARGS=()
while (($#)); do
    case "$1" in
        --live-state-file|--live-state-interval|--live-state-spacing)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                echo "Missing value for viewer option: $1" >&2
                exit 2
            fi
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unsupported extra argument for a frozen formal trial: $1" >&2
            echo "Only live-state viewer options may be added." >&2
            exit 2
            ;;
    esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
TRIAL="runs/native_v1_1_${TAG}_${STAMP}"
mkdir -p "$TRIAL" || exit 1

echo "=== protocol SHA check (V1.1) ==="
run_python protocol_sha.py --check --protocol PROTOCOL_NATIVE_V1.1.md || {
    echo "PROTOCOL_SHA_MISMATCH" >&2
    exit 2
}
cp PROTOCOL_NATIVE_V1.1.md PROTOCOL_NATIVE_V1.1.sha256 PROTOCOL_NATIVE_V1.md PROTOCOL_NATIVE_V1.sha256 "$TRIAL/" || exit 1

ARGS=(--policy lstm --num-envs 64 --num-steps 32 --iterations 300 --seed 0 --threads 8
      --device cuda --physics newton_xpbd --task-backend fast_stop_v2 --lidar warp --feedback vqf
      --depth analytic --detections analytic --detections-warp
      --sim-dt 0.001 --decimation 50 --save-interval 20
      --out "$TRIAL")
printf 'PHYSICS sim_dt=0.001 decimation=50 task_backend=fast_stop_v2\n' | tee "$TRIAL/physics.txt"
printf '%q ' "${ARGS[@]}" "${EXTRA_ARGS[@]}" > "$TRIAL/cli_args.txt"
printf '\n' >> "$TRIAL/cli_args.txt"

echo "=== formal trial train: $TRIAL ==="
run_python -u train_wsl.py "${ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$TRIAL/train.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
TRAIN_RC=${PIPE_STATUS[0]}
TEE_RC=${PIPE_STATUS[1]:-0}
if ((TRAIN_RC == 0 && TEE_RC != 0)); then TRAIN_RC=$TEE_RC; fi

run_python - "$TRIAL" "$TRAIN_RC" <<'PYEOF'
import json, math, sys
from datetime import datetime
from pathlib import Path

trial = Path(sys.argv[1])
train_rc = int(sys.argv[2])
protocol = json.loads((trial / "PROTOCOL_NATIVE_V1.1.sha256").read_text(encoding="utf-8"))
contract_path = trial / "observation_contract.json"
contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
identity_path = trial / "identity.json"
identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
physics_dt = contract.get("physics_dt_s")
decimation = contract.get("decimation")
experimental = contract.get("physics_experimental")
task_backend = contract.get("task_backend")
if task_backend is None and isinstance(contract.get("task"), dict):
    task_backend = contract["task"].get("backend")
profile_enabled = contract.get("profile_enabled")
task_identity_sha = contract.get("task_identity_sha256")
identity_env = identity.get("env", {})
identity_task_sha = identity_env.get("task_identity_sha256")
channels = contract.get("channels", {})
channels_ok = (
    all(channels.get(name, {}).get("mode") == "real"
        for name in ("lidar", "depth", "detections", "state", "feedback"))
    and channels.get("lidar", {}).get("backend") == "warp_kernel"
    and channels.get("depth", {}).get("backend") == "analytic_warp"
    and channels.get("detections", {}).get("backend") == "analytic_detector_warp"
    and channels.get("feedback", {}).get("backend") == "vqf_reference"
)

try:
    physics_ok = (math.isclose(float(physics_dt), 0.001, rel_tol=0.0, abs_tol=1e-12)
                  and int(decimation) == 50 and experimental is False)
except (TypeError, ValueError):
    physics_ok = False
task_ok = task_backend == "fast_stop_v2"
task_identity_ok = bool(task_identity_sha and task_identity_sha == identity_task_sha)
formal_eligible = bool(train_rc == 0 and physics_ok and task_ok and task_identity_ok and contract)
formal_eligible = bool(formal_eligible and channels_ok)
manifest = {
    "protocol_sha256": protocol["protocol_sha256"],
    "protocol_file": "PROTOCOL_NATIVE_V1.1.md",
    "protocol_version": "V1.1",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cli_args_file": "cli_args.txt",
    "exit_code": train_rc,
    "observation_contract_sha256": contract.get("observation_contract_sha256"),
    "capability_metrics_allowed": False,
    "phase": "formal_native_v1_trial" if formal_eligible else "invalid_formal_trial",
    "physics": {"sim_dt": physics_dt, "decimation": decimation,
                "experimental": experimental},
    "task_backend": task_backend,
    "task_identity_sha256": task_identity_sha,
    "channels_all_real": channels_ok,
    "profile_enabled": profile_enabled,
    "formal_eligible": formal_eligible,
    "contract_file": "observation_contract.json" if contract else None,
    "checkpoints": sorted(p.name for p in trial.glob("*.pt")),
}
(trial / "trial_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("TRIAL_MANIFEST " + json.dumps(manifest))
if not formal_eligible:
    raise SystemExit(3)
PYEOF
MANIFEST_RC=$?

FINAL_RC=$TRAIN_RC
if ((FINAL_RC == 0 && MANIFEST_RC != 0)); then FINAL_RC=$MANIFEST_RC; fi
echo "TRIAL_DIR=$TRIAL exit=$FINAL_RC"
exit "$FINAL_RC"
