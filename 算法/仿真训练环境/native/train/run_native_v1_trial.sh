#!/bin/bash
# Formal native V1.1 trial runner: verifies the frozen protocol SHA (V1.1), creates the
# trial dir, copies the protocol, runs the formal LSTM recipe on the official 1 ms/50
# physics, and writes trial_manifest.json.
# Usage: bash run_native_v1_trial.sh <tag> [extra train_wsl args...]
# Optional env: SIM_DT (default 0.001), DECIMATION (default 50).
set -u
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train || exit 1
TAG=${1:?usage: run_native_v1_trial.sh <tag> [extra args]}
shift || true
STAMP=$(date +%Y%m%d_%H%M%S)
TRIAL="runs/native_v1_1_${TAG}_${STAMP}"
mkdir -p "$TRIAL"

echo "=== protocol SHA check (V1.1) ==="
/opt/venv/bin/python protocol_sha.py --check || { echo "PROTOCOL_SHA_MISMATCH"; exit 2; }
cp PROTOCOL_NATIVE_V1.1.md PROTOCOL_NATIVE_V1.1.sha256 PROTOCOL_NATIVE_V1.md PROTOCOL_NATIVE_V1.sha256 "$TRIAL/"

ARGS=(--policy lstm --num-envs 64 --num-steps 32 --iterations 300 --seed 0 --threads 8
      --device cuda --physics newton_xpbd --lidar warp --feedback vqf
      --depth analytic --detections analytic --detections-warp
      --sim-dt "${SIM_DT:-0.001}" --decimation "${DECIMATION:-50}" --profile --save-interval 20
      --out "$TRIAL")
echo "PHYSICS sim_dt=${SIM_DT:-0.001} decimation=${DECIMATION:-50}" | tee "$TRIAL/physics.txt"
echo "$(printf '%q ' "${ARGS[@]}" "$@")" > "$TRIAL/cli_args.txt"

echo "=== formal trial train: $TRIAL ==="
/opt/venv/bin/python -u train_wsl.py "${ARGS[@]}" "$@" 2>&1 | tee "$TRIAL/train.log" \
    | grep -E 'IDENTITY|ITER|SUMMARY|Error|Traceback' | head -8
RC=${PIPESTATUS[0]}

/opt/venv/bin/python - "$TRIAL" "$RC" <<'PYEOF'
import json, os, sys
from datetime import datetime
from pathlib import Path
trial = Path(sys.argv[1]); rc = int(sys.argv[2])
protocol = json.loads((trial / "PROTOCOL_NATIVE_V1.1.sha256").read_text(encoding="utf-8"))
contract_path = trial / "observation_contract.json"
contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
manifest = {
    "protocol_sha256": protocol["protocol_sha256"],
    "protocol_file": "PROTOCOL_NATIVE_V1.1.md",
    "protocol_version": "V1.1",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cli_args_file": "cli_args.txt",
    "exit_code": rc,
    "observation_contract_sha256": contract.get("observation_contract_sha256"),
    "capability_metrics_allowed": False,
    "phase": "formal_native_v1_trial",
    "physics": {"sim_dt": float(os.environ.get("SIM_DT", "0.001")),
                "decimation": int(os.environ.get("DECIMATION", "50")),
                "experimental": False},
    "checkpoints": sorted(p.name for p in trial.glob("*.pt")),
}
(trial / "trial_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("TRIAL_MANIFEST " + json.dumps(manifest))
PYEOF
echo "TRIAL_DIR=$TRIAL exit=$RC"
