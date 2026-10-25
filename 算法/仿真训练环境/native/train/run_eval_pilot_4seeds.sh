#!/bin/bash
# Pilot/pre-protocol 4-seed evaluation (87010-87013) with contract-SHA verification.
# Usage: bash run_eval_pilot_4seeds.sh <checkpoint.pt> [out_dir]
# NOTE: eval must use the SAME backends as training; defaults below mirror the
# protocol formal recipe (5ms/10, newton_xpbd, warp lidar, vqf, analytic depth/detections).
set -u
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train || exit 1
CKPT=${1:?usage: run_eval_pilot_4seeds.sh <checkpoint.pt> [out_dir]}
OUT=${2:-runs/eval_native_pilot_$(date +%Y%m%d_%H%M%S)}
SEEDS=${3:-87010-87013}
mkdir -p "$OUT"

echo "=== protocol SHA check ==="
/opt/venv/bin/python protocol_sha.py --check || { echo "PROTOCOL_SHA_MISMATCH"; exit 2; }

echo "=== checkpoint identity ==="
/opt/venv/bin/python - "$CKPT" <<'PYEOF'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
meta = {"policy_type": ck.get("policy_type"), "hidden_size": ck.get("hidden_size"),
        "iteration": ck.get("iteration"),
        "observation_contract_sha256": ck.get("observation_contract_sha256")}
open("pilot_checkpoint_identity.tmp.json", "w", encoding="utf-8").write(json.dumps(meta))
print(json.dumps(meta))
PYEOF

echo "=== evaluation (4 seeds, 1200 steps, 5ms/10) ==="
/opt/venv/bin/python -u eval_native.py \
    --checkpoint "$CKPT" --seeds "$SEEDS" --max-steps 1200 \
    --device cuda --physics newton_xpbd --lidar warp --feedback vqf \
    --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.005 --decimation 10 \
    --out "$OUT" 2>&1 | tee "$OUT/eval.log" | grep -E 'EVAL_SUMMARY|EVAL_ROWS|EVAL_OUT|Error|Traceback'

echo "=== contract SHA comparison (training checkpoint vs evaluation) ==="
/opt/venv/bin/python - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ck = json.loads(Path("pilot_checkpoint_identity.tmp.json").read_text(encoding="utf-8"))
ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
ev_sha = ev["policy"]["observation_contract_sha256"]
match = ck.get("observation_contract_sha256") == ev_sha
verdict = {
    "checkpoint_sha": ck.get("observation_contract_sha256"),
    "evaluation_sha": ev_sha,
    "match": bool(match),
    "policy_type": ck.get("policy_type"),
    "evaluation_policy_type": ev["policy"].get("policy_type"),
    "physics": ev["physics"],
    "protocol_sha256": json.loads(Path("PROTOCOL_NATIVE_V1.sha256").read_text(encoding="utf-8"))["protocol_sha256"],
    "capability_metrics_allowed": ev["capability_metrics_allowed"],
}
(out / "sha_check.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(verdict))
PYEOF
rm -f pilot_checkpoint_identity.tmp.json
echo "EVAL_DIR=$OUT"
