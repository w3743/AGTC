#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/lib/wsl/lib:${PATH:-}"
printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES-<unset>}"
command -v nvidia-smi
nvidia-smi --query-gpu=name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
/opt/venv/bin/python -B -c "import ast,pathlib; ast.parse(pathlib.Path(\"/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/acceptance/20261025_v3_diagnostic60/diagnostic_wrapper.py\").read_text(encoding=\"utf-8\")); print(\"wrapper AST OK\")"
