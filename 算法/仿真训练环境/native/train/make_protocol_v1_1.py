"""Create PROTOCOL_NATIVE_V1.1.md from the frozen V1 (V1 stays as an archive).

Changes per Lead decision (2026-10-25):
  * official native trial default physics = 1 ms / decimation 50;
  * 5 ms/10 becomes experimental-only and requires solver tuning + re-validation;
  * record the 97-iter 5 ms instability evidence that motivated the change.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "PROTOCOL_NATIVE_V1.md"
V11 = HERE / "PROTOCOL_NATIVE_V1.1.md"

text = V1.read_text(encoding="utf-8")

replacements = [
    ("# PROTOCOL_NATIVE_V1 —— WSL 原生栈正式试验协议（FROZEN）",
     "# PROTOCOL_NATIVE_V1.1 —— WSL 原生栈正式试验协议（FROZEN）\n\n"
     "> 版本：V1.1（2026-10-25 二次冻结）。**取代 V1**；V1 原文与旧 SHA 文件作为归档保留，不删除、不覆盖。"),
    ("## 3. 物理配置\n\n- **native 正式实验默认**：`--sim-dt 0.005 --decimation 10`（5 ms 物理）；`policy_dt` 恒为 0.05 s。\n"
     "  contract 标 `physics_experimental=true`，结果与 1 ms/50 结果**分开记录**。\n"
     "- **CLI 默认不变**：`train_wsl.py` 不传 `--sim-dt/--decimation` 时仍为 1 ms/50；正式 trial 必须显式传参。",
     "## 3. 物理配置\n\n"
     "- **native 正式实验默认（V1.1）**：`--sim-dt 0.001 --decimation 50`（1 ms 物理）；`policy_dt` 恒为 0.05 s。\n"
     "  正式 trial 必须显式传参并以 `physics_experimental=false` 记录。\n"
     "- **5 ms/10 降级为 experimental-only**：仅用于吞吐/低动作强度实验，**不得**作为正式 trial 物理；\n"
     "  需先调 XPBD（提高 solver iterations 或减小 substep）并按 §17 证据复验，未达标前进不了正式 trial。\n"
     "- **CLI 默认不变**：`train_wsl.py` 不传 `--sim-dt/--decimation` 时仍是 1 ms/50。"),
    ("- 正式 trial 目录：`train/runs/native_v1_<tag>_<YYYYmmdd_HHMMSS>/`。",
     "- 正式 trial 目录：`train/runs/native_v1_1_<tag>_<YYYYmmdd_HHMMSS>/`（V1.1 正式 trial）。"),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f"V1.1 transform: pattern not found:\n{old[:120]}")
    text = text.replace(old, new, 1)

# decision table: append V1.1 items
marker = "| 6 | capability 开启条件 |"
idx = text.find(marker)
if idx < 0:
    raise SystemExit("V1.1 transform: decision table marker not found")
line_end = text.find("\n", idx)
table_add = (
    "\n| 7 | 5ms 失稳后的正式默认 | **回退 1ms/50**（V1.1）；5ms 仅 experimental-only |"
    "\n| 8 | 5ms 复验条件 | 需先调 XPBD（iterations/substep）并按 §17 复验，未达标不进正式 trial |"
)
text = text[:line_end] + table_add + text[line_end:]

text = text.rstrip("\n") + """

## 17. 5ms/10 失稳证据（V1.1 修改依据，GPU 实测）

- 运行：64 env × 32 steps、5ms/10、全 real 观测、detections-warp、`--profile --nonfinite-limit 64`、
  `save-interval 20`；日志/产物 `train/runs/lstm_pilot_5ms_fixed/` 与 `proto/runs/lstm_pilot_5ms_fixed.log`。
- 结果：97 iter 内出现 **2 次非有限状态事件**（iter 76：8 envs 连续多步；iter 98 step 702：**全部 64 envs 同时**
  pos/lin_vel/ang_vel/wheel_qd 非有限），触发 `--nonfinite-limit` 后安全退出（exit 3，`failure.json` + `latest.pt` 保留）。
- 对照：同一 checkpoint 在 **1ms/50 GPU 评估**下无 NaN、干净翻车终止；zero-action 5ms 400 步稳定。
- 稳态吞吐：5ms 730.1 samples/s（iter 2–75）；防护后仍能继续训练，但求解器级失稳会污染 reward/loss 并中断长 run。
- 结论：5ms/10 不作为正式 trial 物理；V1.1 正式默认改为 1ms/50。V1 原文保留为历史记录。

## 18. Live-state 查看器（可选，v1.1 起）

- `train_wsl.py --live-state-file <path> --live-state-interval N`（默认关闭）：每 N 个 policy step 用
  `isaac_field.live_state.LiveStateExporter` 写 4 环境布局 + 19 body poses 的 schema-v1 packet，
  供 `scripts/view_training_state_web.py`（Viser）复用；只读消费，不影响训练/协议 SHA。
- 机器人与地形 USD：`C:/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/viz/field_robot_19.usda`（Lead 新增，不改仓库资产）。
"""

# strip old sha value so protocol_sha.py can embed the new one
import re
text = re.sub(r"(?m)^> protocol_sha256:.*$", "> protocol_sha256: PENDING", text, count=1)
V11.write_text(text, encoding="utf-8")
print(f"WROTE {V11}")
