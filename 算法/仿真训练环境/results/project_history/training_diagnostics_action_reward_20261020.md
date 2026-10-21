# 训练诊断动作与奖励补充 — 2026-10-20

本次仅补充训练诊断，不启动训练或 Isaac/Newton native 测试；按项目约定计作一个独立功能，不按源码、测试和记录文件拆分计数。

## 结果

- `isaac_field/training_diagnostics.py` 复用 `action_contract.py` 的 `direct-track-speed-v1` 顺序（左履带速度、右履带速度、云台偏航角速度、云台俯仰角速度），为采样动作与送入 VecEnv 的动作分别统计每维 mean/std/min/max、`|a| >= 0.95`、精确 `[-1, 1]` 边界和越界比例。比例分母为该维有限观测数；阈值含等号。输入形状不符合 `[num_env, 4]` 时，不做维度映射，保留原aggregate统计并标出 missing/unavailable/partial。
- 对每个环境步，把有限标量 `reward_components` 相加并与 callback 的原始 rewards 对照；使用绝对容差 `1e-5`，记录匹配、不匹配、缺失原因，不改写 callback reward。
- completed episode 以环境索引持有跨 rollout 的分项累计。仅完整观测到的分项给出 episode `total`；缺失或非有限分项时，`total` 为 `null`，另外报告只对实际观测求和的 `observed_sum` 和缺失/无效步数，不补零。rollout 原有分项汇总/API字段保留。
- logger 记录保留 logger 当时可见的数值快照；空/非数值快照仍为 `null`，并显式标记 `empty_or_non_numeric_snapshot`。已只读核对本机 SB3 2.9.0 的调用顺序：RecurrentPPO rollout-end 发生于 `ppo_recurrent.py:308`；SB3 on-policy 在 `on_policy_algorithm.py:266` 调用 rollout-end、`335-337` 接着 `dump_logs` 再 `train`；`logger.py:534,544` 的 `dump` 清空 `name_to_value`。因此默认 `log_interval=1` 时下一 rollout-end 通常能看到上一次 train 记录；空值不重建。输出中注明自定义 logging interval 可能改变来源窗口。

实现位置：`isaac_field/training_diagnostics.py:179`、`:294`、`:369`、`:467`、`:653`、`:681`、`:691`。专项验证覆盖四维顺序和边界统计、异常形状不误配、容差内/外奖励核对、跨 rollout 完整/缺失分项、空 logger snapshot；测试位置见 `tests/test_training_diagnostics.py:174`、`:216`、`:241`、`:272`、`:307`。

## 验证

在项目规定的 Ubuntu-24.04 WSL 环境执行，仅运行 CPU 专项 pytest：

```text
CUDA_VISIBLE_DEVICES='' bash scripts/isaaclab_python.sh -m pytest -q tests/test_training_diagnostics.py
```

真实退出码 `0`；最终代码状态下 `10 passed in 0.11s`。WSL 同时输出 localhost/NAT 警告，但 pytest 命令正常执行并返回 0。此次没有启动训练、native 测试或进程管理动作。
