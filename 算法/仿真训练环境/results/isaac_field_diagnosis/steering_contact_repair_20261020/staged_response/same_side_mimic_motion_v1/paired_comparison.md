# Same-side mimic 配对运动响应（2 ms）

本报告由 `summarize_paired_comparison.py` 直接读取两组原始 summary、effort JSONL、冻结 v2 coupling protocol 与 runner exit 记录生成；未改写原始结果。该八动作响应组使用 1 s zero-action settle、2 s drive、5 s brake。

## 配对与门禁

- baseline / mimic 实际退出码：0 / 0。
- 固定物理步长 0.002 s，decimation 25，策略步长 0.05 s；cone `elliptic`、impratio 1.0、shape μ 0.800000011920929、ground/wheel kf 1000.0。
- 两次真实 reset 的逐动作 hash map 一致：True；八动作均为同一 reset hash：True。hash：`7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`。
- 两组 drive/runtime physics/source-hash map 相同：drive=True，runtime physics=True，source maps=True；两组运行后冻结源均未变：True。仅 hook 从 baseline=False 切到 mimic=True。
- Ground+14轮 μ/kf 和 fixed drive runtime gates 两组均通过；14轮实际 effort cap ≤2 Nm 两组均通过。候选 Newton model/host/MJWarp equality 计数：`{"newton_model_mimic_constraint_count": 12, "newton_model_equality_constraint_count": 12, "solver_host_neq": 12, "solver_mjwarp_neq": 12, "all_12_expected_joint_pairs_match_once": true}`。
- Frozen v2 coupling gates：native qd-pair abs p95 ≤0.09259 rad/s，abs max ≤0.46296 rad/s，policy q-pair abs max ≤0.020 rad；baseline all-case=False，mimic all-case=True。

### 八动作并列

| 动作 | 实际Δyaw rad baseline→mimic | Δyaw | 末1 s yaw rate rad/s baseline→mimic | 距离 m baseline→mimic | 制动末0.3 s max |v| m/s baseline→mimic | max |w| rad/s baseline→mimic | 回摆积分 m 或 rad/s·s baseline→mimic | max / mean abs effort Nm baseline→mimic | mimic qd p95 / max / policy q max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forward `[0.12, 0.12, 0.0, 0.0]` | 0.0000 → 0.0000 | +0.0000 | -0.0000 → -0.0000 | 0.9831 → 0.9831 | 0.00000 → 0.00000 | 0.00000 → 0.00000 | 0.0505 m → 0.0505 m | 0.2285 / 0.0185 → 0.2284 / 0.0184 | 0.0003 / 0.0012 / 0.000009 (True) |
| reverse `[-0.12, -0.12, 0.0, 0.0]` | -0.0001 → -0.0001 | +0.0000 | -0.0000 → -0.0000 | -0.9832 → -0.9831 | 0.00000 → 0.00000 | 0.00000 → 0.00000 | 0.0505 m → 0.0506 m | 0.2285 / 0.0184 → 0.2284 / 0.0184 | 0.0003 / 0.0010 / 0.000008 (True) |
| pivot_left_low `[-0.1, 0.1, 0.0, 0.0]` | 0.1130 → 0.2471 | +0.1341 | 0.0804 → 0.1698 | -0.0601 → -0.2172 | 0.00005 → 0.00078 | 0.00286 → 0.00280 | 0.0000 rad/s·s → 0.0000 rad/s·s | 0.8727 / 0.4216 → 0.7661 / 0.4017 | 0.0398 / 0.1002 / 0.000318 (True) |
| pivot_right_low `[0.1, -0.1, 0.0, 0.0]` | -0.0877 → -0.2475 | -0.1598 | -0.0668 → -0.1737 | -0.0492 → -0.1251 | 0.00004 → 0.00053 | 0.00274 → 0.00234 | 0.0000 rad/s·s → 0.0000 rad/s·s | 0.8747 / 0.4246 → 0.8203 / 0.4025 | 0.0410 / 0.0917 / 0.000344 (True) |
| arc_left `[0.12, 0.24, 0.0, 0.0]` | 0.0334 → 0.0659 | +0.0325 | 0.0294 → 0.0685 | 1.3699 → 1.3590 | 0.00011 → 0.00036 | 0.00370 → 0.00252 | 0.0619 m → 0.0568 m | 0.9767 / 0.4028 → 0.8316 / 0.3739 | 0.0253 / 0.0756 / 0.000319 (True) |
| arc_right `[0.24, 0.12, 0.0, 0.0]` | -0.0299 → -0.0742 | -0.0443 | -0.0276 → -0.0723 | 1.3768 → 1.3520 | 0.00003 → 0.00001 | 0.00304 → 0.00222 | 0.0562 m → 0.0617 m | 1.0291 / 0.3998 → 0.8535 / 0.3636 | 0.0285 / 0.0707 / 0.000345 (True) |
| pivot_right_high `[0.4, -0.4, 0.0, 0.0]` | -0.4417 → -0.8723 | -0.4306 | -0.3321 → -0.6193 | -0.0437 → -0.0596 | 0.00004 → 0.00017 | 0.00285 → 0.00230 | 0.0000 rad/s·s → 0.0000 rad/s·s | 1.6347 / 0.4254 → 1.2674 / 0.4035 | 0.0532 / 0.1957 / 0.000484 (True) |
| pivot_left_high `[-0.4, 0.4, 0.0, 0.0]` | 0.4911 → 0.8361 | +0.3450 | 0.3538 → 0.5942 | -0.0839 → -0.1426 | 0.00002 → 0.00121 | 0.00283 → 0.00300 | 0.0000 rad/s·s → 0.0000 rad/s·s | 1.5648 / 0.4332 → 1.4345 / 0.4076 | 0.0516 / 0.2045 / 0.000500 (True) |

各组共同运动 reference gates 每动作全部通过（有限值、effort cap、无损伤/碰撞/episode done、姿态与制动速度门槛、对应方向/位移或 pivot/arc 检查）；记录的 informational gate 单独保留，baseline `pivot_right_low` 的 `signed_pivot_yaw_delta_ge_0_1rad` 为 false，不影响 reference-gate 通过状态。各组 8 个动作的全 reference gate 计数见 `paired_comparison.json`。

## 结论范围

mimic 候选在此固定仿真协议下通过八动作耦合残差门槛，pivot 与弧线的 yaw 响应幅度普遍高于 unlinked baseline；直行/倒车 yaw 仍接近零。制动尾部的 |v|/|w| 数值符合本组原门槛，但回摆积分并非各动作全面改善。该结果验证隔离机制响应，不是摩擦标定或生产修复结论；production 仍需经同初态、旧/新物理的既定验收。

Frozen coupling protocol: `same_side_mimic_v2/protocol.json` (SHA-256 `d7af43935bbf382a98c070bd7f0c64f3f64c87eb16d18edd1458cf88373938b6`). Runner exits and source summaries are enumerated in `paired_comparison.json`.
