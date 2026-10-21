# Same-side mimic 配对运动响应（1 ms）

本报告由 `summarize_paired_comparison.py` 直接读取两组原始 summary、effort JSONL、冻结 v2 coupling protocol 与 runner exit 记录生成；未改写原始结果。该八动作响应组使用 1 s zero-action settle、2 s drive、5 s brake。

## 配对与门禁

- baseline / mimic 实际退出码：0 / 0。
- 固定物理步长 0.001 s，decimation 50，策略步长 0.05 s；cone `elliptic`、impratio 1.0、shape μ 0.800000011920929、ground/wheel kf 1000.0。
- 两次真实 reset 的逐动作 hash map 一致：True；八动作均为同一 reset hash：True。hash：`7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`。
- 两组 drive/runtime physics/source-hash map 相同：drive=True，runtime physics=True，source maps=True；两组运行后冻结源均未变：True。仅 hook 从 baseline=False 切到 mimic=True。
- Ground+14轮 μ/kf 和 fixed drive runtime gates 两组均通过；14轮实际 effort cap ≤2 Nm 两组均通过。候选 Newton model/host/MJWarp equality 计数：`{"newton_model_mimic_constraint_count": 12, "newton_model_equality_constraint_count": 12, "solver_host_neq": 12, "solver_mjwarp_neq": 12, "all_12_expected_joint_pairs_match_once": true}`。
- Frozen v2 coupling gates：native qd-pair abs p95 ≤0.09259 rad/s，abs max ≤0.46296 rad/s，policy q-pair abs max ≤0.020 rad；baseline all-case=False，mimic all-case=True。

### 八动作并列

| 动作 | 实际Δyaw rad baseline→mimic | Δyaw | 末1 s yaw rate rad/s baseline→mimic | 距离 m baseline→mimic | 制动末0.3 s max |v| m/s baseline→mimic | max |w| rad/s baseline→mimic | 回摆积分 m 或 rad/s·s baseline→mimic | max / mean abs effort Nm baseline→mimic | mimic qd p95 / max / policy q max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forward `[0.12, 0.12, 0.0, 0.0]` | 0.0000 → 0.0000 | +0.0000 | -0.0000 → -0.0000 | 0.9831 → 0.9830 | 0.00000 → 0.00001 | 0.00000 → 0.00000 | 0.0508 m → 0.0507 m | 0.2291 / 0.0185 → 0.2292 / 0.0185 | 0.0003 / 0.0009 / 0.000006 (True) |
| reverse `[-0.12, -0.12, 0.0, 0.0]` | -0.0001 → -0.0001 | -0.0000 | -0.0000 → -0.0000 | -0.9832 → -0.9831 | 0.00000 → 0.00000 | 0.00000 → 0.00000 | 0.0508 m → 0.0508 m | 0.2291 / 0.0185 → 0.2292 / 0.0185 | 0.0003 / 0.0006 / 0.000006 (True) |
| pivot_left_low `[-0.1, 0.1, 0.0, 0.0]` | 0.0998 → 0.2590 | +0.1593 | 0.0704 → 0.1813 | -0.1631 → -0.2076 | 0.00001 → 0.00011 | 0.00222 → 0.00342 | 0.0000 rad/s·s → 0.0000 rad/s·s | 0.8866 / 0.4084 → 0.7917 / 0.4417 | 0.0248 / 0.0680 / 0.000258 (True) |
| pivot_right_low `[0.1, -0.1, 0.0, 0.0]` | -0.0977 → -0.2292 | -0.1315 | -0.0725 → -0.1641 | -0.0238 → -0.1714 | 0.00002 → 0.00005 | 0.00209 → 0.00296 | 0.0000 rad/s·s → 0.0000 rad/s·s | 0.9328 / 0.4420 → 0.8031 / 0.4326 | 0.0257 / 0.0689 / 0.000277 (True) |
| arc_left `[0.12, 0.24, 0.0, 0.0]` | 0.0345 → 0.0693 | +0.0348 | 0.0330 → 0.0690 | 1.3475 → 1.3474 | 0.00009 → 0.00077 | 0.00325 → 0.00348 | 0.0654 m → 0.0615 m | 0.9448 / 0.4014 → 0.8195 / 0.4314 | 0.0168 / 0.0463 / 0.000286 (True) |
| arc_right `[0.24, 0.12, 0.0, 0.0]` | -0.0260 → -0.0752 | -0.0492 | -0.0240 → -0.0761 | 1.3667 → 1.3526 | 0.00003 → 0.00010 | 0.00307 → 0.00376 | 0.0614 m → 0.0603 m | 1.0285 / 0.3910 → 0.8445 / 0.4360 | 0.0208 / 0.0559 / 0.000360 (True) |
| pivot_right_high `[0.4, -0.4, 0.0, 0.0]` | -0.4997 → -0.8371 | -0.3374 | -0.3795 → -0.6139 | -0.1250 → -0.1313 | 0.00005 → 0.00015 | 0.00292 → 0.00261 | 0.0000 rad/s·s → 0.0000 rad/s·s | 1.5053 / 0.4267 → 1.2223 / 0.4456 | 0.0371 / 0.1107 / 0.000381 (True) |
| pivot_left_high `[-0.4, 0.4, 0.0, 0.0]` | 0.5209 → 0.8894 | +0.3685 | 0.3854 → 0.6422 | -0.1986 → -0.2023 | 0.00041 → 0.00006 | 0.00304 → 0.00397 | 0.0000 rad/s·s → 0.0000 rad/s·s | 1.4848 / 0.4300 → 1.2989 / 0.4633 | 0.0376 / 0.1293 / 0.000519 (True) |

各组共同运动 reference gates 每动作全部通过（有限值、effort cap、无损伤/碰撞/episode done、姿态与制动速度门槛、对应方向/位移或 pivot/arc 检查）；记录的 informational gate 单独保留，baseline `pivot_right_low` 的 `signed_pivot_yaw_delta_ge_0_1rad` 为 false，不影响 reference-gate 通过状态。各组 8 个动作的全 reference gate 计数见 `paired_comparison.json`。

## 结论范围

mimic 候选在此固定仿真协议下通过八动作耦合残差门槛，pivot 与弧线的 yaw 响应幅度普遍高于 unlinked baseline；直行/倒车 yaw 仍接近零。制动尾部的 |v|/|w| 数值符合本组原门槛，但回摆积分并非各动作全面改善。该结果验证隔离机制响应，不是摩擦标定或生产修复结论；production 仍需经同初态、旧/新物理的既定验收。

Frozen coupling protocol: `same_side_mimic_v2/protocol.json` (SHA-256 `d7af43935bbf382a98c070bd7f0c64f3f64c87eb16d18edd1458cf88373938b6`). Runner exits and source summaries are enumerated in `paired_comparison.json`.
