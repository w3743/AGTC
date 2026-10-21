# 摩擦系数响应对照（5 ms，pyramidal，impratio=1）

三组采用相同 seed、八动作、驱动配置及逐case初态哈希。Ground 与 14 个轮碰撞形状的运行时 μ 均核验通过；KF/KE/KD 均为 1000/2500/100。每组策略轨迹 1280 行、物理步力矩 12800 行，原始文件路径见下表。

本对照只反映当前仿真材料设置下的响应，不用于真实摩擦标定，也不据此选择生产系数。

| μ | exit | Ground+14轮实测 μ | 最大力矩 Nm | 策略轨迹 / effort 行数 | summary |
|---:|---:|---:|---:|---:|---|
| 0.8 | 0 | 0.800000–0.800000 | 1.878016 | 1280 / 12800 | `results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/runs/baseline_5ms_pyramidal_1_kf1000/staged_motion_response_summary.json` |
| 0.4 | 0 | 0.400000–0.400000 | 1.409516 | 1280 / 12800 | `results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/runs/mu_0_4_5ms_pyramidal_1_kf1000/staged_motion_response_summary.json` |
| 1.2 | 0 | 1.200000–1.200000 | 2.000000 | 1280 / 12800 | `results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/runs/mu_1_2_5ms_pyramidal_1_kf1000/staged_motion_response_summary.json` |

共同初态哈希：`7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`。

每格顺序为 drive distance / yaw delta / last-1s yaw rate / 最大轮力矩；之后列制动末 0.3 s 最大 |v|、|w| 及反向积分。

| 动作 | μ=0.8 | μ=0.4 | μ=1.2 |
|---|---|---|---|
| forward [0.12, 0.12] | d=0.982 m; Δyaw=0.000 rad; w₁=0.000; τmax=0.228 Nm; brake max|v/w|=0.00001/0.00001; 反向位移 0.0503 m | d=0.967 m; Δyaw=0.000 rad; w₁=0.000; τmax=0.259 Nm; brake max|v/w|=0.00001/0.00000; 反向位移 0.0502 m | d=0.983 m; Δyaw=0.000 rad; w₁=-0.000; τmax=0.227 Nm; brake max|v/w|=0.00000/0.00000; 反向位移 0.0498 m |
| reverse [-0.12, -0.12] | d=-0.984 m; Δyaw=-0.000 rad; w₁=-0.000; τmax=0.227 Nm; brake max|v/w|=0.00000/0.00000; 反向位移 0.0505 m | d=-0.963 m; Δyaw=-0.000 rad; w₁=0.000; τmax=0.259 Nm; brake max|v/w|=0.00006/0.00016; 反向位移 0.0478 m | d=-0.983 m; Δyaw=-0.000 rad; w₁=-0.000; τmax=0.227 Nm; brake max|v/w|=0.00000/0.00000; 反向位移 0.0497 m |
| pivot_left_low [-0.1, 0.1] | d=-0.055 m; Δyaw=0.165 rad; w₁=0.107; τmax=0.842 Nm; brake max|v/w|=0.00118/0.00563; 反向偏航 0.0000 rad | d=-0.009 m; Δyaw=0.420 rad; w₁=0.237; τmax=0.544 Nm; brake max|v/w|=0.00233/0.00232; 反向偏航 0.0000 rad | d=-0.053 m; Δyaw=0.119 rad; w₁=0.082; τmax=1.033 Nm; brake max|v/w|=0.00035/0.01017; 反向偏航 0.0000 rad |
| pivot_right_low [0.1, -0.1] | d=-0.013 m; Δyaw=-0.145 rad; w₁=-0.098; τmax=0.814 Nm; brake max|v/w|=0.00013/0.00440; 反向偏航 0.0000 rad | d=-0.032 m; Δyaw=-0.408 rad; w₁=-0.237; τmax=0.562 Nm; brake max|v/w|=0.00257/0.00344; 反向偏航 0.0000 rad | d=-0.061 m; Δyaw=-0.105 rad; w₁=-0.073; τmax=1.004 Nm; brake max|v/w|=0.00007/0.00973; 反向偏航 0.0000 rad |
| arc_left [0.12, 0.24] | d=1.347 m; Δyaw=0.056 rad; w₁=0.051; τmax=0.772 Nm; brake max|v/w|=0.00008/0.00507; 反向位移 0.0625 m | d=1.340 m; Δyaw=0.163 rad; w₁=0.128; τmax=0.694 Nm; brake max|v/w|=0.00024/0.00298; 反向位移 0.0726 m | d=1.392 m; Δyaw=0.040 rad; w₁=0.040; τmax=1.068 Nm; brake max|v/w|=0.00016/0.00932; 反向位移 0.0652 m |
| arc_right [0.24, 0.12] | d=1.371 m; Δyaw=-0.054 rad; w₁=-0.046; τmax=0.841 Nm; brake max|v/w|=0.00010/0.00324; 反向位移 0.0552 m | d=1.302 m; Δyaw=-0.127 rad; w₁=-0.114; τmax=0.566 Nm; brake max|v/w|=0.00229/0.00199; 反向位移 0.0608 m | d=1.360 m; Δyaw=-0.038 rad; w₁=-0.033; τmax=1.058 Nm; brake max|v/w|=0.00048/0.00699; 反向位移 0.0652 m |
| pivot_right_high [0.4, -0.4] | d=-0.082 m; Δyaw=-0.506 rad; w₁=-0.326; τmax=1.878 Nm; brake max|v/w|=0.00070/0.00498; 反向偏航 0.0000 rad | d=-0.029 m; Δyaw=-1.232 rad; w₁=-0.818; τmax=1.410 Nm; brake max|v/w|=0.00094/0.00206; 反向偏航 0.0000 rad | d=-0.048 m; Δyaw=-0.350 rad; w₁=-0.242; τmax=2.000 Nm; brake max|v/w|=0.00013/0.01046; 反向偏航 0.0000 rad |
| pivot_left_high [-0.4, 0.4] | d=0.016 m; Δyaw=0.462 rad; w₁=0.286; τmax=1.735 Nm; brake max|v/w|=0.00003/0.00614; 反向偏航 0.0000 rad | d=-0.020 m; Δyaw=1.576 rad; w₁=1.143; τmax=1.406 Nm; brake max|v/w|=0.00165/0.00215; 反向偏航 0.0000 rad | d=-0.149 m; Δyaw=0.360 rad; w₁=0.251; τmax=2.000 Nm; brake max|v/w|=0.00046/0.00975; 反向偏航 0.0000 rad |

完整逐策略步轨迹和逐物理步力矩见各 run 目录中的 `policy_trajectories.jsonl` 与 `native_physics_efforts.jsonl`。
