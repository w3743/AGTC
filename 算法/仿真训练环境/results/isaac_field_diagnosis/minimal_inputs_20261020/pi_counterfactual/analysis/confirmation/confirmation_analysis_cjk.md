# P / PI 原生物理确认

**本轮在预定容差内配对，非逐数值相同；这是物理响应检查，不是导航验收。**

P(Ki=0) exit=0，PI(Ki=0.1) exit=0；修复后的 comparison exit=0、status=`passed`。原 comparison exit=1 保留在原文件；失败原因是比较器读取 `expected_timing.drive_s`，而协议键名为 `drive`。四个 case 按固定 protocol 门禁通过。

## 预定初态容差

| 字段 | 允许最大差 | 实际最大差 |
|---|---:|---:|
| `actual_v_w` | 0.0001 | 1.03489456e-05 |
| `position_xy_m` | 0.0001 | 0 |
| `roll_pitch_yaw_rad` | 0.0001 | 1.35299888e-06 |
| `root_z_m` | 0.0001 | 7.59959221e-07 |
| `track_command_mps` | 0 | 0 |
| `wheel_actual_qd_rad_s` | 0.001 | 0.000260742847 |
| `wheel_target_rad_s` | 0 | 0 |

## 低差速响应

| 动作 | 控制器 | 8 s净 yaw (rad) | 末1 s平均 abs(yaw rate) (rad/s) | 左/右轮速误差 (rad/s) | 最大单轮 effort (Nm) |
|---|---|---:|---:|---:|---:|
| .05 | P / Ki=0 | -0.0115 | 0.0014 | L 4.617, R 4.617 | 0.185 |
| .05 | PI / Ki=.1 | -0.3210 | 0.0390 | L 2.380, R 3.519 | 0.670 |
| .10 | P / Ki=0 | -0.0294 | 0.0036 | L 9.228, R 9.228 | 0.370 |
| .10 | PI / Ki=.1 | -0.6296 | 0.0747 | L 4.695, R 4.046 | 0.884 |

## 制动与命令归零

| 工况 | 控制器 | 两侧 cmd 持续归零 (s) | 末0.3 s最大 abs(v) (m/s) | 末0.3 s最大 abs(w) (rad/s) | 门禁 |
|---|---|---:|---:|---:|---|
| 直行 | P / Ki=0 | 0.55 | 0.0000 | 0.0000 | True |
| 直行 | PI / Ki=.1 | 0.55 | 0.0000 | 0.0000 | True |
| 差速 | P / Ki=0 | 2.20 | 0.0000 | 0.0000 | True |
| 差速 | PI / Ki=.1 | 2.20 | 0.0001 | 0.0033 | True |

每组共有 920 个策略采样和 9,200 个 Newton 物理子步记录。Newton 子步采样的最大单轮 effort：P `2.000000 Nm`，PI `2.000000 Nm`；protocol 上限为 2 Nm + 1e-5 Nm。图中制动曲线的竖线标记两侧履带速度命令持续归零，不表示车体在该时刻瞬间停止。

PI 的 Ki=0.1 是未标定的仿真假设；proxy-wheel effort 不是实测电机扭矩。结果只支持这个固定空场物理工况下的速度/偏航/制动响应描述，不证明训练效果、导航成功、绕障或作物安全。

Raw inputs SHA-256：

- `protocol`: `1236e551140423afd21749fd31a9431436fbf810d47f1331a83d36e0960fc8d0`
- `pair_summary`: `b154745dae42aac09561413e11fdf436e34b3bb5fe3376538080ea30c8505a42`
- `comparison_repair_exit`: `fb43ffaae1d9b66c76a8d00ecac98b61860f2c1366ccc515d7f71bcbcd13b53c`
- `original_comparison_exit`: `e41c59bc05edf7582b46c5d8f9831c9cc3e5755f9cf7fcf09a5d6b99913e114b`
- `P_summary`: `8dde8a0908057dd650577e051af66b0e40c285a56f004eb0a0eedace69954bf3`
- `PI_summary`: `17f7031a349c432da27a7a07472d9460cba2a8ed416feaed6bf3839c9ac31334`
- `P_raw_policy_samples`: `8a1f249d3ed3050ede64eb0c8a33d04dbb30019c07d247d590c74c3db7daed41`
- `PI_raw_policy_samples`: `1ccedbe931397d8fb49bf82afbf3945471f17a310f80206f7eb73cb3158d3b40`
- `P_raw_substep_effort`: `92e05ed73ea6467e810cbb4bb189d736b27cf6b212e457ea76c62ab8e8f43ce2`
- `PI_raw_substep_effort`: `fc5bb0d772f78a7d93869464d203ebd5f5ebada3d2f636f958ee8ba23dc1b301`

图文件：`confirmation_comparison_cjk.png`。

## 全程制动限制与补充 raw 指标

原 protocol 的制动门禁只检查末 0.3 s，因此“末段低速/命令为零通过”不代表整个制动过程都未恶化。PI 直行制动期间出现约 `-0.1300 m/s` 的反向速度回摆，侧履带轮速也有明显波动；虽然末段门禁通过，仍应把该过程响应视为需要继续检查的现象。

以下是额外的全程描述统计，不改变或放宽预先设定的门禁。每个 case 取 100 个策略样本，样本间隔约 0.05 s；反向速度积分按离散和 `sum(max(-v_i, 0) * 0.05 s)` 计算，覆盖 5 s。`max_abs_w` 是这 100 个样本上的最大值。

| 控制器 | 全程最小 v (m/s) | 反向速度积分 (m) | 全程最大 abs(w) (rad/s) |
|---|---:|---:|---:|
| P / Ki=0 | -0.00001436 | 0.00001723 | 0.00024221 |
| PI / Ki=.1 | -0.13002637 | 0.07293372 | 0.00056991 |

机器可读补充指标：

```json
{
  "case": "brake_straight_0.20_to_zero",
  "samples": 100,
  "policy_dt_expected_s": 0.05,
  "sampled_duration_s": 5.0,
  "reverse_speed_integral_method": "sum(max(-v_i, 0) * dt)",
  "P_Ki_0": {
    "min_v_mps": -0.00001435573631169973,
    "reverse_speed_integral_m": 0.000017230286300673472,
    "max_abs_w_rad_s": 0.00024220775230787694
  },
  "PI_Ki_0_1": {
    "min_v_mps": -0.13002637028694153,
    "reverse_speed_integral_m": 0.07293372424253736,
    "max_abs_w_rad_s": 0.0005699089961126447
  }
}
```
