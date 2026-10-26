# 官方 SLE 测距算法研究结论（基于头文件 + 预编译库反汇编，非猜测）

> 2026-10-25｜研究对象：`application/samples/products/sle_measure_dis` + `libcal_dis.a`
> 工具链反汇编参数：`riscv32-linux-musl-objdump -d -r`（库成员解包于 `%TEMP%\nl_libcal`）

## 1. `calib_val` 的精确语义（官方唯一校准入口）

- 头文件：`float calib_val; /* IQ的校准值 */`（`slem_alg_smooth_dis.h:83-84`）。
- **反汇编事实**：`slem_alg_calc_smoothed_dis` 把它传给 `esprit_ver_two`，后者在**平滑之前**执行
  `dist_first = max(距离估计 − calib_val, 0)`、`dist_second` 同理（三处 `fsub.s`）。
  ⇒ `calib_val` 是**从原始距离里减掉的常偏置，单位米**，**只能偏移、不能缩放**。
- 官方文档原文（`BS2XV100 SDK Sample使用指南.md:1768-1775`）：
  「锚点周围 3m 内空旷无遮挡…在设备周围 5 个方向的 1m 位置各测 5 次，**平均值减 1 为校准值**，在
  `sle_measure_dis_server_alg.c` 代码位置**减去校准值**」。
  ⇒ 校准值 = `mean(dist_first @1m) − 1.0`，填到**生效的** `slem_posalg_set_base_para()`
  （`sle_measure_dis_server_alg.c:32`）。
- ⚠️ 官方文档图指向的 `measure_dis_posalg_set_base_para()`（同文件 :59-64）**是死函数**（无调用点），改它无效。
- 若应用层自己做校准，库内 `calib_val` 应保持 0，避免双重校准。

## 2. TOF 链路在当前构建里是「死」的（不要指望 TOF 改善精度）

- `iq_tof_dis_merge()` 只把内部恒为 0 的变量加到距离上（空操作），**完全不读入参**。
- `tof_calib=2070` 只被 `iq_tof_phase_shift()` 使用，触发条件是 `51 ≤ (tof_result − tof_calib) ≤ 299`。
- 我们的构建**未定义 `GLE_CS_MODE3_SUPPORT`**，`tof_result` 恒 0 ⇒ 相位斜坡分支永不触发。
- 结论：TOF / `tof_calib` 不参与当前测距，不是误差来源；想开启需协议栈上报 CS mode3。

## 3. `ranging_method` 与官方注释的量程（`slem_disalg.h:50-67`）

| 值 | 枚举 | 官方注释 | 库内实际（反汇编） |
|---|---|---|---|
| 1 | METHOD_1M | 1MHz 跳频，上限 150m | 79 信道 |
| 2 | METHOD_2M | — | **直接被库拒绝**（返回 0x8000A400）|
| 3 | METHOD_1M_2M | 低复杂度，上限 75m | 39 信道 |
| 4 | METHOD_ADJ_R_END | 动态 r，上限 150m | 79 信道（官方默认） |
| 5 | METHOD_ADJ_R_END_V2 | 动态 r v2，150m | 79 信道 |
| 6 | METHOD_ADJ_R_END_V3 | 动态 r v3，150m | 79 信道 |

算法栈：`trans_data`（IQ→复相关/幅度/RSSI）→ `spatial_smoothing` → `esprit_decomp`/`esprit_ver_two`
（ESPRIT 相位斜率）→ `calc_r_end`/`fix_pointf` → `kde_calc`/`find_peak`（KDE 峰选）→ `dist_calc_prob`。

## 4. 输出字段与平滑（`slem_alg_smooth_dis.h:104-119`，反汇编对位）

| 字段 | 含义 | 单位/范围 |
|---|---|---|
| `dis_smoothed` | 库内两段 Kalman 平滑后的距离（**官方主输出**） | 米 |
| `dis_ori` | 未平滑首选值（`dist_first`），抖动大 | 米 |
| `dis_slight_smoothed` | 第二级 Kalman（轻度平滑） | 米 |
| `prob` | 置信度：RSSI < −70dBm→0；> −45dBm→≈1 | [0,1] |
| `smooth_num` | 连续有效测距次数，上限 6（`slem_get_min_smooth_num()==6`） | 0..6 |
| 错误帧 | 库返回错误时输出模板 `dis_smoothed=−1, prob=−128` | 必须丢弃 |

平滑特性：Q=1e-5 / R=1e-4；需 3 帧初始化、smooth_num 到 6 才稳定；**停 >5s 或计数差 ≥5 会重启平滑**；
库内野值门限最小 8m（几乎不拦抖动）。

## 5. 实测与结论（本次两块 H2821E，官方基线）

见 [`Docs/range_link/official-baseline.md`](../range_link/official-baseline.md)：
- 官方原始输出（未校准）在 0.5m / 1.0m 点：**1.560 m / 2.161 m**（σ≈1cm）。
- 低干扰条件下原始输出 ≈ **真值 + 1.08~1.16 m 的常偏置**（不是比例误差）。
- 旧版 0.454 倍校准模型是错误拟合（两点数据异常，实际摆位可能不是 1.0m），导致 1m 读数 −26%。
- 修正：用**官方偏移校准**（`mean@1m − 1 m`）或应用层等价的 `SCALE_PM=1000, CALIB_MM=mean−1000`。

## 6. 我们的其它发现（对精度影响大）

1. **client 每 500ms 发用户数据会干扰 IQ 上传/配对**：实测 0.5m 处原始值从 1583mm（10s 周期）被拉到
   1420mm（500ms 周期）。修正：用户数据周期改为 5s（或测距时暂停业务）。
2. **旧版自定义滤波窗口+节流顺序错误**：节流在滤波之前，导致窗口样本跨 4s、换位后长时间卡旧值。
   修正：每帧进滤波（默认窗口 1 = 直通），节流只作用于打印/notify；新增连续野值重置。
3. **无效帧必须丢弃**：库返回 −1 的帧不能进校准/滤波/notify。
4. `prob` 不能当精度指标（RSSI 好就高分），只能做质量门限。
5. IQ 数据必须传原始 11bit 值（库内自己做 flip/offset、RSSI−256），不要手工换算。
6. 平滑状态是全局的；官方未调用 `slem_init_smooth()`，必要时可在连接/使能后重置一次。

## 7. 对「其它距离点误差大」的解释（按证据强度）

1. **校准模型/常量错误**（已修正，可算术复现 −26%）。
2. **用户数据业务干扰 IQ**（已实测，-160mm@0.5m）。
3. **原始值随环境/摆位/时间漂移 5~10%**（同一距离不同会话 raw 1.42~1.56m 波动）→ 单点校准无法覆盖，
   需在**使用现场**重新标定，或用多点分段校准。
4. 近场（<0.5m）可能受球面波/IQ 饱和影响；官方校准点选 1m、适用范围 3m 内。
5. ESPRIT 解模糊多解（`dist_first`/`dist_second` 切换）可用两者差值监控。
