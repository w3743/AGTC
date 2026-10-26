# 官方资料检索：SLE（HADM/Channel Sounding）测距参数、精度、校准与误差原因

调研员：web-researcher ｜ 任务：task-11 ｜ 日期：2026-10-25
目标：回答「官方怎么测距」——不靠猜。每条给来源与可信度：
**【官方文档】**=docs.hisilicon.com / developer.huawei.com / 联盟规范；**【官方代码】**=SDK 随包头文件与预编译库（厂商交付物，本地可查证）；**【社区】**=个人博客/公众号；**【低可信】**=疑似 AI 生成或营销稿。

---

## 0. 结论速览（先看这段）

1. **官方只给了一个校准方法：`1m` 处测 5 个方向、每个方向测多次，取平均再「减 1」，把结果写进 `sle_measure_dis_server_alg.c` 的 `calib_val`。**
   ——"减 1"这个动作说明：**`calib_val` 是「距离偏移量」，单位与输出距离一致（米）；本质是减一个常偏置，不是乘一个比例。**
   【官方文档】https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/debug/SDK%20Sample%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/SDK%20Sample%20%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.html

2. **官方文档对 `freq_space / con_anchor_num / refresh_rate / acb_interval / cs_interval / posalg_freq / glp_mode` 只写了字段名和一句中文描述，没给单位、没给取值范围、没给推荐值。** 线上 HADM API 页同样只列"名称 + 描述"。
   【官方文档】https://docs.hisilicon.com/projects/fbb_ws63/master/api-reference/middleware/sle/service/hadm/hadm.html
   我们 SDK 头文件里 `acb_interval`/`cs_interval` 的注释明确写着 **"根据参数计算得到的"** → 它们是**派生量**，真正要设的是 `freq_space`/`refresh_rate`/`con_anchor_num`（+`posalg_freq`/`glp_mode`）。
   【官方代码】`SDK\...\include\middleware\services\bts\sle\sle_hadm_manager.h:47-64`

3. **在我们的日志里反推出关键事实：库输出的原始距离是"大常偏置"型，`raw ≈ 真实距离 + ~1.26 m`**（用你们两个校准点反算：0.5m→1.729、1.0m→2.295m，偏置 1.229/1.295m）。
   → 你们现在的「比例 454‰ + 偏移 303mm」校准模型**用错了**：对一个"常偏置"的输出做 0.45 倍缩放，必然表现为"0.5m 凑准、越远越偏小"（1m 只剩 739mm，正是 -26%）。
   **正解：先用单点偏移（官方法）消掉 ~1.3m 偏置，再按 3~5 个点做分段线性/最小二乘。** 详见 §4、§7。

4. **"越远误差越大"的官方/机理排序**（详见 §5）：
   ① 原始值本身在固定距离上就抖动/跳变（你们日志同一位置 `RAW` 在 1594~2332mm 之间跳，±0.35m）；
   ② 大常偏置 + 错误的校准模型（比例 vs 偏移）；
   ③ 解模糊"首选/候补"两个候选值（`dist_first`/`dist_second`）+ 置信度 `prob`，可能选错解；
   ④ IQ 饱和/能量门限（近场强信号）与相位噪声随距离放大；
   ⑤ 多径/墙体/人体/天线方向（官方要求 3m 内空旷、五方向校准）；
   ⑥ 库内 Kalman/平滑滞后（移动/扫频时读数漂移）；
   ⑦ TOF 未真正参与融合（详见 §5.7）。

5. **能改善远距离的可调项**：`ranging_method` 换算法（1/3/4/5/6，注意 **2 会被库直接拒绝**）、`r_start`/`threshold_cond2` 调解模糊窗口、`freq_space` 调频率间隔、`tof_calib` 调 TOF 偏置、`refresh_rate` 调刷新率。**这些官方都没有推荐值**，必须自己扫参数（§6 给了实验矩阵）。

---

## 1. 参数逐条：官方怎么说 vs 我们 SDK 里的实证

官方 HADM API 页只给字段名+一句话：【官方文档】`sle_set_channel_sounding_param_ex_t`：
`is_cs_param_chg`／`freq_space`／`con_anchor_num`／`refresh_rate`／`acb_interval`／`cs_interval`／`posalg_freq`／`glp_mode`。
URL：https://docs.hisilicon.com/projects/fbb_ws63/master/api-reference/middleware/sle/service/hadm/hadm.html

| 参数 | 官方文档原文 | 官方 SDK 代码补充（本地实证） | 官方样例取值 | 结论/建议 |
|---|---|---|---|---|
| `freq_space` | 频率间隔 | `uint8_t`，无单位说明 | **0** | 0 大概率="默认间隔"。可试 1/2/4 观察 raw 变化；单位【不确定】 |
| `con_anchor_num` | 需要连接的锚点数量 | `uint8_t` | **1** | 单锚点测距填 1；多锚点要配 GTTT 那套 API（`sle_set_channel_sounding_param_ex_gttt_t`，含 `is_gtt`/`is_gtt_tt_slem`/`tt_tx_conhdl`…） |
| `refresh_rate` | 刷新率 | `uint8_t`，无量纲 | **1** | 语义【不确定】（Hz？档位？）。可试 2/4 提高刷新 |
| `acb_interval` | **"根据参数计算得到的"** acb 链路周期 | `uint16_t`，派生量 | **0**（交给库算） | **不要手填** |
| `cs_interval` | **"根据参数计算得到的"** 测距周期 | `uint16_t`，派生量 | **1** | 同上，派生 |
| `posalg_freq` | 算法频率 | `uint16_t`，无量纲 | **未赋值（0）** | 可试与 refresh_rate 同值；语义【不确定】 |
| `glp_mode` | 2.4GHz 模式 | `uint8_t` | 未赋值（0） | 保持默认 |
| `calib_val` | （HADM 页无此字段，只在算法头）"**IQ 的校准值**"（float） | `slem_disalg.h` / `slem_alg_smooth_dis.h`：「IQ的校准值」 | **0**（官方样例不预置） | 官方唯一给出的校准法就是"1m 五方向均值 − 1" → **单位=米、加法偏置** |
| `tof_calib` | 算法头："**ToF 的校准值**"（int） | 样例 `#define TOF_DEFAULT 2070` | **2070** | 单位【不确定】（mm？ps？）。改之前先打印 `tof_result` 看量级 |
| `ranging_method` | 算法头：`slem_dis_alg_type` 枚举 | 见下表，**每个算法注明了测距上限** | **4**（`DIS_ALG_MODE`） | 见 §2 |
| `r_start` | 参数头："测距参数2，默认 2" | `int8/uint8_t` | **2** | 解模糊窗口相关，**可调** |
| `threshold_cond2` | 参数头："测距参数1，默认 20" | `float` | **20** | 候选解判决门限，**可调** |
| `rssi_limit` | 参数头："RSSI 异常值剔除门限" | `int8_t`，样例 `-120` | **-120** | 你们 RSSI −46~−50，远高于门限，**不是**你们的问题 |

来源：【官方代码】`SDK\bearpi-pico_h2821e\protocol\bt\algorithm\posalg\cal_dis_bs21\cal_dis_bs21_h\slem_disalg.h`、
`...\slem_alg_smooth_dis.h`、`SDK\...\include\middleware\services\bts\sle\sle_hadm_manager.h`；
样例取值：`SDK\...\application\samples\products\sle_measure_dis\sle_measure_dis_client\sle_measure_dis_client.c:42-58`、
`...\sle_measure_dis_server\sle_measure_dis_server_alg.c:16-35`。

### 1.1 校准值到底写哪一行（官方样例有个坑）
官方文档图指向的位置与代码实际生效点**不一致**：
- **实际生效**：`slem_posalg_set_base_para(alg_para, key_id)`（`sle_measure_dis_server_alg.c:29-35`），它被 `measure_dis_posalg_get_distance()`（:71）调用；
- **死函数**：`measure_dis_posalg_set_base_para(alg_para)`（:59-64）**没有被调用**，只改这里不生效。
  （你们设计文档里已经踩到并记录了这一点——`Docs\design\range_link_design.md:21,102`，官方文档没说，属于 SDK 自身坑。）

---

## 2. 官方给的"测距算法"与量程（`ranging_method`）

【官方代码】`slem_disalg.h:50-67` 原文枚举（带**官方注释的测距上限**）：

| 值 | 枚举 | 官方注释 |
|---|---|---|
| 1 | `METHOD_1M` | 信道 **1MHz 跳频**算法（**测距上限 150m**） |
| 2 | `METHOD_2M` | 信道 2MHz 跳频算法（**无上限注释**） |
| 3 | `METHOD_1M_2M` | 信道 1MHz 跳频**低复杂度**算法（**上限 75m**） |
| 4 | `METHOD_ADJ_R_END` | 信道 1MHz 跳频**动态 r** 算法（上限 150m）← 官方样例默认 |
| 5 | `METHOD_ADJ_R_END_V2` | 动态 r 算法 **v2**（上限 150m） |
| 6 | `METHOD_ADJ_R_END_V3` | 动态 r 算法 **v3**（上限 150m） |

⚠️ **实测（反汇编官方预编译库 `libcal_dis.a` 的 `slem_alg_check_para`）：算法 2（`METHOD_2M`）会被参数检查直接拒绝**（只接受 1、3、4、5、6），返回 `0x8000a400`；且:
- 方法 1/4/5/6 → 每次用 **79 个信道** IQ；
- 方法 3（低复杂度）→ 只用 **39 个信道** IQ。
【官方代码】`libcal_dis.a` 反汇编（`riscv32-linux-musl-objdump -d`，本地），函数 `slem_alg_check_para`。

→ **建议实验顺序：4（现状）→ 1（1M 标准算法）→ 6（动态 r v3）→ 3（低复杂度/近距）**，每个测 0.5/1/2/3m 各 30 组对比。**不要试 2。**

---

## 3. 官方精度/量程指标（有就说，没有就明说"没有"）

- **BS21E/H2821E 单板的 SLE 测距精度指标：官方文档里查不到**（SDK Sample 指南、HADM API 页、芯片简介均无 accuracy 数字）。这是文档缺口，不是我们没找到。
- **量程**：仅算法头注释给了上限（1M 系列 150m、低复杂度 75m），这是**算法可解析范围**，不是可用精度范围。
  【官方代码】`slem_disalg.h:51-66`
- **定位用途表述**：HADM 被官方定义为"高精度测距"，用于"定位标签、室内导航、接近检测"。
  【官方文档】https://docs.hisilicon.com/projects/fbb_ws63/master/api-reference/middleware/sle/service/hadm/hadm.html
- **HarmonyOS 侧术语库**：HADM = "通过测量**信号往返时间或相位差**计算设备间距离，**可实现厘米级测量精度**"。
  【官方文档，但属上层 API 宣传口径】https://device.harmonyos.com/cn/docs/security/harmonyos-guides/fusion-connectivity-terminology
- **星闪 1.0 白皮书（需求指标）**：手机-耳机场景 "<1m ±5cm、>1m ±5%"，两天线测角 10m 内 ±5°。
  【官方/联盟白皮书，本地】`Docs\03_白皮书与报告\星闪无线短距通信技术（SparkLink1.0）产业化推进白皮书.pdf`（P17 附近）
- **小熊派官方公众号**："BM H21E 实测 **100m 精确测距**"、"已发布 5m/10m 误差实测**视频**"（**未给数字**）。
  【社区/厂商宣传】https://my.oschina.net/u/4445410/blog/18621344
- **社区课程教材**："2821E 星闪芯片具备测距能力，可以达到**厘米级**精准度"——【社区，可信度中低】https://gitee.com/ohbu/openharmony-nearlink-experiment/raw/master/19.%20可信星闪信标（Trusted%20starBeacon）发现端%20综合实验.pdf
- **不可引用**：`jishuzhan.net/article/2006586343326531586` 与其 CSDN 镜像的"±0.5m/50m/200 并发"等数字——出现不存在的 API（`ai_bs21_sle.h`、`AT+SLERANGE`），**疑似 AI 生成**【低可信】。

**给 lead 的口径**：官方**没有**公布 H2821E 的 SLE 测距精度；能查到的官方一致口径是"高精度（相位/TOF 融合）、厘米级（HarmonyOS 术语）、需求 ±5cm@<1m/±5%@>1m"，实测受校准与环境支配。**在完成校准前不要对外承诺精度。**

---

## 4. 校准：官方唯一方法 + 你们为什么错

### 4.1 官方原文（SDK Sample 指南「SLE measure dis → 校准」）
> 测试环境：锚点周围 **3m 范围内空旷无遮挡、无墙体、柱体、金属等遮挡物**。
> 测试方法：按照图 1 所示在设备周围 **5 个方向的 1m 位置，进行 5 次测距，5 次测距值的其平均值减 1 为锚点 A 的校准值**。
> 在如图 2 代码位置加上校准值。`application\samples\products\sle_measure_dis\sle_measure_dis_server\sle_measure_dis_server_alg.c`

【官方文档】https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/debug/SDK%20Sample%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/SDK%20Sample%20%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.html

解读（重要）：
- "**平均值减 1**"→ 校准值是一个**偏移量**（把 1m 处读数拉回 1.0），单位与距离输出一致；样例日志 `SLEM get distance done. distance:%d.%03d` 与你们 `dist*1000→mm` 的用法，都指向 **输出单位=米**。
- 官方**没有**教"两点比例校准"；比例/斜率属于"多点标定"，官方在另一个测距案例里明确说"**单点只能校准偏移，不能同时拟合斜率；产品化推荐多个已知距离点做回归并记录拟合误差**"。
  【官方文档，RSSI 测距案例的方法论章节】http://docs.hisilicon.com/projects/fbb_ws53/master/samples/connectivity/sle/link-mgmt/rssi-ranging/ （"为什么必须标定" / "100 cm 单点只能校准偏移 A，不能同时拟合 n"）

### 4.2 用你们自己的数据算一遍（这就是 1m 偏小 26% 的来源）
你们固件：`d_cal = d_raw × SCALE_PM/1000 − CALIB_MM`，`SCALE_PM=454`、`CALIB_MM=303`。
README 记录：0.5m→482mm、1.0m→739mm（12 组均值）。反算**固件原始输出**：
- 0.5m：`raw = (482+303)/0.454 ≈ 1729 mm`
- 1.0m：`raw = (739+303)/0.454 ≈ 2295 mm`

于是：**raw − 真实距离 = +1229mm（0.5m）、+1295mm（1.0m）≈ 常数 +1.26m**。
也就是说固件给的 `dist_first` 是"真值 + ~1.26m"，**是一条截距很大的直线，不是需要缩放 0.45 倍的曲线**。

- 若按官方法（0.5/1m 各取平均，用 1m 点：calib = 2.295 − 1 = **1.295**）：
  - 0.5m → 1.729 − 1.295 = **0.434m**（−66mm）
  - 1.0m → 2.295 − 1.295 = **1.000m**（0 误差）
- 而你们按 0.454 缩放：1.0m → 2.295×0.454 − 0.303 = **0.739m**（−261mm）。

→ **结论：校准模型用错是"越远越偏小"的直接主因。** 先换成"偏移校准"（SCALE_PM=1000、CALIB_MM≈1295），再看剩余误差；不要在没有原始数据回归的情况下拍一个 0.45 的斜率。

> 补充：`calib_val` 在库内是"加"还是"减"需要一次二分实验确认（官方只说"加上校准值"）。**实验**：把 `calib_val = (1m 处 dist_first 均值 − 1.0)` 填进去（例：1.295）。若 1m 读数 ≈ 1.000 → 库是"减 calib_val"，按此填；若 ≈ 3.59 → 是"加"，改填 `−1.295`。一次烧录即可判定。

---

## 5. 为什么"不同距离误差不同"——按证据强度排序

### 5.1 原始值本身抖动大（最强证据，先解决它）
你们 `Docs\range_link\demo-evidence.log` 同一位置连续 12 帧：`RAW=2332,2271,2040,2108,1594,1788,1926,2029,2100,1982,1938,1787 mm`，而 `PROB=920~949`、`RSSI=-47~-50dBm`。
- 抖动幅度 **1594→2332mm，跨度 738mm**，而 `PROB` 依然很高 → **`PROB` ≠ 精度**，不能只用 prob 当质量门限。
- 同时 `dist_first`(平滑) 从 754→555mm 单调下降，说明库内 Kalman/平滑在跟着某个状态漂移。
【本地实证】`Docs\range_link\demo-evidence.log:3-14`；【官方代码】`slem_smoothed_dis_result` 的 `dis_smoothed/dis_ori/dis_slight_smoothed/prob/smooth_num`（`slem_alg_smooth_dis.h:104-119`），库内符号 `g_kalman_p_first/g_kalman_p_second/g_kalman_p_prob/g_kalman_q/g_kalman_r`（`libcal_dis.a`，即**平滑是库内 Kalman，不是应用层**）。

### 5.2 大常偏置 + 校准模型错误（§4.2，已定量）

### 5.3 解模糊"候选解"选错（库内就是这么设计的）
- `slem_dis_result` 有 **`dist_first`（首选）/ `dist_second`（候补）** 两个距离 + `prob`（置信度）+ `height`（"置信度中间信息"）。
  【官方代码】`slem_disalg.h:134-149`
- 预编译库符号显示算法是 **ESPRIT**（`esprit_decomp`、`esprit_ver_two`、`calc_phi0`、`calc_r_end`、`spatial_smoothing`）+ **KDE 峰值选择**（`find_peak`、`kde_calc`、`dist_calc_prob`）+ 定点转换（`fix_pointf`）。
  → 相位法在"虚拟大带宽"上做超分辨估计，**存在多解**，靠 `prob`/`threshold_cond2` 选解；**噪声/多径/近场会让它选到错误分支**，表现就是"某个距离点上系统性偏大/偏小，且跳变"。
  【官方代码】`libcal_dis.a` 符号表（`riscv32-linux-musl-nm`，本地）
- **可调**：`r_start`（默认 2）、`threshold_cond2`（默认 20）就是这套判决的参数。
- **诊断手段**：把 `dist_second` 也打出来（你们现在只打了 `dist_first`）。**若 `dist_first` 与 `dist_second` 频繁互换/相差 ~0.3–1.3m，就是解模糊在跳。**

### 5.4 IQ 饱和/幅度门限（近场效应，官方库内的硬门限）
反汇编 `slem_alg_check_para` 可见库对 IQ 有**硬性有效性检查**：
- IQ 为 **11bit（第 11 位符号位，10bit 数据）**，`ALG_IQ_FLIP_BARRIER=0x03ff`、`ALG_IQ_FLIP_OFFSET=0x0800`；
- 库内判定"有效 IQ"的窗口约为 **5 ~ 2043**（代码：`v-5 <= 2038`）；
- **若某一路超过半数采样落在窗外，直接返回错误 `0x8000a453`**；RSSI 越界返回 `0x8000a450`；
- `ALG_IQ_LOW_ENERGY_BARRIER=5`（"IQ 绝对值小于该值认为幅度过小，可能异常"）。
  【官方代码】`slem_disalg.h:16-41` + `libcal_dis.a` 反汇编 `slem_alg_check_para`
→ **近距离（<0.3m）信号强、AGC 压缩/IQ 削顶会让相位失真**；远距离幅度低、信噪比差 → 相位噪声增大 → 误差随距离上升。这也解释了官方为何把校准点放在 **1m**（不在近场）。

### 5.5 相位噪声/距离的关系（物理层必然）
SLE 是**单载波**（1/2/4MHz 带宽，250µs 双向交互），靠 79 信道跳频合成虚拟带宽做相位测距（见 `ALG_CHANNEL_NUM=79`）。
相位误差 Δφ 折算距离误差 ∝ λ·Δφ/2π，且**相位斜率估计的方差随距离增大而放大**（相同 Δφ 在更大延迟上对应更大距离误差），这就是"越远越差"的物理根源；官方/社区的通用建议也是"**近/中/远分段标定**"（见 §3 的 RSSI 案例"必要时对不同区域分别标定"）。
【官方代码】`slem_alg_smooth_dis.h:23`（79 信道）；【官方文档（方法论）】WS53 RSSI 测距案例"限制"章节。

### 5.6 环境/姿态/多径（官方明文要求）
官方要求校准环境"3m 内无墙体/柱体/金属"，并要求**5 个方向**各测 —— 就是在承认**天线方向性/多径会造成方向相关的偏置**。你们桌面环境 + 板载天线平行摆放，属于"未按官方条件"，偏置不可比。
【官方文档】同上 SDK Sample 指南；小熊派产品页另注"**外接 IPX 天线需先调整天线电阻**" https://www.bearpi.cn/core_board/bearpi/pico/h2821E/

### 5.7 TOF（CS mode 3）是否启用：能查到的事实
- 库/样例里 `GLE_CS_MODE3_SUPPORT` **是被强制定义的**（`cal_dis_bs21_h\f2c.h:20-22`：`#ifndef ... #define ...`）→ **样例中 TOF 相关分支是编译进来的**，`tof_result` 会随 IQ 上报；
- 库内 **`iq_tof_dis_merge()`** 存在，注释明确："**IQ和TOF测距值移位融合，防中继保护**"，`tof_calib` 是它的偏置（样例默认 2070）；另有独立输出 `tof_dis`（"通过飞行时间计算的测距值"）。
  【官方代码】`slem_disalg.h:168-187`、`slem_alg_smooth_dis.h:128-139`
- **但是否真的测到有效 TOF，取决于协议栈 CS 配置与对端能力，官方未文档化**；样例在 `GLE_CS_MODE3_SUPPORT` 下会打印 `local tof_result = ...`。
  → **行动**：开 `CONFIG_RANGE_LINK_VERBOSE` 打印 `tof_result`，若它长期为 0/恒定/明显超出量级，说明 TOF 没生效，此时纯相位解算 → 远距离更差、也更容易被中继攻击。
  【不确定】`tof_calib=2070` 的单位与物理含义（mm？0.1ns？），官方未给。

### 5.8 平滑/Kalman 滞后与"扫距"假象
库内 Kalman 是**时间序列平滑**：静止时收敛，**移动/换位置时会有几十帧的跟随延迟**；你们演示日志里 `dist_first` 单调下滑而 `RAW` 上下跳，正是"平滑输出 + 未收敛"的表现。
→ 测校准时**必须静置 ≥2~3 秒再取均值**，并记录 `smooth_num`（"连续有效测距的次数"，`slem_alg_smooth_dis.h:117`）作为"是否收敛"的标志。

---

## 6. 可调项实验矩阵（每条都是"一次烧录 = 一个数据点"）

> 前提：固定环境（桌面清空、两板同向竖直、离地约 1m、间距用卷尺量 0.5/1/2/3/5m），每次静置 3s 后取 30 组均值与标准差；记录 `dist_first / dist_second / prob / rssi / smooth_num / (tof_result)`。

| # | 变量 | 取值 | 目的/判据 |
|---|---|---|---|
| E1 | 校准模型 | 偏移法：`SCALE=1000, CALIB=avg_raw(1m)−1000`（≈1295mm） | **先做**。判据：0.5/1/2m 三点残差应显著小于现在（预期 1m≈0，0.5m −5~−8cm） |
| E2 | 库内 `calib_val` | `= avg_dist_first(1m) − 1.0`（先用 1m 单点） | 验证库是加还是减（§4.2 末尾的实验） |
| E3 | `ranging_method` | 4 → 1 → 6 → 3（**不要用 2**） | 看哪个在 1~3m 线性度/抖动最好 |
| E4 | `r_start` | 2（默认）→ 1 → 0 | 验证"~1.25m 常偏置是否与解模糊起始索引有关"（若 raw 随 r_start 每步变化 ~0.6m，即命中） |
| E5 | `threshold_cond2` | 20 → 30 → 40（提高判决门限） | 看 `dist_second` 跳变是否减少 |
| E6 | `freq_space` | 0 → 1 → 2 | 官方无推荐值，看对 raw 稳定性/量程的影响 |
| E7 | `refresh_rate`（+`posalg_freq` 同值） | 1 → 2 → 4 | 提高刷新率，看抖动是否可被平均掉 |
| E8 | 多点标定 | 0.5/1/1.5/2/3m 五点最小二乘（或分段） | 官方方法论：产品化推荐多点回归 + 记录拟合误差 |
| E9 | 打印 `dist_second`+`tof_result` | 打开 | 判断"解模糊跳变"和"TOF 是否生效" |
| E10 | 天线/姿态 | 板载天线同向竖直、抬高离地 1m、远离桌面金属；如需外接天线先改天线电阻 | 排除方向性偏置 |

---

## 7. 可执行的校准/调优步骤（建议直接照做）

**Step 0（10 分钟）**：开 `CONFIG_RANGE_LINK_VERBOSE=y`，打印 `dist_first / dist_second / prob / smooth_num / tof_result`；用**未校准**固件在 0.5/1.0/2.0m 各静置采 30 组。
判据：得到 raw 的"真值-读数"三点表 → 立刻能看出是"常偏置"还是"有斜率"（我们的预测：截距 ~1.26m、斜率 ~1.0）。

**Step 1（30 分钟）**：改回**偏移型校准**（官方法）：`SCALE_PM=1000`，`CALIB_MM = avg_raw@1m − 1000`（现在数据下 ≈1295）。复测 0.5/1/2m。
→ 预期：1m 归零；若 0.5m 仍偏小 5~8cm、2m 偏大 5~10cm，说明还有轻微斜率，进 Step 3。

**Step 2（可选，10 分钟）**：按 §4.2 的二分实验确认库内 `calib_val` 的符号/单位；确认后把它作为"库级"校准，应用层只做小残差修正（避免双重校准——你们现在 `calib_val=0` 是对的）。

**Step 3（1 小时）**：**多点标定**：0.5/1.0/1.5/2.0/3.0m 五点，最小二乘拟合 `true = a·raw + b`（或分段线性：0–1m、1–3m 各一套）。把 `a`、`b` 写进应用层（你们已有 `SCALE_PM/CALIB_MM` 机制）。**记录每点残差与标准差**（官方要求"记录拟合误差"）。

**Step 4（1~2 小时）**：参数扫描（§6 E3/E4/E5）挑抖动最小的一组；用"同一距离 30 组的 σ"作为唯一评判标准，而不是均值误差（均值可被校准吃掉，σ 才是精度上限）。

**Step 5（现场验收）**：按官方条件（3m 内空旷、5 方向）重做校准；给出"距离-误差-σ"表作为对外口径：
- 建议表述：**0.5~1m：±3~5cm；1~2m：±5~10cm；2~3m：±10~20cm（需按实测更新）**——不要引用"±0.5m/50m"这类无出处数字。

---

## 8. 误差原因排序（结论表）

| 排名 | 原因 | 证据强度 | 处理 |
|---|---|---|---|
| 1 | **原始 `dist_first` 在固定距离抖动 ±0.2~0.35m，且带 ~1.26m 常偏置** | 强（你们日志 + 两点反算） | 先做偏移校准 + 静置取均值 + 用 `smooth_num` 判收敛 |
| 2 | **校准模型用错：对"偏置型"数据做了 0.454 比例缩放** | 强（算术可复现 739mm） | 改偏移校准（§4.2/§7 Step1） |
| 3 | **解模糊多解（`dist_first`/`dist_second`）+ 库内 ESPRIT/KDE 选解** | 强（库符号 + 结构体字段） | 打印 `dist_second`，调 `r_start`/`threshold_cond2` |
| 4 | **近场 IQ 饱和/门限 + 远场相位噪声**（误差随距离上升） | 中强（反汇编硬门限 + 物理） | 避开 <0.3m；分距离段标定 |
| 5 | **多径/遮挡/天线方向**（官方要求 3m 无遮挡 + 五方向） | 中（官方文档明文） | 换官方条件复测；外接天线先改电阻 |
| 6 | **库内 Kalman/平滑滞后**（移动/换位时读数漂移） | 中（库符号 + 日志趋势） | 静置 2~3s 再取数；必要时用 `dis_ori` 做快速跟踪 |
| 7 | **TOF 未参与融合**（CS mode3 是否真生效未知） | 弱~中（宏已开、运行期未知） | 打印 `tof_result`；必要时调 `tof_calib` |
| 8 | 参数未调优（`freq_space/refresh_rate/posalg_freq` 全是样例默认） | 弱（官方无推荐值） | 按 §6 扫参数 |

---

## 9. 来源清单

**官方（高可信）**
1. `BS2XV100 SDK Sample使用指南`（SLE measure dis 章节：角色/镜像/校准原文/校准落点；SLE UART 章节：UART-L0 烧录、UART-L2 日志 115200）
   https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/debug/SDK%20Sample%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/SDK%20Sample%20%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.html
2. `SLE HADM` API（参数结构/回调/枚举，仅字段名，无单位与推荐值）
   https://docs.hisilicon.com/projects/fbb_ws63/master/api-reference/middleware/sle/service/hadm/hadm.html
3. `RSSI 测距` 官方案例（**校准方法论**：单点只校偏移、多点拟合 n、分段标定、记录拟合误差；并明确"要真距离选 HADM"）
   http://docs.hisilicon.com/projects/fbb_ws53/master/samples/connectivity/sle/link-mgmt/rssi-ranging/
   https://docs.hisilicon.com/projects/fbb_ws63/master/samples/connectivity/sle/link-mgmt/rssi-ranging.html
4. `星闪软件开发指导书`（BS2XV100；连接数上限 8、Channel Map 避 WiFi、Passkey 等）
   https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/software/星闪软件开发指导书/星闪软件开发指导书.html
5. `SP31 SLP 使用指南`/`SLP 接口`（`rangingFreq` 最高 100Hz、遥控器 1~20Hz；SLP 是另一套硬件方案）
   https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/software/SP31/软件开发指南/软件%20开发指南.html
6. HarmonyOS 术语：HADM = 往返时间或相位差，厘米级
   https://device.harmonyos.com/cn/docs/security/harmonyos-guides/fusion-connectivity-terminology
7. 星闪 1.0 白皮书（需求 ±5cm@<1m、±5%@>1m）（本地 PDF）
   `Docs\03_白皮书与报告\星闪无线短距通信技术（SparkLink1.0）产业化推进白皮书.pdf`

**官方代码（高可信，本地可复现）**
8. `SDK\bearpi-pico_h2821e\protocol\bt\algorithm\posalg\cal_dis_bs21\cal_dis_bs21_h\slem_disalg.h`（算法枚举+量程、r_start/threshold_cond2、calib_val/tof_calib 注释、IQ 门限）
9. `...\cal_dis_bs21_h\slem_alg_smooth_dis.h`（79 信道、`slem_smoothed_dis_result` 字段、`slem_alg_para_dis`）
10. `...\cal_dis_bs21_h\f2c.h:20-22`（`GLE_CS_MODE3_SUPPORT` 被强制定义）
11. `SDK\...\include\middleware\services\bts\sle\sle_hadm_manager.h`（CS 参数注释："acb_interval/cs_interval 由参数计算得到"）
12. `SDK\...\cal_dis_bs21\standard-bs21e-1100e\libcal_dis.a`（符号：ESPRIT/spatial_smoothing/KDE/Kalman；反汇编：`slem_alg_check_para` 的 IQ 5~2043 门限、method 2 非法、79/39 信道）
13. 官方样例 `application\samples\products\sle_measure_dis\...`（参数默认值、生效/死函数两处 `set_base_para`）

**社区（中可信）**
14. 小熊派公众号《Pico H2821E 迭代测距功能登场》（100m、"载波相位"、5m/10m 视频，无数字）
    https://my.oschina.net/u/4445410/blog/18621344
15. 电赛 C 题 星闪+UWB 数字钥匙（Kalman Q=0.05/R=0.1、>50cm 野值剔除、连续 5 次强制接受、差分平方解算；**其测距用 UWB，星闪只做 ID/通信**）
    https://developer.huawei.com/consumer/cn/blog/topic/03221347146763033
16. 亿佰特 E105-BS21 规格书（串口 115200、AT 形态，可作交叉验证）
    https://www.ebyte.com/product/2220.html
17. 你们的实测材料（本报告 §4.2/§5.1 的数据源）
    `Docs\range_link\README.md`（482/739mm 表）、`Docs\range_link\demo-evidence.log`、`Docs\design\range_link_design.md`

**低可信（不要引用）**
18. `jishuzhan.net/article/2006586343326531586` 及 CSDN 镜像：不存在的 API + 无出处指标，**疑似 AI 生成**。

---

## 10. 仍不确定 / 需实测确认

| 项 | 状态 |
|---|---|
| `dist_first` 单位 | 推断为**米**（日志 `%d.%03d`、官方"减 1"、`fix_pointf`/×1000 习惯）；官方未明写 |
| `calib_val` 是加还是减、单位 | 官方只说"加上校准值"；**用 §4.2 的二分实验一次确认** |
| `tof_calib=2070` 的含义/单位 | 官方未文档化 |
| `freq_space/refresh_rate/posalg_freq/glp_mode` 单位与推荐值 | **官方文档缺口**，只能扫参数 |
| `r_start` 与 ~1.25m 常偏置的因果关系 | 假设（默认 2 ↔ 观测偏置 1.26m）；E4 实验验证 |
| CS mode3/TOF 在整机上是否真生效 | 宏已定义，运行期需打印 `tof_result` 判定 |
| H2821E 官方精度指标 | 各官方文档均**未给出** |
