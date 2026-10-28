# 星闪 SLE 测距 / 定位 网络调研报告（BearPi-Pico H2821E / 海思 BS21E）

调研员：web-researcher ｜ 任务：task-4 ｜ 日期：2026-10-25
范围：官方文档、星闪联盟规范、小熊派资料、开源代码、公开实测数据。所有结论附 URL；
标注【不确定】= 未能用权威来源交叉验证；标注【低可信】= 来源疑似 AI 生成/营销稿，仅作方向参考。

---

## 0. 一句话结论

1. **能测距，精度"分米级到亚米级"**：官方 SDK 示例 `sle_measure_dis` 走的是**星闪 HADM（Channel Sounding：IQ/相位 + 可选 TOF）**，不是 RSSI；SDK 内部已经带平滑/置信度输出。规范对星闪测距的**需求指标**是「<1m 时 ±5cm，>1m 时 ±5%」、测角「10m 内 ±5°」；车钥匙/定位场景公开标称 0.6m 级、分米级。**1m 附近 ±5~10cm 是理想可达，几米外 0.2~0.5m 波动是常态**（需自己实测标定）。
2. **两块板只能得到"距离"（+置信度/RSSI/可选 TOF），得不到"坐标"**。要坐标必须 **≥3 个位置已知的锚点 + 自己解算**（三边/最小二乘）；星闪 1.0 的《基础应用层 位置信息管理》规范里定义了"锚点/标签/管理节点"角色和多锚点解算能力位，但**海思 SDK 没有现成的定位示例**。cm 级定位要换 **SLP（需外挂 SP31 芯片 + 多天线板）** 或 UWB。
3. **最权威的教程不在网上，而在 SDK 文档**：`BS2XV100 SDK Sample使用指南` 里有 `SLE measure dis` 与 `SLE UART` 的完整编译/烧录/校准（1m 五方向）步骤；小熊派官网有**免编译的"H2821E 测距固件 / SLE 透传固件 / AT 固件(1对8)"**可直接下载体验。

---

## 1. 小熊派 BearPi-Pico H2821E 官方资料

| 内容 | 链接 | 要点 |
|---|---|---|
| 产品页（H2821E） | https://www.bearpi.cn/core_board/bearpi/pico/h2821E/ | BS21E，RISC-V 64MHz/160KB SRAM/1MB Flash；UART×3（最大 4Mbit/s）、USB2.0；**板载 2.4G 天线 + IPX 天线座（"使用前需要调整天线电阻"）**；官方注："星闪通信需双星闪设备使用，建议成对购买" |
| **固件中心**（关键） | https://bearpi.cn/core_board/bearpi/firmware/ | 共 7 款固件，其中有 **H2821E 测距固件**、**H2821E SLE 透传固件**、**H2821E AT 固件(SLE 版本，1对8)** —— 不写代码即可先验证硬件/天线/串口 |
| 官方公众号：H2821E 测距上线 | https://my.oschina.net/u/4445410/blog/18621344 （原文 https://mp.weixin.qq.com/s?__biz=MzI3NjE4MTIwMw==&mid=2247488463&idx=1&sn=7722bc429ae7c89e3d7b33c3f49755ba ） | 明说"星闪技术采用**载波相位测量**"；"工程师实测 BM H21E 模组实现 **100 米精确测距**"；"已发布 **5 米与 10 米测距场景的误差实测视频**"（正文未给具体误差数字）；延展特性提到"多信标节点协作定位（三角/多边定位算法）" |
| 官方公众号：1 对 8 通讯（AT 固件） | http://mp.weixin.qq.com/s?__biz=MzI3NjE4MTIwMw==&mid=2247488205&idx=1&sn=b568bd8a312cbfc49869e36a44fca6e9 | AT 固件体验 1 对 8，串口 115200，需 CH340 驱动 |
| Gitee SDK 镜像 | https://gitee.com/chentao_cc/bearpi-h2821_pico ｜ https://gitee.com/diemit/fbb_bs2x | 底层协议栈只给库、应用层开源（NDA 才给底层） |
| 海思案例库（板卡介绍） | https://developers.hisilicon.com/cn/caselibrary/34/xxp_bs21kfb | BS21 系列：SLE1.0、1/2/4MHz 带宽、SLE 网关、最大空口 12Mbps |
| 第三方测评（eeworld 汇总） | https://datasheet.eeworld.com.cn/view/14253865.html | 拉距/功耗/丢包/SLE client 源码解读等系列（**不含测距专项**） |
| 评测：BLE 串口透传 | https://en.eeworld.com.cn/bbs/forum.php?mod=viewthread&tid=1291470 | 双板 BLE 透传的 GATT 流程，可对照 SLE 的 SSAP |

> 教程站说明：bearpi.cn 的教程页是 `https://bearpi.cn/core_board/bearpi/pico/h2821/...`（如"环境搭建windows"），
> 站点是 SPA/需登录跳转，**没有直接抓到"SLE 测距"专门章节**；能找到的官方"教你怎么用双板"的文字教程，
> 最好的是 SDK 文档（见 §2）与 NLChat 的双板透传教程（见 §5）。

---

## 2. SLE 测距原理与精度（重点：是不是 Channel Sounding）

### 2.1 结论：是 Channel Sounding（IQ/相位），带 TOF 与 RSSI 辅助

**本地 SDK 源码证据**（`SDK\bearpi-pico_h2821e\application\samples\products\sle_measure_dis\`）：

- `sle_measure_dis_server_alg.c` 里用的是 **HADM / Channel Sounding 接口**：
  `sle_hadm_register_callbacks()`、`sle_channel_sounding_iq_report_t`（**IQ 采样**）、
  `cs_iq_report_cb`、`tof_result`（`GLE_CS_MODE3_SUPPORT` 下启用）、`rssi`、`es_sn`、`timestamp_sn`、`samp_cnt`；
- 距离由 `slem_alg_calc_smoothed_dis(&result_dist, alg_para)` 算出，算法参数含
  `calib_val`（校准值）、`tof_calib`（默认 2070）、`ranging_method`（默认 4）；
- 输出结构 `measure_dis_profile_msg_dis_t` 字段：
  `dist_first`（平滑后距离）、`dist_second`（原始）、`dist_double`（轻平滑）、`high`、`prob`（置信度）、`rssi`、`smooth_num`；
- 日志格式：`SLEM get distance done. distance:%d.%03d, time = %d ms.`（**距离单位应为 m，带 3 位小数**，每次测量还打印耗时 ms）。

**架构含义**：client 把本端 IQ 数据通过 SSAP 上报给 server，server 侧跑算法算距离 —— 这正好对应官方文档
"server 端(广播)与 client 端(扫描)建立 sle 连接，**server 端计算出距离**"。

命名上 `GLE_CS_*`/`sle_channel_sounding_*` 与 BLE Channel Sounding 高度同源 →【推断】星闪 SLE 的 HADM
复用了同一套通道探测（IQ/相位+TOF）引擎，所以"星闪测距 ≈ BLE CS 的星闪低功耗版"，精度同量级。

华为侧也印证"星闪 HADM 测距"是正式能力名：HarmonyOS API `@ohos.FusionConnectivity.ranging` 中有
`nearlinkHadm`（星闪 HADM 测距类型是否支持）：
https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/api/js-apis-fusionconnectivity-ranging

### 2.2 官方文档原文（最有用的一页）

`BS2XV100 SDK Sample使用指南`（HTML）：
https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/debug/SDK%20Sample%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/SDK%20Sample%20%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.html

**"SLE measure dis" 章节原文要点**：
- 概述：server(广播) 与 client(扫描) 建 SLE 连接，**server 算距离**；
- 编译：menuconfig → `Enable SLE MEASURE DIS Server sample` / `Enable SLE MEASURE DIS Client sample`（**两块板=两个镜像**）；
- 运行：两块板分别烧 server/client，串口打印 `Connected`，server 侧持续打印距离；
- **校准**（与你说的"1m 五方向"完全一致）：
  - 测试环境："锚点周围 **3m 范围内空旷无遮挡、无墙体、柱体、金属等遮挡物**"；
  - 方法："按照图 1 所示在设备周围 **5 个方向的 1m 位置，进行 5 次测距，5 次测距值的平均值减 1 为锚点 A 的校准值**"；
  - 落点：把校准值加到 `application\samples\products\sle_measure_dis\sle_measure_dis_server\sle_measure_dis_server_alg.c`。

同页 **"SLE UART Sample"** 原文要点（串口坑）：
- 两块单板，**每块把 UART-L0 和 UART-L2 都引出到 PC**；
- **用 UART-L0 烧录** Server/Client 版本；
- PC 用两个串口助手接两块板的 **UART-L2，波特率 115200**，互发字符验证透传。

同页 **"SLE UART DMA Sample"**（如果之后要传数据流）：
- 跑流最高 **2.2Mbps**；默认连接间隔 25ms、每包 236B；
- **TRx buffer 260 时空口最大包 252B**；应用层每次建议 **≤236B 的整数倍**；
- "UART 跑流速度一般为 波特率×0.6 左右"（4Mbps 波特率 → ~2.4Mbps）。

### 2.3 连接数 / 测距周期

- **并发连接上限 8**：`星闪软件开发指导书`（BS2XV100 系列）原文："只作为 SLE 设备工作时，最多只支持**八路连接**；
  同时作为 BLE 和 SLE 设备工作时，一共支持八路连接。"
  https://docs.hisilicon.com/repos/fbb_bs2x/zh-CN/master/software/%E6%98%9F%E9%97%AA%E8%BD%AF%E4%BB%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%AF%BC%E4%B9%A6/%E6%98%9F%E9%97%AA%E8%BD%AF%E4%BB%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%AF%BC%E4%B9%A6.html
- **测距频率**：SDK 里 server 连接间隔 `conn_interval = 0x64`（=100 × 1.25ms = **125ms**），client `cs_interval = 1`
  →【代码推断】默认约 **8Hz（~125ms 一轮）**。官方没有给 measure_dis 的标称测距频率；
  参考另一形态 SLP：`rangingFreq` 最高 100Hz、遥控器场景 1~20Hz（见 §2.4）。
- **同频干扰避让**：SLE 有 Channel Map 接口，官方示例建议 WiFi 信道 6 时避让 **2427~2447**（同上"星闪软件开发指导书"）。

### 2.4 精度数据（横向对照，注意区分"规范需求/标称/实测"）

| 来源 | 数值 | 性质 |
|---|---|---|
| 星闪 1.0 产业化白皮书（**本地** `Docs\03_白皮书与报告\星闪无线短距通信技术（SparkLink1.0）产业化推进白皮书.pdf`） | 手机-耳机：**<1m 时 ±5cm，>1m 时 ±5%**；两天线测角 10m 内 **±5°** | **需求指标**（对星闪的要求，不是某颗芯片实测） |
| 同白皮书（车钥匙章节） | "测量钥匙或手机的距离，**精度到 cm 级别**" | 场景需求 |
| 星闪联盟 `T/XS 30007-2025 基础应用层 位置信息管理`（本地 PDF） | 只定义服务/属性/能力位，**不含精度指标** | 规范 |
| SLP（高精度定位，需 **SP31/SP31C 芯片 + 多天线板**）官方 SDK 发布说明 | **SLP 测距 ±10cm，测角 ±5°** | 官方标称（不是 H2821E 单板能力） |
| 星闪车钥匙报道 | 星闪 SLE **相位差测距 + TOF**，精度可达 **0.6m**；比 BLE CS 早，**SLE 可同时测 4 个 channel**（BLE CS 单音一次一个） | 媒体（可信度中） |
| 技象科技 | SLE 定位精度 **0.6m**；"MLE/高性能模式" 0.1~0.3m | 厂商博客（可信度中） |
| watchdata 星闪 FOBKEY | SLE 相位+TOF "**分米级**精准测距"，对比传统蓝牙 RSSI 2~5m | 厂商稿 |
| 小熊派官方 | **100m 精确测距**；5m/10m 误差实测**视频**（正文无数字） | 官方宣传（缺数字）【不确定具体误差】 |
| IEEE（SparkLink SLB 室内车辆感知） | 地下车库 **定位中位误差 1.13m（静止）/1.32m（动态）** | 学术实测（SLB 多锚点）https://xplorestaging.ieee.org/document/11269909 【链接为 staging，可能不稳定】 |
| CSDN 星闪系列教程 | "星闪的定位精度只有**分米级**" | 个人教程 |
| 专利 CN116101223A | "星闪测距精度能精确到 **5cm**" | 专利撰写口径【低可信】 |

**结论口径建议**：对领导/用户说"**1m 内 ±5~10cm、3~10m 内 ±5%~±0.5m、100m 仍可出数但误差明显变大**"，
并以自己实测的 5m/10m 数据表为准；不要引用"±0.5m（10m 内）/最大 50m"那类数字 ——
它来自下面这条**疑似 AI 生成**的文章。

### 2.5 ⚠️ 需警惕的低可信来源

`华为海思 BS21E (H2821E) 星闪组网测距定位 技术可行性方案`
https://jishuzhan.net/article/2006586343326531586 （镜像 https://blog.csdn.net/bother3000/article/details/156452322 ）
- 出现 **`ai_bs21_sle.h` / `ai_bs21_sle_start_ranging()` / `AT+SLERANGE`** 等**在海思 SDK/AT 手册中不存在的 API**（真实 API 是
  `sle_*`/`ssaps_*`/`sle_hadm_*`；安信可 Ai-BS21 的 AT 指令也不是这套）；
- 给出"±0.5m（10 米内）/ 最大 50 米 / 单 G 节点 200+ T 节点 / 休眠 <1mA"等整齐数字，**均无出处**；
- 其"三边定位代码"本身（最小二乘公式、滑动平均、Kalman）可当**算法模板**看，但**参数与硬件结论不要引用**。

---

## 3. 星闪 1.0 的"定位"到底提供什么（能否得到坐标）

### 3.1 规范：《星闪无线通信系统 基础应用层 位置信息管理》T/XS 30007-2025

- 标准信息（起草单位：华为、中国信通院、中汽中心、长安、重邮、中科晶上、联想、爱旗等）：
  https://www.spc.org.cn/online/94f71d404b77d34a532e3b581ce29859.html
- 测试标准 `T/XS 50011-2025`（2025-03-26 发布实施）：https://old-page.sparklink.org.cn/standard/view.php?id=63
- 团体标准库条目：https://www.ttbz.org.cn/standardDetail/lmejc1ksa0rxjwaxx457agkqeo9azxc.html
- **本地全文**：`Docs\02_星闪规范\星闪无线通信系统 基础应用层 位置信息管理.pdf`（以下引用其正文）

**规范定义的内容（原文归纳）**：

1. **三类角色**：`位置信息管理节点`（G 节点，服务端/客户端）、`位置信息锚点`、`位置信息标签`；
   "位置信息锚点不能同时为位置信息标签"；锚点可同时是管理节点。
2. **服务属性**（首要服务，可被其他服务引用）：
   - `<<设备类型指示>>`（bit0 管理节点 / bit1 锚点 / bit2 标签）
   - `<<固定合作设备集合标识>>`、`<<临时合作设备集合标识>>`（6 字节，安全相关）
   - **`<<SLB测量能力指示>>` / `<<SLE测量能力指示>>`**（4 字节能力位图）
   - **`<<SLB解算能力指示>>` / `<<SLE解算能力指示>>`**（4 字节能力位图）
   - **`<<锚点相对位置信息>>`、`<<锚点绝对位置信息>>`**（锚点坐标！）
   - `<<标签相对位置信息>>`、`<<标签绝对位置信息>>`
3. **SLE 测量能力位**（证明 SLE **原生支持**测距/测角类测量信号）：
   bit0/1 AoD 发送/接收，bit2/3 AoA 发送/接收，bit4~5 双边单消息 TOF，bit6~7 双边两消息 TOF，
   bit8~9 双边三消息 TOF，bit10~12 **双边两消息复合信道（含跳频虚拟大带宽）**，bit13~16 多锚点组播，
   bit17 端口/bit18 波束/bit19 天线切换，**bit20/21 多音信号发送/接收**。
   （SLB 的位图基本一致，少"多音"两位）
4. **解算能力位**（bit 举例）：AoD/AoA 角度解算；多锚点 AoD/AoA 定位解算；**基于双边单/两/三消息 TOF 的距离解算**；
   **多锚点 TOF 定位解算**；**基于双边两消息复合信道的距离/定位解算**；AoD/AoA + 距离联合解算。
   → "解算方法包括：基于角度+距离解算、**基于多个锚点的距离测量信息解算**、基于多个锚点的角度测量信息解算等"。
5. **坐标系构建**（原文 9.4 附近）："**多锚点构建 xoy 平面：至少 3 个锚点的 z 坐标为 0**，该三个锚点构成 xoy 平面；
   其中两个锚点 y 坐标为 0，连线为 x 轴……"；单锚点只能建本地坐标系。
6. **接入关系**：锚点/标签是接入层 **T 节点**，位置信息管理节点是 **G 节点**；数据用 **SSAP** 交互。
7. **附录 B（参考性）**：**基于 SLE 的数字钥匙**应用，主机/锚点/标签的 `<<位置信息管理>>` 属性参考值，
   例如 `<<SLE解算能力指示>> = 0x00018000`（bit15/16：双边两消息复合信道距离解算 + 多锚点定位解算），
   `<<SLE测量能力指示>>` 含 `0x00010000`（多锚点组播）等。

### 3.2 结论：两块板能不能"定位"

- **规范层面**：SLE **可以**承担测距，也可承担多锚点定位解算（规范已定义能力位与坐标属性）；
  "位置"由**锚点坐标 + 多锚点测距**解算得出，**不需要**像 UWB 那样的 TDoA/ToA 时间同步组网。
- **芯片/SDK 层面**：H2821E（BS21E）单板 **单天线**，`sle_measure_dis` 只给**距离**；
  **没有**现成的"位置信息管理"服务实现，也**没有**测角（AoD/AoA）示例 →
  **两块板只能"测距 + 数据通信"**；要坐标必须**自己加锚点**：
  - 2D 定位：**≥3 个固定锚点**（坐标已知，别布置在一条直线上）→ 三边/最小二乘；
  - 更稳：4 锚点；再加状态机/滤波；
  - 3D/测角：需要多天线或 SLP，H2821E 单板做不到。
- **并发**：若 1 块板当"标签"、多块当"锚点"，注意**SLE 最多 8 路连接**，
  且 `sle_measure_dis` 是 **1 对 1** 的业务，要改成多连接（可参考 SDK 的 `sle_multi_conn` 示例）轮询测距。
- **SLB vs SLE 定位区别**：SLB 是 OFDM 高速模式，定位同样靠"测距/测角 + 多锚点解算"（白皮书/架构规范），
  **公开资料未见 SLB 采用 UWB 式 TDoA 的芯片实现**【不确定】；星闪真正的"UWB 级"是 **SLP**：
  白皮书原文"**星闪 SLP 采用 UWB 信号实现厘米级高精定位**，通过 SLE/SLB 辅助的宽窄融合定位，
  在测距数据帧时采用 UWB 信号，其余（发现/连接/配置/结果交互）用 SLE/SLB" ——
  https://www.cvianet.org.cn/static/files/file-27669.pdf （指向遥控白皮书）
  → 也就是说 **cm 级不是 H2821E 单板能做到的**，要 SLP 芯片（SP31/SP31C）或多天线板。
- 星闪三模式（SLB / SLE / **SLP 同步高精度定位**）见国标草案（接入层）：SLP "采用新型帧格式、高脉冲功率、
  超宽带随机启动及信道训练序列(CTS)"，提供测距/测角/存在感知/目标跟踪：
  搜索关键词见 https://std.samr.gov.cn/ （文件：`星闪无线通信系统 接入层 同步低功耗空口.pdf`）
- 架构规范里还有"相邻 G 节点之间的**测距信息交互**/联合测距"（本地 `Docs\02_星闪规范\星闪无线通信系统 架构.pdf`），
  说明多 G 节点协同定位在标准里是留了口子的【不确定具体产品支持情况】。

---

## 4. 常见坑清单（来自官方文档 + 代码 + 社区）

### 4.1 镜像 / 角色
- **两块板必须烧不同镜像**：server 端 `Enable SLE MEASURE DIS Server sample`，client 端 `Enable SLE MEASURE DIS Client sample`
  （官方文档 + 代码里的 `CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS_SERVER/CLIENT` 条件编译）。
- 角色不对称：**server 广播、client 扫描并发起连接；距离由 server 算**（client 把 IQ 上报给 server）。
  所以"看距离"的串口在 server 端（除非自己把结果回传）。日志关键字：`Connected` / `SLEM get distance done`。
- 连接/广播参数：server `announce_interval_* = SLE_ADV_INTERVAL_SLAVE_DEFAULT`、`conn_interval = 0x64`(125ms)；
  client 扫描间隔 `seek_interval[0] = 0x12c0`（代码值）。
- 改角色后**先复位、再上电**；多设备时改 MAC/名称，避免连错（SDK 多连接示例里也是这么配的）。

### 4.2 串口 / 烧录
- **UART-L0 = 烧录口，UART-L2 = 日志/数据口**（官方 SLE UART 章节原文：用 UART-L0 烧录，PC 连 UART-L2）；
  **波特率 115200、8N1**；需要 CH340/CP210x 类驱动。
- UART-H0(L?) 与日志冲突：官方提醒"使用 H0 串口时请关闭日志线程，防止日志与 uart 使用同一串口异常"。
- UART 最高 4Mbit/s；`UART 跑流速度 ≈ 波特率 × 0.6`。
- 双板互连透传要 **TX-RX 交叉 + 共地**；只用 USB-TypeC 供电时注意两块板不要共地环路。

### 4.3 天线 / 环境（精度杀手）
- 官方校准条件："锚点周围 **3m 内空旷无遮挡，无墙体、柱体、金属**" → 金属/人体/墙体遮挡会让读数明显偏大或跳变。
- 板载 2.4G 天线；**若改用 IPX 外接天线，"使用前需要调整天线电阻"**（小熊派产品页原文）。
  这是常见坑：不改电阻会出现外接天线几乎无增益甚至更差。
- 两块板**同向、竖直、离地 1m 左右、避开人体**；不要贴桌面/金属板。
- WiFi 干扰：用 SLE Channel Map 避让（官方示例：WiFi ch6 → 避让 2427~2447）。
- 双模共存：BLE+SLE 总共 8 连接，别同时开太多。

### 4.4 校准
- **1m 五方向校准**（官方）：5 个方向各 1m、每方向多次 → 取 5 次测量均值 **减 1** 得校准值 →
  写入 `sle_measure_dis_server_alg.c`（对应 `alg_para->calib_val`；默认 0，`tof_calib` 默认 2070）。
- 建议**自己做多点（0.5/1/2/3/5/10m）标定表**，拟合 linear/affine（斜率+截距），
  电赛项目就是用 `r = (avg - 61.0) / 1.105` 这种实测拟合（见 §5）。
- 校准是"每套硬件/每个天线姿态"的事，不要跨板复用。

### 4.5 漂移 / 滤波
- SDK **已内置平滑**：输出 `dist_first`（平滑）/`dist_second`（原始）/`dist_double`（轻平滑）/`prob`（置信度）/`rssi`。
  实战建议：**用 `prob` 做野值门限**，用 `rssi` 做有效性判断（代码里 `DIS_ALG_RSSI_LIMIT = -120`）。
- 通用做法（社区/电赛验证过）：
  1. **滑动平均/EMA**（窗口 5~10；EMA α≈0.2~0.3）；
  2. **野值剔除**（偏离预测 >0.5m 丢弃，连续丢弃 N 次（如 5 次）后强制接受，防卡死）；
  3. **Kalman**（1D，Q≈0.01~0.05、R≈0.1），对速度有要求时再加匀速模型；
  4. 静态场景加"静止检测 + 慢滤波"，动态场景降平滑。
- 只有在需要坐标时才做三边解算；**不要用未滤波的原始距离直接解算**（圆不交于一点，坐标乱跳）。

---

## 5. 现成代码 / 教程 / 上位机

| 项目 | 链接 | 价值 |
|---|---|---|
| **官方 SDK 示例（最该用）** | 本地 `SDK\bearpi-pico_h2821e\application\samples\products\sle_measure_dis\`（server/client/slem/alg） | 测距全链路：广播→连接→CS 能力交互→IQ 上报→算法→串口输出；`sle_uart` 提供透传 |
| 官方文档（编译/烧录/校准步骤） | 见 §2.2 URL | 唯一完整的"官方教程" |
| 小熊派固件中心 | https://bearpi.cn/core_board/bearpi/firmware/ | 测距固件/透传固件/AT 固件，**零代码先跑通** |
| NLChat（SLE 透传 + 安卓 APP） | https://github.com/Hny0305Lin/NLChat ；双板透传教程 https://github.com/Hny0305Lin/NLChat/blob/v1.3.88.2024.0706/SLE_Device_Configure_Tutorial.md | Kconfig 选 server/client、双板串口互发的最短路径教程（H2821 系列） |
| SLE 双板串口透传实战（H3863，流程/日志可照搬） | https://blog.csdn.net/qjh2321/article/details/168453107 ；https://aiot.csdn.net/69df62980a2f6a37c5a021d3.html | announce/seek/配对/SSAP/服务发现全流程日志 |
| **电赛 C 题：星闪 SLE + 4 锚点 UWB 数字钥匙（开源）** | 文章 https://developer.huawei.com/consumer/cn/blog/topic/03221347146763033 ；代码 https://gitcode.com/gcw_rVSV2mp6/diansai-C | **最佳"工程化"参考**：Kalman（Q=0.05/R=0.1）、野值剔除（>50cm 丢弃、连续 5 次强制接受）、差分平方解算、角度环绕处理、三级区域状态机、OLED 显示、双 UART 收锚点数据（115200）。注意：它**测距用 UWB**，星闪只做 ID/通信 —— 恰好说明"cm 级靠 UWB" |
| SLE Mesh 图传（H3863） | https://github.com/NearLink-ePaper/NearLink-Mesh-ePaper | SLE 多跳 + 数据可靠传输，工程量大但可参考 |
| H2821 对讲机（语音透传验证距离） | https://oshwhub.com/liwenke302/demo-hi2821 | 双板远距离数据/语音透传的硬件与玩法 |
| 亿佰特 E105-BS21（AT 模块） | https://www.ebyte.com/product/2220.html ；规格书 http://m.ebyte.com/downpdf/2220.html | AT 指令形态（`AT+ROLE`/透传/115200），若不想写代码可用模组快速验证 |
| RSSI 测距官方示例（WS63，**对比用**） | https://docs.hisilicon.com/projects/fbb_ws63/master/samples/connectivity/sle/link-mgmt/rssi-ranging.html | 7 点中值滤波 + EMA + 对数距离路径损耗模型 + **1m 单点校准**（`CONFIG_SLE_RSSI_RANGING_RSSI_AT_1M` 默认 -45dBm）；说明"RSSI 方案需要标定，精度远不如 CS" |
| 星闪 1 对 8 / AT | http://mp.weixin.qq.com/s?__biz=MzI3NjE4MTIwMw==&mid=2247488205&idx=1&sn=b568bd8a312cbfc49869e36a44fca6e9 | 多连接体验 |

**上位机**：
- **未找到**专门的"星闪 SLE 测距上位机"（Python/Qt）开源项目【不确定：可能存在于收费/闭源工具或 QQ 群】。
- 可用的替代：① 串口助手直接看 server 打印；② 自己写 **Python + pyserial + matplotlib/pyqtgraph** 实时曲线
  （数据就是 `distance:%d.%03d`、`prob`、`rssi` 三列，10 行代码搞定）；
  ③ 多连接/数据路线可参考 NLChat 的安卓端思路，或把数据经 UART-L2 打到 PC。

---

## 6. 英文关键词检索结果（简要）

- `HiSilicon BS21 SLE ranging` / `SparkLink SLE channel sounding` / `SLE distance measurement`
  → 命中主要是**海思官方文档站**（fbb_bs2x：`SLE measure dis`、SLP、RSSI ranging）与**安信可/亿佰特/润和**模组规格书；
  BS21 系列规格书原文（FCC 版）明确写 "Supports SLE1.0 protocol ... **high precision ranging**"：
  https://fcc.report/FCC-ID/2ATPO-AIBS21/7434976.pdf
  以及 `AI-BS21-32S Specification`：https://docs.ai-thinker.com/_media/ai-bs21-32s_v1.0.0_specification-20240305.pdf
- 学术侧：SparkLink 定位论文（SLB 多锚点、通道拼接）中位误差 ~1.1~1.3m：
  https://xplorestaging.ieee.org/document/11269909 【链接可能失效】
- **未检索到**独立第三方对 BS21E 测距精度的系统测评（只有小熊派自家视频）【不确定】。

---

## 7. 对"两块 H2821E 做数据传输 + 测位置"的可行建议

### 7.1 先厘清目标
- **只要"距离/接近判断"（如防丢、靠近解锁、区域提示）→ 两块板就够**，用 `sle_measure_dis`。
- **要"平面坐标 (x,y)" → 两块板不够**，需 ≥3 锚点 + 主机解算；**要 cm 级 → H2821E 做不到**，
  需 SLP（SP31/SP31C + 多天线板）或 UWB。

### 7.2 推荐落地路线（低风险→高价值）
**阶段 0（半天，零代码）**：烧小熊派**固件中心**的「H2821E 测距固件」+「SLE 透传固件」，
确认双板能连、能出距离、UART-L0/L2 与 115200 都通，记录 1m/3m/10m 的原始抖动幅度。
→ 目的：先判断"这块板+这个环境"的精度底线，避免后面写代码时怀疑人生。

**阶段 1（1~2 天，官方示例）**：编译 `sle_measure_dis` 的 server/client 两个镜像，
- 做官方 **1m 五方向校准**（5 方向 × 5 次，均值-1 → `calib_val`）；
- 采集 0.5/1/2/3/5/10m 各 50 组的 `dist_first/dist_second/prob/rssi`，**拟合 affine 标定**；
- 写一个 Python 上位机画三条曲线（原始/平滑/置信度），把"精度-距离"曲线固化下来。

**阶段 2（2~3 天，测距 + 数据一体）**：在 `sle_measure_dis` 的 SSAP server 上加一个**自定义 Property**，
用 `ssaps_notify_indicate` 把"距离 + 你自己要传的数据" 一起 notify 给 client（或反向）；
- 包长 ≤236B，一次别超 TRx buffer（260 → 空口 252B）；
- 消息用 TLV/自定义头，带 seq 与时间戳；（官方示例本身就在用 SSAP 传 IQ，证明"测距时传数据"完全可行）

**阶段 3（按需，定位）**：
- **方案 A（推荐给别人看/答辩友好）**：3~4 块 H2821E 当**固定锚点**（坐标人工量好）、1 块当标签，
  标签轮询各锚点测距（注意 8 连接上限 / 参考 `sle_multi_conn`），主机侧做**三边+最小二乘+Kalman**；
  预期 2D 误差 **0.3~1m**（视遮挡/布局），别承诺 cm。
- **方案 B（要 cm 级）**：加 UWB（DW1000/DW3000）做测距，星闪只做身份/数据/组网 ——
  这正是电赛开源项目的架构（星闪 SLE 负责 ID+通信，UWB 负责 cm 级测距）。
- **方案 C（产品化）**：用海思 SLP 方案（SP31C + 三/多天线板），官方标称 ±10cm/±5°，
  但**超出 H2821E 单板，属于换硬件**。

### 7.3 明确的注意事项（避免踩坑）
1. **必须两个镜像**（server/client），别想着"一块板同时当 server 和 client"（示例不支持）。
2. 距离在 **server 端**打印；client 端不打印距离 —— 想两端都显示要在 client 侧解析或加回传。
3. **UART-L0 烧录、UART-L2 看日志、115200 8N1**；不要用 L0 看日志。
4. 用外接 IPX 天线前**改天线电阻**，否则等于没接。
5. 校准值**每套板/每种姿态单独做**；测量时避开金属、人体、墙面，两块板同向竖直。
6. 别直接相信单次值：**用 `prob`/`rssi` 门限 + 滑动平均 + 野值剔除**，静止场景可上 Kalman。
7. 精度承诺要保守：**1m 内 ~±5~10cm、3m 内 ~±20~30cm、10m 量级 ~±0.5m** 是合理区间【不确定，需实测确认】。
8. 文献里"±0.5m / 50m / 200 并发"的说法来自**疑似 AI 生成文章**，不要写进方案书（见 §2.5）。

---

## 8. 待验证 / 不确定清单

| 项 | 状态 |
|---|---|
| H2821E 单板 SLE 测距的**官方精度指标** | 海思文档**未给**；只有小熊派"5m/10m 实测视频"（未抓到数字）→ 需自测 |
| `dist_first` 单位是否 m | 由打印格式 `%d.%03d` 与"距离"语义推断，**未在文档中明写** |
| measure_dis 默认测距频率 | 由 `conn_interval=0x64` 推断 ~125ms/轮，**文档未明写** |
| SLE 是否支持测角（AoD/AoA）落在 H2821E 上 | 规范有 AoD/AoA 能力位；润和 HH-D03（BS21E）规格书写"支持测距、测角"，但小熊派 H2821E 单板单天线 + SDK 无示例 → **单板不可用**【不确定】 |
| SLB 是否有 TDoA 定位实现 | 未找到公开芯片/SDK 资料；星闪定位以"测距+测角+多锚点解算"为主【不确定】 |
| 是否有 SLE 测距专用上位机 | 未找到开源项目【不确定】 |

---

### 附：本报告引用的本地文件（可让 lead 直接查证）
- `Docs\02_星闪规范\星闪无线通信系统 基础应用层 位置信息管理.pdf`（T/XS 30007 系列，属性/能力位/角色/坐标系）
- `Docs\03_白皮书与报告\星闪无线短距通信技术（SparkLink1.0）产业化推进白皮书.pdf`（±5cm/±5%/±5° 需求指标、车钥匙 cm 级）
- `Docs\02_星闪规范\星闪无线通信系统 架构.pdf`（G 节点间"测距信息交互"）
- `SDK\bearpi-pico_h2821e\application\samples\products\sle_measure_dis\`（HADM/CS/IQ/TOF 实现、平滑与置信度）
