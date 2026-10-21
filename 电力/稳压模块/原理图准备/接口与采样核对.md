# 稳压模块单页原理图接口与采样核对

日期：2026-10-10。对应现行 V1.3 候选架构。此文件是原理图准备资料，不是已完成网表、原生 ERC、样机验证或安全认证。范围只覆盖核心板接口、采样路由、COMP 候选、BW6101 与 TCAN332 的引脚/基本外围；未修改核心板，不调用 EDA 会话。

## 原理图边界与页面组织

全电路保持在**一个 SCH_PAGE 和一张板**。本页可分六个功能区，但不是六张页：①辅助供电、WeAct 核心板与 CAN；②双向单半桥、DRV8161 与电流取样；③8 路累积抽头采样（PB14 同时作为 SC_SUM/T8 ADC）与 TLV6700；④8 路 BW6101 单体被动泄放；⑤BUS 软件 dump、UCC27511A 与三处温度；⑥BUS、SC、tap harness、CAN、探头及装配注释。功率回路放在页中部，采样/保护置于近端，外部超容和 dump 电阻只画连接器与说明，不另建 PCB。

当前骨架记录为一板、一原理图、一页；尚无实际放置的约 180 个器件可供精确计数。因此图幅尚未冻结：先以 A1 横向排版检查可读性；按实际器件、注释和线网数量复核，若不能保持引脚/网络名清楚，再改 A0。不要用分页或隐藏页解决拥挤。

## WeAct Normal QFN48 核心板排针映射

依据本地 WeAct Normal QFN48 V1.0 原理图 `FOC驱动审阅/硬件/数据手册/WeAct_normal_QFN48_V1.0.pdf`（SHA256 `57ECA2E3467ABD45BD6D4BC993EB11E9C83389A7C390DA2543C3F20CFC2C640D`）。每个连接器为 2×12、2.54 mm；下表是板端丝印/原理图映射，不表示本模块连接器型号已定。

| P1 脚 | 核心板网络 | 本模块候选用途 | P2 脚 | 核心板网络 | 本模块候选用途 |
|---:|---|---|---:|---|---|
| 1–2 | VCC | +5V_CORE 输入 | 1–2 | 3.3V | MCU/逻辑供电参考，不外接驱动电源 |
| 3–4 | GND | GND_SYS | 3–4 | GND | GND_SYS |
| 5 | PB12 | NTC_DUMP / ADC1_IN11 | 5 | PB10 | TEST_CTRL，数字 IDR 状态回读 |
| 6 | PB13 | TIM1_CH1N 互补 PWM | 6 | PB11 | NTC_MAIN_MOS / ADC12_IN14 |
| 7 | PB14 | SC_SUM_ADC / T8 累积抽头 / ADC1_IN5 | 7 | PB2 | 板载 VBUS 检测输入，勿复用/改板 |
| 8 | PB15 | TEST_FB / ADC2_IN15（经 10k/10k 分压） | 8 | VREF+ | 按板载参考连接；本模块不另驱动 |
| 9 | PC6 | 保留 | 9 | PB0 | BUS_ADC / ADC1_IN15 |
| 10 | PA8 | TIM1_CH1 高侧 PWM | 10 | PB1 | DRV_SO / ADC1_IN12 / COMP1_INP |
| 11 | PA9 | 保留；PB6 关联焊桥须核 | 11 | PA7 | DRV_SO 分支 / COMP2_INP（ADC2_IN4 能力不使用） |
| 12 | PA10 | 保留 | 12 | PC4 | NTC_SC / ADC2_IN5 |
| 13 | PA11 | FDCAN1_RX，AF9 | 13 | PA5 | T6 / ADC2_IN13 |
| 14 | PA12 | FDCAN1_TX，AF9 | 14 | PA6 | T7 / ADC2_IN3 |
| 15 | PA15 | 保留 | 15 | PA3 | T4 / ADC1_IN4 |
| 16 | PC10 | 保留 | 16 | PA4 | T5 / ADC2_IN17 |
| 17 | PC11 | 保留 | 17 | PA1 | T2 / ADC12_IN2 |
| 18 | PB3 | 保留 | 18 | PA2 | T3 / ADC1_IN3 |
| 19 | PB4 | 保留；不得误认为 VBUS 检测 | 19 | NRST | 保留/复位 |
| 20 | PB5 | 功率许可 PB5→DRV_FAULT_N 串阻支路 | 20 | PA0 | T1 / ADC12_IN1 |
| 21 | PB6 | P-MOS_CTRL；确认 SB3、SB6 实物开路 | 21 | PC14 | 保留 |
| 22 | PB7 | DUMP_PWM / TIM4_CH2 | 22 | PC15 | 保留 |
| 23 | PB8/BOOT0 | 不用，保持板载 BOOT 配置 | 23 | VBAT | 按板载连接 |
| 24 | PB9 | 保留 | 24 | PC13 | 保留 |

核心板 P1-13/14 上 PA11/PA12 与 Type-C USB D−/D+ 硬连接，选择 CAN 时不得同时接 USB 数据线。VBUS 检测为 PB2 经 SB4 与板载 100 kΩ/10 kΩ/10 nF 网络，不是 PB4。PB6 经 SB3 接 PA9、经 SB6 接 CC1；只有逐板确认两桥开路后才使用 PB6，不能假定焊桥默认状态。本模块不改核心板。SWD 使用板上 P3 四针：1=3.3 V 目标参考、2=PA13/SWDIO、3=PA14/SWCLK、4=GND。调试器 3.3 V 只能作目标参考，禁止与核心板并联主动供电。

## 电流 CSA 双阈值候选路由

DRV8161 SO 同时短分支到 PB1 与 PA7：PB1 是 ADC1_IN12/COMP1_INP，PA7 是 COMP2_INP（该脚 ADC2_IN4 能力不用于本方案）。PA3 不承担比较器，保留为 T4/ADC1_IN4。COMP1 反相端候选用内部 DAC3_CH1 产生上阈值，COMP2 反相端候选用内部 DAC3_CH2 产生下阈值；G431 的 DAC3 两通道是内部路由，不要求占用外部 DAC 引脚。不要把 PB14 当作 COMP 输入。

候选方向为：COMP1 在 SO 高于上阈值时断言；COMP2 在 SO 低于下阈值时断言（用比较器输出极性或 TIM1 Break 输入极性实现，最终寄存器配置须确认）。将两 comparator 输出使能为 TIM1 Break 源，令任一越界都锁住安全 PWM 状态。必须核实 STM32G431 实际 RM0440 位定义、内部源合并与极性、MOE/OSSI/OSSR、Break 锁存/清除、DAC/COMP 上电顺序、采样/空白设置及实测传播延迟。此为可路由的快关断候选，尚未证明每周期动作、延迟上限或失效安全。

在 CSAREF=3.3 V、GCSA=20、Rshunt=0.5 mΩ 的理想关系下，SO≈1.65 V+0.010 V/A×I_SC（放电为正）；±80 A 对应约 0.85 V/2.45 V。±140 A 理想端点约 0.25–3.05 V，刚到候选 CSA 摆幅边界，不能称作有裕量的可靠范围。80 A 是过流保护/器件筛选目标候选，阈值还须计 CSA 偏移/增益、参考与 DAC 误差、分流器公差/温漂、布线噪声、比较器配置、关断时延和峰值上冲。

## 八节累积抽头 ADC 与模拟 TEST 回读

PA0–PA6 分别采 T1–T7；PB14 单独分压采 SC+ 顶端，并复用同一 ADC 读数作为累积抽头 T8。PB15 保留为模拟 TEST_FB，经 10k/10k 分压读取共用 TEST 电压；PB10 仍为 TEST_CTRL 数字输出。取消单独 T8 的 47 kΩ TEST 支路。每路 ADC RC 电容/截止频率仍待定。

| 累积 tap | 核心板 ADC | 排针 | ADC 通道 | 备注 |
|---:|---|---|---|---|
| T1 | PA0 | P2-20 | ADC12_IN1 | |
| T2 | PA1 | P2-17 | ADC12_IN2 | |
| T3 | PA2 | P2-18 | ADC1_IN3 | |
| T4 | PA3 | P2-15 | ADC1_IN4 | 与旧候选 PA3/COMP2 路由解耦 |
| T5 | PA4 | P2-16 | ADC2_IN17 | |
| T6 | PA5 | P2-13 | ADC2_IN13 | |
| T7 | PA6 | P2-14 | ADC2_IN3 | |
| T8 / SC_SUM 顶端 | PB14 | P1-7 | ADC1_IN5 | 同一通道，不构成独立 T8 与总压交叉校验 |
| TEST_FB | PB15 | P1-8 | ADC2_IN15 | TEST 经 10k/10k 分压回读 |

T1–T7 每路使用候选网络 `Tj—68 kΩ—ADCj`、`ADCj—10 kΩ—GND_SYS`、`ADCj—47 kΩ—TEST`；PB15 的 TEST_FB 候选为 TEST—10 kΩ—ADC、ADC—10 kΩ—GND_SYS。对于 T1–T7，名义式为 `K=1+68/10+68/47=9.2468`、`c=68/47=1.4468`、`Vtap[j]=K×Vadc[j]−c×Vtest`。T8/SC_SUM 使用独立简单分压 `SC+—68 kΩ—PB14_ADC`、`PB14_ADC—10 kΩ—GND_SYS`，所以 `Vtap8=7.8×Vadc(PB14)`，`c8=0`。阻值仍是候选，比例误差、RC 建立时间、ADC 输入漏电/采样时间和高压输入故障额定未闭合。

所有 tap ADC 都相对 GND_SYS 读取。令 `V_SCminus` 为 SC− shunt 侧 Kelvin 点相对 GND_SYS 的电压，则 `Vcell1=Vtap1−V_SCminus`；`Vcell2…Vcell7=Vtap[j]−Vtap[j−1]`；`Vcell8=Vtap8−Vtap7`；`V_SC_total=Vtap8−V_SCminus`。后两者需要 `Vtap7` 与 SC−参考/补偿；不得默认 SC− 与系统地等电位。SC−偏移的采集或基于电流/分流器的补偿方法及其误差预算仍待闭合。

PB14 是 T8 与 SC_SUM 的同一个 ADC 节点/读数；`ΣVcell[j]` 与端点差值是望远镜恒等式，不是独立交叉校验，也不能证明 T8 通道自身正常。该路由取消了独立冗余 T8 ADC 观测；部分开短路可做范围、线束连续性和相邻差分诊断，TLV6700 提供独立硬件 SC 总压过压比较路径，但不能宣称覆盖所有单通道故障或完整安全诊断。

诊断时 PB10 正常推挽 LOW；单独诊断帧短时拉 HIGH，检查 PB10 IDR，并通过 PB15 的模拟 TEST_FB 观察 TEST 高低值，等待 RC/ADC 稳定后检查 tap 响应，再恢复 LOW、核验 IDR 和模拟回读。回读不能代替通道开短路诊断或电阻漂移覆盖。T1 误差预算必须使用实测/校准后的 TEST_FB，而不能把 PB10 的逻辑态当作精确电压。

工作窗 2.2–2.45 V/节仍是目标规划窗，不是获准硬安全窗；相邻抽头误差假设 ±0.15 V 已大于半窗 0.125 V，若两端都作硬界，保证余量 `0.25−2×0.15=−0.05 V`。先闭合测量误差、安全边界与 SC−参考误差，再批准门限/储能。

## BW6101 每节引脚与被动外围

按 BITWELL 厂商 BW6101 规格书 SOT23-5L 顶视引脚表和典型应用图：每颗跨接一只超容单体。SEL 接该节 GND 选 2.45 V 档；IOUT 通过该节泄放电阻接该节 GND；VDD/GND 分别接该节正/负极。LED 报警不用时 NC。不能把任何引脚或均衡电阻跨接到整串系统地。

| 脚 | 名称 | 本方案候选连接 |
|---:|---|---|
| 1 | VDD | 本节超容正端 |
| 2 | GND | 本节超容负端 |
| 3 | SEL | 接本节 GND，选择 2.45 V 档 |
| 4 | LED | NC；不放 LED 与限流电阻 |
| 5 | IOUT | 接本节 120 Ω 泄放电阻，再到本节 GND |

120 Ω/0.25 W 是小电流均衡算例：在 2.7 V 时约 22.5 mA、61 mW；不依赖器件内部大电流。厂商 PDF 写 700 mA@VDD=2.65 V，厂商网页写无扩流管 200 mA，口径冲突；本方案限流由外部电阻决定。PDF 仅给 25°C 的 2.45 V 档 2.42/2.45/2.48 V（min/typ/max）和报警 2.51/2.55/2.59 V；阈值不是主充电限压，单体的软件保护仍需按误差边界设定。无需另加跨单体的泄放路径；原厂典型应用由本节电容提供 VDD-GND 供电。

## UCC27511A 全引脚与 BUS dump 驱动外围

按 TI UCC27511A DBV/SOT-23-6 引脚表；具体订货型号和封装仍待 BOM 绑定。此驱动只执行 G431 软件 PWM 控制的低侧 BUS dump，不具备独立 BUS 过压触发。

| 脚 | 名称 | 本方案候选连接 |
|---:|---|---|
| 1 | VDD | `12V_DRV`，近端 100 nF 与 1 µF 至脚 4 |
| 2 | OUTH | 经独立 10 Ω 候选栅电阻至 `DUMP_FET_GATE` |
| 3 | OUTL | 经独立 10 Ω 候选栅电阻至同一 `DUMP_FET_GATE` |
| 4 | GND | `GND_SYS`，门极回流短接 |
| 5 | IN− | 接 `GND_SYS`，固定非反相驱动 |
| 6 | IN+ | PB7/TIM4_CH2 `DUMP_PWM` 经 100 Ω 候选串阻；脚上 10 kΩ 下拉 |

`DUMP_FET_GATE` 再以 10 kΩ 候选电阻下拉到 MOSFET source/GND_SYS。输入/栅极电阻均是待波形、脉冲功率与复位态核定的候选，不是冻结值。原厂资料：[TI UCC27511A 数据手册](https://www.ti.com/lit/ds/symlink/ucc27511a.pdf)。

## TCAN332 全引脚与 CAN 接口外围

按 TI TCAN332 数据手册 Rev. F，选用封装型号仍待 BOM 绑定，但 D 与 DCN 图示均为 8 引脚顶视图。PA12/FDCAN1_TX 接 TXD，RXD 回 PA11/FDCAN1_RX。

| 脚 | 名称 | 本方案连接/状态 |
|---:|---|---|
| 1 | TXD | 接 MCU PA12/FDCAN1_TX |
| 2 | GND | GND_SYS，与核心板地共参考 |
| 3 | VCC | 核心板板载 3.3 V；旁路电容回脚 2 |
| 4 | RXD | 接 MCU PA11/FDCAN1_RX |
| 5 | NC | 保持不接 |
| 6 | CANL | CAN 连接器 CANL |
| 7 | CANH | CAN 连接器 CANH |
| 8 | NC | 保持不接 |

原厂要求每个 VCC 近端放 100 nF 陶瓷去耦；更大的本地 bulk 由核心板 3.3 V 负载/布线核定。TCAN332 没有 standby/shutdown 引脚。总线两物理端各用 120 Ω；只有本模块处于总线末端时才装本地端接，非末端保持不接。分裂端接是可选 EMC 方案而非必装；TVS/额外保护按整车 ESD/EFT/surge 要求核定后加在连接器附近。不要在本次准备稿中冻结端接位置、TVS 型号或总线隔离器。

## 外部连接器候选与未冻结项

| 接口 | 建议表示的信号 | 备注 |
|---|---|---|
| J_CORE_P1 / J_CORE_P2 | 上表逐脚映射 | 2×12、2.54 mm，接口封装/性别以实物量测确认；FOC 旧工程的 M20-7831242/项目双排母座只作格式参考，不复用或修改旧工程 |
| J_SC_POWER | SC+、SC−/shunt Kelvin 侧 | 高电流连接器、线径、爬电和端子型号待核 |
| J_SC_TAPS | T0、T1…T8 共 9 根 | 只做单体电压采样；与 SC_POWER 的连接/插拔顺序须有装配约束，具体连接器待定 |
| J_BUS_POWER | BUS_EXT+、GND_SYS | 外部电池 BUS；电压/电流等级、保险与接插件待核 |
| J_DUMP | BUS_EXT+、DUMP_SW | 接外置 dump 电阻；电阻/连接器仍需满足候选脉冲功率与能量要求 |
| J_CAN | GND_SYS、CANH、CANL | CAN/USB 数据互斥；端接/TVS按节点位置和整车环境核定 |
| J_NTC_SC / J_NTC_MAIN / J_NTC_DUMP | 各自 NTC_ADC、GND_SYS | 保持三处独立测温；NTC 曲线、偏置电阻、滤波/线束值未定 |
| +5V_CORE / +12V_DRV | 两个独立辅助电源网名 | +5V 优先核定现有车载供电复用；+12V 来自成品驱动电源模块候选；不得把 10/12 V 接核心板 VCC 或 3.3 V |

## 准备阶段未闭合项

- 最终 COMP1/COMP2 polarity、DAC 输出范围/量化误差、TIM1 Break 源合并与复位/锁存恢复，及含驱动/MOS 的实测过流关断时延。
- TEST_FB 与共用 TEST 网络加载/动态误差、T1–T7 精度与抽头开短路/ADC 掉电反灌覆盖；PB14 同时作为 T8 和 SC_SUM 的单通道失效识别边界。
- 八路分压的比例精度、RC 电容/截止频率、ADC 采样时间和校准流程；三路 NTC 具体阻容和热安装。
- 每种外部连接器的具体料号、电流/耐压、引脚方向、防反插与装配约束。
- TCAN332 本地端接位置、TVS/滤波是否需要、CAN 地/屏蔽约定；WeAct 3.3 V 完整负载/热预算。
- 单页完整器件数、图幅与 ERC；真实板图与物理布线、功率/热验证。
- 目标 0.8–1 kW、约 1 s 仍为未验证目标候选；本方案没有独立硬件 BUS_OV dump。

## 可追溯来源

- [ST STM32G431CB 数据手册 DS12589](https://www.st.com/resource/en/datasheet/stm32g431cb.pdf)：UFQFPN48 引脚、PA/PB/PC 模拟通道、COMP 输入和 GPIO 复用。
- [ST RM0440](https://www.st.com/resource/en/reference_manual/dm00355726.pdf) 与 [AN4232](https://www.st.com/resource/en/application_note/an4232-getting-started-with-analog-comparators-for-stm32f3-series-and-stm32g4-series-devices-stmicroelectronics.pdf)：DAC3/COMP/TIM 互连核查；AN4232 指出 STM32G4 不支持单 comparator window mode。
- [BITWELL BW6101 厂商产品页](https://bitwell.cn/productview.aspx?CateID=607&ID=269) 与[原厂规格书 PDF](https://bitwell.cn/Upload/%E4%BF%9D%E6%8A%A4%E8%8A%AF%E7%89%87/BW6101_C%E4%BF%9D%E6%8A%A4IC-09311941072.pdf)：引脚、典型应用、阈值和静态/泄放数据。
- [TI TCAN332 数据手册 Rev. F](https://www.ti.com/lit/ds/symlink/tcan332.pdf)：引脚、100 nF 去耦、总线端接和保护布置。
- 本地 WeAct 原理图：[WeAct_normal_QFN48_V1.0.pdf](../../../../FOC驱动审阅/硬件/数据手册/WeAct_normal_QFN48_V1.0.pdf)。旧 FOC 连接器格式只读参考：[系统辅助板/核验.json](../../../../FOC驱动审阅/硬件/系统辅助板/核验.json) 中 J_CORE_P1/P2 的 M20-7831242 与项目封装记录；这不是本模块的已批准实物绑定。
- 参数背景：[DRV8161/UCC/TLV 参数核对侧车文件](DRV8161_UCC_TLV_参数核对_20261010.md)、[计算与保护边界](../计算与保护边界.md)和[最简元器件方案](../最简元器件方案.md)。
