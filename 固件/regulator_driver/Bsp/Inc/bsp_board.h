#ifndef __BSP_BOARD_H__
#define __BSP_BOARD_H__

#include "type.h"
#include "stm32g4xx_hal.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 稳压模块（sc-regulator V1.6-BUSFIX）—— 板级引脚总表
 *
 * 这是本工程**唯一**定义引脚和 ADC 通道的地方。各 bsp_*.c 只引用、不重复定义。
 * 改板、换脚、改通道，只改这一个文件，不用去翻六个 .c。
 *
 * ── 依据 ────────────────────────────────────────────────────────────────────
 *   原理图：电力/稳压模块/eda工程文件/sc-regulator/sch/01 稳压模块全部电路.esch2
 *           （本地副本 D:/PCB/稳压模块/…；导出图 稳压模块_V1.6-BUSFIX_逻辑原理图.pdf）
 *   引脚表：电力/稳压模块/原理图准备/接口与采样核对.md
 *           「WeAct Normal QFN48 核心板排针映射」一节，2026-10-22 接口映射纠正版
 *   器件值：电力/稳压模块/原理图准备/BOM.csv（R_* 逐只核对，见下方各换算宏）
 *   MCU 复用/ADC 通道：DS12589 Rev6 Table 12，UFQFPN48 列
 *
 *   排针 J_CORE_P1 = 核心板 P1（WeAct Normal QFN48），J_CORE_P2 = 核心板 P2，
 *   各 2×12、2.54 mm。下文 P1.n / P2.n 就是这两排的针号。
 *
 * ── 这块板是什么 ────────────────────────────────────────────────────────────
 *   一颗**双向半桥 DC/DC**，在 9S4P 电池直连的 BUS（22.5–37.8 V）与 8S1P 超容之间
 *   搬能量；外加一路独立软件控制的泄放（dump）电阻、8 节超容累积抽头采样、
 *   3 路 NTC、SC 总压硬件比较（TLV6700）与 8 颗 BW6101 被动均衡。
 *
 *   板上的 MCU 就是核心板上的 STM32G431CBU6 —— **本板自己没有主控**，
 *   只提供 J_CORE_P1/P2 两排母座插核心板。同一颗核心板换插到 FOC 板上，
 *   就是另一套固件（本仓库的 foc_driver 工程）。
 *
 * ── 功率级芯片（全部由电阻配置，没有 SPI）───────────────────────────────────
 *   U1  DRV8161DGSR   半桥栅极驱动 + 电流放大器（CSA）
 *       · DT/MODE 3.3 kΩ 到 GND → 2-pin PWM 模式，**芯片内置死区典型 370 ns**
 *       · CSAGAIN  30 kΩ 到 GND → 20 V/V
 *       · VDSLVL   27 kΩ 到 GND → LEVEL4（VDS 阈值 typ 0.8 V DC）
 *       · CSAREF   板载 3.3 V → CSA 零点名义 1.65 V
 *   Q_MAIN_H/L  IRLB4030PBF  ×2   半桥上下管
 *   Q_BUS_P     IXTH76P10T        BUS 串联 P-MOS，由 PB6 独立控制
 *   Q_DUMP      IRLB4030PBF       泄放低边管，UCC27511A 驱动
 *   U_SC_OV     TLV6700DDCR       超容总压窗口比较（独立硬件路径）
 *
 * ── 引脚总表 ────────────────────────────────────────────────────────────────
 *   信号           排针      MCU    复用/通道             方向    说明
 *   ----------------------------------------------------------------------------------
 *   PWM_H         P1.10     PA8    TIM1_CH1    (AF6)     输出    半桥上管（→ INH）
 *   PWM_L         P1.6      PB13   TIM1_CH1N   (AF6)     输出    半桥下管（→ INL）
 *   DUMP_PWM      P1.22     PB7    TIM4_CH2    (AF2)     输出    泄放 PWM（→ UCC27511A IN+）
 *   PMOS_CTRL     P1.21     PB6    GPIO                  输出    BUS 串联 P-MOS 栅极
 *   PERMIT        P1.20     PB5    GPIO                  输出    功率许可（串阻接 DRV_FAULT_N）
 *   TEST          P2.5      PB10   GPIO                  输出    诊断用 TEST 网络驱动（平时 LOW）
 *   FDCAN1_RX     P1.13     PA11   FDCAN1_RX   (AF9)     输入    ⚠️ 与核心板 USB D− 硬连
 *   FDCAN1_TX     P1.14     PA12   FDCAN1_TX   (AF9)     输出    ⚠️ 与核心板 USB D+ 硬连
 *   AD_T1         P2.20     PA0    ADC1_IN1             模拟    超容第 1 节抽头
 *   AD_T2         P2.17     PA1    ADC1_IN2             模拟    第 2 节抽头
 *   AD_T3         P2.18     PA2    ADC1_IN3             模拟    第 3 节抽头
 *   AD_T4         P2.15     PA3    ADC1_IN4             模拟    第 4 节抽头
 *   AD_T5         P2.16     PA4    ADC2_IN17            模拟    第 5 节抽头
 *   AD_T6         P2.13     PA5    ADC2_IN13            模拟    第 6 节抽头
 *   AD_T7         P2.14     PA6    ADC2_IN3             模拟    第 7 节抽头
 *   SC_SUM/AD_T8  P1.7      PB14   ADC1_IN5             模拟    SC+ 顶端（同一读数兼作 T8）
 *   NTC_DUMP      P1.5      PB12   ADC1_IN11            模拟    泄放电阻温度
 *   NTC_MAIN      P2.6      PB11   ADC2_IN14            模拟    主功率级温度
 *   NTC_SC        P2.12     PC4    ADC2_IN5             模拟    超容温度
 *   DRV_SO        P2.10     PB1    ADC1_IN12            模拟    DRV8161 电流放大器输出（也是 COMP1_INP）
 *   BUS_ADC       P2.9      PB0    ADC1_IN15            模拟    母线电压（150k/10k 分压）
 *   TEST_FB       P1.8      PB15   ADC2_IN15            模拟    TEST 经 10k/10k 分压回读
 *   (DRV_SO_ALT)  P2.11     PA7      —                  模拟    SO 第二分支，只接 COMP2_INP，不进 ADC
 *   VREF_CORE     P2.8      VREF+    —                  电源    ADC 满量程 + CSA 基准
 *
 * ── 五个必须记住的复用陷阱（改动前先读）──────────────────────────────────────
 *
 *  1) PA11/PA12 = FDCAN1_RX/TX + USB D−/D+
 *     核心板把这两个脚同时引到 Type-C。**接 CAN 时不得再插 USB 数据线**，
 *     否则 USB 收发器会成为 CAN 总线的额外负载。
 *
 *  2) PB6 = P-MOS_CTRL，但核心板上 PB6 经 SB3 接 PA9、经 SB6 接 CC1
 *     **必须先用万用表确认 SB3/SB6 实物开路**，否则 PB6 会同时驱动 PA9 和 Type-C 的 CC1。
 *     这不是软件能兜住的，只能改板或挑开焊桥。
 *
 *  3) PB12 = NTC_DUMP(ADC1_IN11) + TIM1_BKIN
 *     本板 PB12 只做模拟输入，**TIM1 的 BreakState 必须保持 DISABLE**，
 *     否则 NTC 分压上的任何低电平都会被当成刹车，半桥反复掉输出。
 *     真正的过流快关断应走 COMP1/COMP2 → TIM1 Break（片内路由，不占引脚），尚未实现，见 bsp_bridge.c。
 *
 *  4) PA6 = AD_T7(ADC2_IN3) + TIM1_BKIN
 *     同上，PA6 在本板是模拟脚，**绝不能**把 TIM1 的 BKIN 源配到 PA6。
 *
 *  5) P2.7 = PB2 是核心板自己的 VBUS 检测（经 SB4 + 板载 100k/10k/10nF）
 *     不要复用、不要改板。同理 P1.19 = PB4 常被误当成 VBUS 检测，它不是。
 *
 * ── 封装硬限制（UFQFPN48，改板时别把线画到这些脚上）──────────────────────────
 *   PC0 / PC1 / PC2 / PC3 / PC5 / PC12 在 UFQFPN48 上**不存在**。
 *   原理图核对时若发现这些脚名，说明看错了封装列（DS12589 Table 12 有 5 种封装，
 *   只有 UFQFPN48 那一列才算数）。
 *
 * ── 没被本工程使用的保留脚 ──────────────────────────────────────────────────
 *   PC6 / PA9 / PA10 / PA15 / PC10 / PC11 / PB3 / PB4 / PB8-BOOT0 / PB9 / PC13 / PC14 / PC15
 *   全部保持复位默认（模拟输入或浮空输入），不配置、不驱动。
 *   其中 PB8-BOOT0 尤其不能动：它是 BOOT0，拉高会让芯片从 System Memory 启动。
 ********************************************************************************************/

/************************************双向半桥（TIM1）******************************************/
/* 为什么只有两路：DRV8161 工作在 2-pin PWM 模式（DT/MODE 脚 3.3 kΩ 上电锁存），
 * 上下管各吃一根输入线，互补由 MCU 的 CH1 / CH1N 直接给出。
 * 手册真值表：INH=1,INL=0 → GH 高、GL 低；INH=0,INL=1 → GH 低、GL 高；
 *            两者同为高或同为低 → 两栅极都低、SH 进 Hi-Z（这就是它自带的防直通）。 */
#define BSP_BOARD_BRIDGE_H_PORT                                         (GPIOA)
#define BSP_BOARD_BRIDGE_H_PIN                                          (GPIO_PIN_8)    /* P1.10 */
#define BSP_BOARD_BRIDGE_L_PORT                                         (GPIOB)
#define BSP_BOARD_BRIDGE_L_PIN                                          (GPIO_PIN_13)   /* P1.6  */
#define BSP_BOARD_BRIDGE_AF                                             (GPIO_AF6_TIM1)

/* ---- 时基 ---- */
/* ⚠️ 这个频率**尚未定型**。原理图准备文档明确写着「PWM 频率、采样同步、限流策略和
 *    交越停机时序待确定」，所以下面这个 20 kHz 只是**首轮上电用的保守值**，不是结论。
 *
 * 为什么先给 20 kHz：开关损耗最低，第一次通电最容易活下来。
 *
 * 但它带来的电感纹波很大，必须知道这个数：
 *     ΔI_pp = V_bus × D × (1 − D) / (L × f)
 *   本板 L_MAIN_A + L_MAIN_B 串联 = 20 µH，母线取额定 32.4 V：
 *     f = 20  kHz, D = 0.5  → ΔI_pp ≈ 20.3 A
 *     f = 50  kHz, D = 0.5  → ΔI_pp ≈  8.1 A
 *     f = 100 kHz, D = 0.5  → ΔI_pp ≈  4.1 A
 *   20 A 峰峰值对 0.5 mΩ 分流器 + ±80 A 的 CSA 窗口来说测得出来，但对电感和母线
 *   电容的纹波电流应力不小。定型时要和磁件温升、EMI 一起算。
 *
 * 中心对齐：f_pwm = f_tim / (2 × ARR)。170 MHz / (2 × 4250) = 20 kHz。
 * 选中心对齐是为了让上下管开通点关于中点对称，电流纹波更小，
 * 也方便将来在计数器顶点做 ADC 同步采样（现在没开，见 bsp_bridge.c）。 */
#define BSP_BOARD_BRIDGE_PWM_ARR                                        (4250U)
#define BSP_BOARD_BRIDGE_PWM_FREQ_HZ                                    (20000U)

/* 死区（ns）。
 *
 * ⚠️ 这里刻意设成 0，不是漏了。
 *   DRV8161 在 2-pin 模式下自己插死区，典型 370 ns（数据手册范围 316–422 ns），
 *   原理图准备文档的原话是「首轮 MCU 不再叠加大 DTG，实测后再定」。
 *   两边都加死区不会更安全，只会把有效占空比压缩掉 2×370 ns：
 *   20 kHz 下一个周期 50 µs，370 ns 只占 0.74%；但对占空比接近 0 或接近 1 的
 *   工作点，额外死区会让最小/最大占空比够不到，直接损失调节范围。
 *
 *   若实测发现关断尖峰压不住，再把这里改到 50–100 ns 兜底；
 *   改完必须用示波器看 SHx 上升沿与 GHx 下降沿是否还重叠。
 *
 * TIM1 时钟 170 MHz → tDTS = 5.882 ns，DTG 寄存器 1 个步进约 5.9 ns。 */
#define BSP_BOARD_BRIDGE_DEADTIME_NS                                    (0U)

/************************************泄放（TIM4_CH2）******************************************/
/* 泄放电阻不在板上（外置，接 J_DUMP），由 Q_DUMP 低边开关 + UCC27511A 驱动。
 * UCC27511A 的 IN− 固定接 GND（非反相），IN+ 由 PB7 经 100 Ω 串阻进来、脚上 10 kΩ 下拉
 * → MCU 复位期间栅极天然关断，泄放不动作。 */
#define BSP_BOARD_DUMP_PWM_PORT                                         (GPIOB)
#define BSP_BOARD_DUMP_PWM_PIN                                          (GPIO_PIN_7)    /* P1.22 */
#define BSP_BOARD_DUMP_PWM_AF                                           (GPIO_AF2_TIM4)

/* 泄放 PWM 频率。
 *
 * 这个频率决定的是**泄放功率的调节粒度**，不是功率级的开关损耗 ——
 * Q_DUMP 串的是外置大功率电阻，导通期间电流基本由 R 决定，纹波无关紧要。
 * 20 kHz 只是顺手和半桥同频，避免两个开关动作在采样上互相拍频。
 *
 * TIM4 在 APB1（本工程 170 MHz），边沿对齐：ARR = 170 MHz / 20 kHz − 1 = 8499。
 *
 * 注意占空比 0 = 完全不泄放、100% = 长期导通。长期导通会烧电阻，
 * 上层必须自己限幅和限时 —— BSP 层不做这个判断。 */
#define BSP_BOARD_DUMP_PWM_ARR                                          (8499U)
#define BSP_BOARD_DUMP_PWM_FREQ_HZ                                      (20000U)

/************************************功率路径 / 杂项 GPIO**************************************/
/* BUS 串联 P-MOS 栅极（Q_BUS_P，IXTH76P10T，P 沟道）。
 * ⚠️ 只控制栅极电平，**不能**用它和 PWM 关断同时硬切电感回路 ——
 *    电感电流突变会在 Q_BUS_P 上打出远超额定的 Vds。要断流先停 PWM，再等电流衰减。
 * ⚠️ 用之前必须确认核心板 SB3（PB6↔PA9）、SB6（PB6↔CC1）是开路的。 */
#define BSP_BOARD_PMOS_CTRL_PORT                                        (GPIOB)
#define BSP_BOARD_PMOS_CTRL_PIN                                         (GPIO_PIN_6)    /* P1.21 */

/* 功率许可：PB5 经串阻接到 DRV_FAULT_N 网络，用作「允许功率级工作」的软件闸门。
 * 串阻值与有效极性以原理图为准，本层只负责给出高低电平。
 *
 * ⚠️ 这条线是**单向**的：MCU 只能拉，读不回来。DRV8161 真报故障时它会开漏拉低
 *    （栅极确实被关断，功能安全成立），但软件层**看不到**是过流、欠压还是过温——
 *    本板没有独立的 nFAULT 回读脚，TIM1 Break 又刻意禁用（见 bsp_bridge.h 第 4 条）。
 *    "驱动故障可观测"这一项只能靠 ADC 间接判断；要彻底解决得改板：PERMIT 改开漏
 *    并加一根回读脚，或在 DRV_FAULT_N 上分出一路输入。 */
#define BSP_BOARD_PERMIT_PORT                                           (GPIOB)
#define BSP_BOARD_PERMIT_PIN                                            (GPIO_PIN_5)    /* P1.20 */

/* 诊断用 TEST 网络驱动。正常运行时必须推挽 LOW：
 * T1–T7 每路都有一只 47 kΩ 从抽头侧接到 TEST，TEST 状态会直接改变这些分压比。
 * 只有做通道诊断时才短时拉高，并通过 TEST_FB 的模拟回读确认。 */
#define BSP_BOARD_TEST_PORT                                             (GPIOB)
#define BSP_BOARD_TEST_PIN                                              (GPIO_PIN_10)   /* P2.5  */

/************************************ADC*******************************************************/
/* ⚠️ 通道顺序 = 转换顺序 = 结果寄存器下标，三者必须一致。
 *    改这里必须同步改 bsp_adc.c 里的 Rank 赋值、缓冲区长度和 bsp_adc.h 里的下标宏。 */

/* ---- ADC1 · 规则组（8 通道，连续转换 + DMA 循环搬运）----
 * 为什么超容抽头分在两个 ADC 上：本板把 T1–T4 布到 PA0–PA3（ADC1），
 * T5–T7 布到 PA4–PA6（ADC2），这是原理图定的，软件改不了。 */
#define BSP_BOARD_ADC_T1_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T1_PIN                                            (GPIO_PIN_0)    /* P2.20 */
#define BSP_BOARD_ADC_T1_CHANNEL                                        (ADC_CHANNEL_1)
#define BSP_BOARD_ADC_T2_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T2_PIN                                            (GPIO_PIN_1)    /* P2.17 */
#define BSP_BOARD_ADC_T2_CHANNEL                                        (ADC_CHANNEL_2)
#define BSP_BOARD_ADC_T3_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T3_PIN                                            (GPIO_PIN_2)    /* P2.18 */
#define BSP_BOARD_ADC_T3_CHANNEL                                        (ADC_CHANNEL_3)
#define BSP_BOARD_ADC_T4_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T4_PIN                                            (GPIO_PIN_3)    /* P2.15 */
#define BSP_BOARD_ADC_T4_CHANNEL                                        (ADC_CHANNEL_4)
#define BSP_BOARD_ADC_T8_PORT                                           (GPIOB)
#define BSP_BOARD_ADC_T8_PIN                                            (GPIO_PIN_14)   /* P1.7  */
#define BSP_BOARD_ADC_T8_CHANNEL                                        (ADC_CHANNEL_5)
#define BSP_BOARD_ADC_NTC_DUMP_PORT                                     (GPIOB)
#define BSP_BOARD_ADC_NTC_DUMP_PIN                                      (GPIO_PIN_12)   /* P1.5  */
#define BSP_BOARD_ADC_NTC_DUMP_CHANNEL                                  (ADC_CHANNEL_11)
#define BSP_BOARD_ADC_DRV_SO_PORT                                       (GPIOB)
#define BSP_BOARD_ADC_DRV_SO_PIN                                        (GPIO_PIN_1)    /* P2.10 */
#define BSP_BOARD_ADC_DRV_SO_CHANNEL                                    (ADC_CHANNEL_12)
#define BSP_BOARD_ADC_BUS_PORT                                          (GPIOB)
#define BSP_BOARD_ADC_BUS_PIN                                           (GPIO_PIN_0)    /* P2.9  */
#define BSP_BOARD_ADC_BUS_CHANNEL                                       (ADC_CHANNEL_15)
#define BSP_BOARD_ADC1_MASK                                             \
    (BSP_BOARD_ADC_T1_PIN | BSP_BOARD_ADC_T2_PIN | BSP_BOARD_ADC_T3_PIN | \
     BSP_BOARD_ADC_T4_PIN | BSP_BOARD_ADC_T8_PIN | BSP_BOARD_ADC_NTC_DUMP_PIN | \
     BSP_BOARD_ADC_DRV_SO_PIN | BSP_BOARD_ADC_BUS_PIN)

/* ---- ADC2 · 规则组（6 通道）---- */
#define BSP_BOARD_ADC_T5_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T5_PIN                                            (GPIO_PIN_4)    /* P2.16 */
#define BSP_BOARD_ADC_T5_CHANNEL                                        (ADC_CHANNEL_17)
#define BSP_BOARD_ADC_T6_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T6_PIN                                            (GPIO_PIN_5)    /* P2.13 */
#define BSP_BOARD_ADC_T6_CHANNEL                                        (ADC_CHANNEL_13)
#define BSP_BOARD_ADC_T7_PORT                                           (GPIOA)
#define BSP_BOARD_ADC_T7_PIN                                            (GPIO_PIN_6)    /* P2.14 */
#define BSP_BOARD_ADC_T7_CHANNEL                                        (ADC_CHANNEL_3)
#define BSP_BOARD_ADC_NTC_MAIN_PORT                                     (GPIOB)
#define BSP_BOARD_ADC_NTC_MAIN_PIN                                      (GPIO_PIN_11)   /* P2.6  */
#define BSP_BOARD_ADC_NTC_MAIN_CHANNEL                                  (ADC_CHANNEL_14)
#define BSP_BOARD_ADC_NTC_SC_PORT                                       (GPIOC)
#define BSP_BOARD_ADC_NTC_SC_PIN                                        (GPIO_PIN_4)    /* P2.12 */
#define BSP_BOARD_ADC_NTC_SC_CHANNEL                                    (ADC_CHANNEL_5)
#define BSP_BOARD_ADC_TEST_FB_PORT                                      (GPIOB)
#define BSP_BOARD_ADC_TEST_FB_PIN                                       (GPIO_PIN_15)   /* P1.8  */
#define BSP_BOARD_ADC_TEST_FB_CHANNEL                                   (ADC_CHANNEL_15)
#define BSP_BOARD_ADC2_MASK                                             \
    (BSP_BOARD_ADC_T5_PIN | BSP_BOARD_ADC_T6_PIN | BSP_BOARD_ADC_T7_PIN | \
     BSP_BOARD_ADC_NTC_MAIN_PIN | BSP_BOARD_ADC_NTC_SC_PIN | BSP_BOARD_ADC_TEST_FB_PIN)

/* 只接比较器、不进 ADC 的那一路：PA7 是 DRV_SO 的第二分支（SO 经 22 Ω 同时扇出到
 * PB1 和 PA7），只接 COMP2_INP。这里只把脚配成模拟输入，避免浮空脚在比较器输入端
 * 拾噪。它不参与任何 ADC 序列，所以没有通道宏。 */
#define BSP_BOARD_ADC_SO_ALT_PORT                                       (GPIOA)
#define BSP_BOARD_ADC_SO_ALT_PIN                                        (GPIO_PIN_7)    /* P2.11 */

/* ADC 满量程（mV）。VREF_CORE（P2.8）同时是 DRV8161 的 CSAREF 参考（板载 3.3 V）
 * 和 MCU 的 VREF+，两者同源，所以电流换算与电压换算共用同一个基准。
 * ⚠️ 上电第一件事：万用表量 P2.8 对 GND，确认 3.3 V 再谈任何标定。
 *    若实测不是 3300 mV，改这里，否则所有换算都按比例偏。 */
#define BSP_BOARD_ADC_VREF_MV                                           (3300U)

/************************************FDCAN1****************************************************/
/* TCAN332DR 没有 STB/SHDN 引脚，所以本板没有「收发器待机」这根线，
 * 也就没有 foc_driver 里那种 CAN_STB 控制。想静默总线只能停 FDCAN 外设本身。
 * ⚠️ PA11/PA12 与核心板 Type-C 的 D−/D+ 硬连，接 CAN 时不能同时插 USB 数据线。 */
#define BSP_BOARD_CAN_RX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_RX_PIN                                            (GPIO_PIN_11)   /* P1.13 */
#define BSP_BOARD_CAN_TX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_TX_PIN                                            (GPIO_PIN_12)   /* P1.14 */
#define BSP_BOARD_CAN_AF                                                (GPIO_AF9_FDCAN1)

#ifdef _cplusplus
}
#endif

#endif /* __BSP_BOARD_H__ */
