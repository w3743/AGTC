#ifndef __BSP_BOARD_H__
#define __BSP_BOARD_H__

#include "type.h"
#include "stm32g4xx_hal.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * FOC 驱动模块（三相无刷）—— 板级引脚总表
 *
 * 这是本工程**唯一**定义引脚和 ADC 通道的地方。各 bsp_*.c 只引用、不重复定义。
 * 改板、换脚、改通道，只改这一个文件，不用去翻七个 .c。
 *
 * ── 依据 ────────────────────────────────────────────────────────────────────
 *   原理图网表：FOC驱动模块/原理图准备/当前原理图连接.json
 *               169 器件 / 506 脚 / 102 网，2026-10-21 快照
 *   MCU 复用/ADC 通道：DS12589 Rev6 Table 12，UFQFPN48 列
 *   排针 J_MCU_L = 核心板 P1（WeAct Normal QFN48），J_MCU_R = 核心板 P2
 *
 * ── 这块板驱动什么 ──────────────────────────────────────────────────────────
 *   一颗**三相无刷电机**，经 DRV8353 三相逆变器驱动，带霍尔换相、低边电流采样、
 *   泄放电阻（dump）与硬件刹车。
 *
 *   无刷（FOC）在本工程 foc_driver 里，直流有刷在另一个工程 water_car_motor 里，
 *   两者模块名刻意区分：
 *     本工程          bsp_inverter  —— 三相逆变器（无刷），6 路互补 PWM
 *     water_car_motor bsp_hbridge    —— H 桥（直流），1 路 PWM + 1 路方向
 *
 * ── 引脚总表 ────────────────────────────────────────────────────────────────
 *   以下排针号已与 BSP引脚对照.md / 原理图网表逐条核对，勿凭印象改。
 *   信号           排针      MCU    复用/通道             方向    说明
 *   ----------------------------------------------------------------------------------
 *   PWM_UH        L.10      PA8    TIM1_CH1    (AF6)     输出    上桥 A
 *   PWM_VH        L.11      PA9    TIM1_CH2    (AF6)     输出    上桥 B
 *   PWM_WH        L.12      PA10   TIM1_CH3    (AF6)     输出    上桥 C
 *   PWM_UL        L.6       PB13   TIM1_CH1N   (AF6)     输出    下桥 A
 *   PWM_VL        L.7       PB14   TIM1_CH2N   (AF6)     输出    下桥 B
 *   PWM_WL        L.8       PB15   TIM1_CH3N   (AF6)     输出    下桥 C
 *   BREAK_N       L.5       PB12   TIM1_BKIN   (AF6)     输入    硬件刹车，低有效
 *   DRV_ENABLE    R.6       PB11   GPIO                  输出    DRV8353 ENABLE，高=放行
 *   CAN_STB       L.20      PB5    GPIO                  输出    收发器待机，高=待机
 *   DUMP_THERM_OK L.24      PB9    GPIO                  输入    泄放电阻热许可
 *   HALL_A        L.15      PA15   TIM2_CH1    (AF1)     输入    霍尔 A
 *   HALL_B        L.18      PB3    TIM2_CH2    (AF1)     输入    霍尔 B
 *   HALL_C        R.5       PB10   TIM2_CH3    (AF1)     输入    霍尔 C
 *   DRV_SPI_CS_N  R.16      PA4    GPIO                  输出    SPI1 软片选
 *   DRV_SPI_SCK   R.13      PA5    SPI1_SCK    (AF5)     输出    SPI 时钟
 *   DRV_SPI_MISO  R.14      PA6    SPI1_MISO   (AF5)     输入    ⚠️ 同时是 TIM1_BKIN
 *   DRV_SPI_MOSI  R.11      PA7    SPI1_MOSI   (AF5)     输出    SPI 数据
 *   SO_U_ADC      R.17      PA1    ADC1_IN2             模拟    相电流 U（注入组 rank1）
 *   SO_V_ADC      R.15      PA3    ADC1_IN4             模拟    相电流 V（注入组 rank2）
 *   SO_W_ADC      R.20      PA0    ADC1_IN1             模拟    相电流 W（注入组 rank3）
 *   TEMP_MOS_ADC  R.18      PA2    ADC1_IN3             模拟    MOS 温度
 *   TEMP_MOT_ADC  R.10      PB1    ADC1_IN12            模拟    电机温度
 *   SAFE_ADC      R.9       PB0    ADC1_IN15            模拟    安全信号
 *   BUS_ADC       R.12      PC4    ADC2_IN5             模拟    母线电压（只有 ADC2 有）
 *   FDCAN1_RX     L.13      PA11   FDCAN1_RX   (AF9)    输入    ⚠️ 同时是 TIM1_CH4 + TIM1_BKIN2
 *   FDCAN1_TX     L.14      PA12   FDCAN1_TX   (AF9)    输出
 *   VREF_CORE     R.8       VREF+  —                    电源    ADC 满量程 + CSA 基准
 *
 * ── 三处必须记住的复用陷阱（全部已在本工程的代码里规避，改动前先读）────────────
 *
 *  1) PB12 = TIM1_BKIN + ADC1_IN11
 *     本板把它当刹车输入，就**不能**再拿它做模拟通道。代价是稳压板的
 *     DUMP_PWM 也许想用 PB12 —— 那是另一块板的事，不在这里。
 *
 *  2) PA6 = SPI1_MISO + TIM1_BKIN
 *     本板的 PB12 已经接了 BKIN，PA6 只当 SPI 用。**绝不能**把 TIM1 的刹车源
 *     配到 PA6：那样 SPI 每传一帧、MISO 上出现低电平就会被当成故障，整机反复掉输出。
 *
 *  3) PA11 = FDCAN1_RX + TIM1_CH4 + TIM1_BKIN2
 *     本板用 PA11 做 CAN_RX。TIM1_CH4 只在**片内**当 ADC 触发源（TRGO=OC4REF），
 *     引脚上是 AF9 不会输出 AF6 的波形，这一点是安全的；
 *     但 Break2 **必须保持 DISABLE**（bsp_inverter 里写死），否则 CAN 总线上
 *     每一帧数据都可能触发硬件刹车。
 *
 * ── 封装硬限制（UFQFPN48，改板时别把线画到这些脚上）──────────────────────────
 *   PC0 / PC1 / PC2 / PC3 / PC5 / PC12 在 UFQFPN48 上**不存在**。
 *   原理图核对时若发现这些脚名，说明看错了封装列（DS12589 Table 12 有 5 种封装，
 *   只有 UFQFPN48 那一列才算数）。
 ********************************************************************************************/

/************************************三相逆变器（TIM1）*****************************************/
/* 上桥三路：同一个 GPIOA，可以一次 HAL_GPIO_Init 配完 */
#define BSP_BOARD_INV_UH_PORT                                           (GPIOA)
#define BSP_BOARD_INV_UH_PIN                                            (GPIO_PIN_8)    /* J_MCU_L.10 */
#define BSP_BOARD_INV_VH_PORT                                           (GPIOA)
#define BSP_BOARD_INV_VH_PIN                                            (GPIO_PIN_9)    /* J_MCU_L.11 */
#define BSP_BOARD_INV_WH_PORT                                           (GPIOA)
#define BSP_BOARD_INV_WH_PIN                                            (GPIO_PIN_10)   /* J_MCU_L.12 */
#define BSP_BOARD_INV_HIGH_MASK                                         \
    (BSP_BOARD_INV_UH_PIN | BSP_BOARD_INV_VH_PIN | BSP_BOARD_INV_WH_PIN)

/* 下桥三路：同一个 GPIOB */
#define BSP_BOARD_INV_UL_PORT                                           (GPIOB)
#define BSP_BOARD_INV_UL_PIN                                            (GPIO_PIN_13)   /* J_MCU_L.6  */
#define BSP_BOARD_INV_VL_PORT                                           (GPIOB)
#define BSP_BOARD_INV_VL_PIN                                            (GPIO_PIN_14)   /* J_MCU_L.7  */
#define BSP_BOARD_INV_WL_PORT                                           (GPIOB)
#define BSP_BOARD_INV_WL_PIN                                            (GPIO_PIN_15)   /* J_MCU_L.8  */
#define BSP_BOARD_INV_LOW_MASK                                          \
    (BSP_BOARD_INV_UL_PIN | BSP_BOARD_INV_VL_PIN | BSP_BOARD_INV_WL_PIN)

/* 六路 + 刹车输入共用同一个复用号 AF6，写成宏避免六处各写一遍 */
#define BSP_BOARD_INV_AF                                                (GPIO_AF6_TIM1)

/* 刹车输入 BREAK_N：与 DRV8353 的 nFAULT、U3 比较器输出线与，板上 2.2k 上拉，低有效。
 * 同一个脚也要能被 bsp_gpio_read_break_n() 当普通输入读回（IDR 在 AF 模式下仍反映真实电平）。 */
#define BSP_BOARD_INV_BKIN_PORT                                         (GPIOB)
#define BSP_BOARD_INV_BKIN_PIN                                          (GPIO_PIN_12)   /* J_MCU_L.5  */

/* ---- 时基 ---- */
/* 中心对齐：f_pwm = f_tim / (2 × ARR)。
 * 170 MHz / (2 × 4250) = 20000.0 Hz，正好 20 kHz。
 * 选中心对齐而不是边沿对齐，是为了：① 三相 PWM 对称，电流纹波小；
 * ② 计数器在 ARR 处三相下桥同时导通，是低边采样的天然采样点（见 bsp_adc 的说明）。 */
#define BSP_BOARD_INV_PWM_ARR                                           (4250U)
#define BSP_BOARD_INV_PWM_FREQ_HZ                                       (20000U)

/* 死区时间（ns）。TIM1 时钟 170 MHz → tDTS = 5.882 ns。
 *
 * 谁负责死区？两道都在：
 *   - DRV8353 的 TDRIVE 状态机自带"自动死区"（靠 VGS 握手判断栅极真的关断了才开另一路），
 *     所以即使 MCU 死区给 0，芯片层面也不会直通；
 *   - MCU 侧再插一段固定死区作为兜底，防止 INHx/INLx 在极端情况下同时有效时把
 *     有效死区压到 0（虽然 6x 模式下 1/1 是 Hi-Z，但 1/0 与 0/1 的切换仍需时间）。
 * 200 ns 的依据：IPT015N10N5 的关断延时量级 + 栅极电阻 10k 下拉的放电时间余量；
 * 实际以示波器量 SHx 上升沿与 GHx 下降沿的重叠为准，宁可先大后小。 */
#define BSP_BOARD_INV_DEADTIME_NS                                       (200U)

/* ADC 注入组触发点在计数器顶点前 8 个计数（≈47 ns）处翻高。
 * ADC 时钟 42.5 MHz（周期 23.5 ns），8 个计数是它的两倍，稳。 */
#define BSP_BOARD_INV_TRGO_OFFSET                                       (8U)

/************************************栅极驱动 / 杂项 GPIO***************************************/
/* DRV8353 ENABLE（第 33 脚）：高=放行栅极驱动。
 * 板上 R_DRV_EN_PD 10k 下拉 → 复位期间天然禁用。这是"最后一道闸"，
 * 它拉低时六路栅极被强制关断，与 PWM 引脚在做什么无关。 */
#define BSP_BOARD_DRV_EN_PORT                                           (GPIOB)
#define BSP_BOARD_DRV_EN_PIN                                            (GPIO_PIN_11)   /* J_MCU_R.6  */

/* TCAN1042 STB：低=正常通信，高=待机。板上 R_CAN_STB_PU 10k 上拉 → 复位期间天然待机。
 * ⚠️ 注意与 water_car_motor 板相反：那块板的 STB 是低=待机。换板时别照抄。 */
#define BSP_BOARD_CAN_STB_PORT                                          (GPIOB)
#define BSP_BOARD_CAN_STB_PIN                                           (GPIO_PIN_5)    /* J_MCU_L.20 */

/* 泄放电阻热许可：U4 比较器输出经 R_THERM_SER 100k 串入，板上无上下拉，靠比较器推挽驱动。 */
#define BSP_BOARD_DUMP_THERM_OK_PORT                                    (GPIOB)
#define BSP_BOARD_DUMP_THERM_OK_PIN                                     (GPIO_PIN_9)    /* J_MCU_L.24 */

/************************************霍尔（TIM2）***********************************************/
/* 三路霍尔既可以当普通 GPIO 读（本层当前做法），也可以接 TIM2 的 CH1/CH2/CH3 做输入捕获，
 * 从而直接拿到换相时刻。宏里同时给出 GPIO 与 AF 两套信息，将来升级不用改板级文件。
 *
 * 为什么不加内部上拉：板上霍尔前端有两个分支，由跳线 JP_Hx 选 ——
 *   DIRECT：信号经 4.7k 上拉到 3V3，MCU 直连；
 *   5V_DIV：先串电阻再对地下拉分压，把 5V 霍尔降到 3.3V 以内。
 * 两种情况下 MCU 内部再并一个上拉都不对（DIRECT 重复、5V_DIV 会破坏分压比）。 */
#define BSP_BOARD_HALL_A_PORT                                           (GPIOA)
#define BSP_BOARD_HALL_A_PIN                                            (GPIO_PIN_15)   /* J_MCU_L.15 */
#define BSP_BOARD_HALL_B_PORT                                           (GPIOB)
#define BSP_BOARD_HALL_B_PIN                                            (GPIO_PIN_3)    /* J_MCU_L.18 */
#define BSP_BOARD_HALL_C_PORT                                           (GPIOB)
#define BSP_BOARD_HALL_C_PIN                                            (GPIO_PIN_10)   /* J_MCU_R.5  */
#define BSP_BOARD_HALL_AF                                               (GPIO_AF1_TIM2)

/************************************SPI1（DRV8353 寄存器）*************************************/
/* 只做寄存器配置，速率不高（分频后 5~10 MHz 足够），但 MISO 是关键路径：
 * 读回的故障寄存器要靠它，所以走 AF 推挽而不是软件模拟。 */
#define BSP_BOARD_SPI_SCK_PORT                                          (GPIOA)
#define BSP_BOARD_SPI_SCK_PIN                                           (GPIO_PIN_5)    /* J_MCU_R.13 */
#define BSP_BOARD_SPI_MISO_PORT                                         (GPIOA)
#define BSP_BOARD_SPI_MISO_PIN                                          (GPIO_PIN_6)    /* J_MCU_R.14 */
#define BSP_BOARD_SPI_MOSI_PORT                                         (GPIOA)
#define BSP_BOARD_SPI_MOSI_PIN                                          (GPIO_PIN_7)    /* J_MCU_R.11 */
#define BSP_BOARD_SPI_AF_MASK                                           \
    (BSP_BOARD_SPI_SCK_PIN | BSP_BOARD_SPI_MISO_PIN | BSP_BOARD_SPI_MOSI_PIN)
#define BSP_BOARD_SPI_AF                                                (GPIO_AF5_SPI1)

/* 片选走软件 GPIO，不用硬件 NSS：DRV8353 要求 CS 在帧间保持高，
 * 且一次配置要连发多个 16 位帧，用软件控制更容易做到"整段拉低、整段拉高"。 */
#define BSP_BOARD_SPI_CS_PORT                                           (GPIOA)
#define BSP_BOARD_SPI_CS_PIN                                            (GPIO_PIN_4)    /* J_MCU_R.16 */

/************************************ADC*******************************************************/
/* ⚠️ 通道顺序 = 转换顺序 = 结果寄存器下标，三者必须一致。
 *    改这里必须同步改 bsp_adc.c 里的 Rank 赋值与缓冲区长度。 */

/* ---- 注入组：与 PWM 同步，一个 PWM 周期采一次，硬件上限 4 个通道 ---- */
#define BSP_BOARD_ADC_SO_U_PORT                                         (GPIOA)
#define BSP_BOARD_ADC_SO_U_PIN                                          (GPIO_PIN_1)    /* J_MCU_R.17 */
#define BSP_BOARD_ADC_SO_U_CHANNEL                                      (ADC_CHANNEL_2) /* rank1 */
#define BSP_BOARD_ADC_SO_V_PORT                                         (GPIOA)
#define BSP_BOARD_ADC_SO_V_PIN                                          (GPIO_PIN_3)    /* J_MCU_R.15 */
#define BSP_BOARD_ADC_SO_V_CHANNEL                                      (ADC_CHANNEL_4) /* rank2 */
#define BSP_BOARD_ADC_SO_W_PORT                                         (GPIOA)
#define BSP_BOARD_ADC_SO_W_PIN                                          (GPIO_PIN_0)    /* J_MCU_R.20 */
#define BSP_BOARD_ADC_SO_W_CHANNEL                                      (ADC_CHANNEL_1) /* rank3 */
#define BSP_BOARD_ADC_INJ_MASK                                          \
    (BSP_BOARD_ADC_SO_U_PIN | BSP_BOARD_ADC_SO_V_PIN | BSP_BOARD_ADC_SO_W_PIN)

/* ---- ADC1 规则组：慢信号，连续转换 + DMA 循环搬运 ---- */
#define BSP_BOARD_ADC_TEMP_MOS_PORT                                     (GPIOA)
#define BSP_BOARD_ADC_TEMP_MOS_PIN                                      (GPIO_PIN_2)    /* J_MCU_R.18 */
#define BSP_BOARD_ADC_TEMP_MOS_CHANNEL                                  (ADC_CHANNEL_3)
#define BSP_BOARD_ADC_TEMP_MOTOR_PORT                                   (GPIOB)
#define BSP_BOARD_ADC_TEMP_MOTOR_PIN                                    (GPIO_PIN_1)    /* J_MCU_R.10 */
#define BSP_BOARD_ADC_TEMP_MOTOR_CHANNEL                                (ADC_CHANNEL_12)
#define BSP_BOARD_ADC_SAFE_PORT                                         (GPIOB)
#define BSP_BOARD_ADC_SAFE_PIN                                          (GPIO_PIN_0)    /* J_MCU_R.9 */
#define BSP_BOARD_ADC_SAFE_CHANNEL                                      (ADC_CHANNEL_15)
#define BSP_BOARD_ADC1_REG_MASK                                         \
    (BSP_BOARD_ADC_TEMP_MOS_PIN | BSP_BOARD_ADC_TEMP_MOTOR_PIN | BSP_BOARD_ADC_SAFE_PIN)

/* ---- ADC2 规则组：母线电压 ----
 * ⚠️ PC4 在 G431 上**只有 ADC2_IN5**，ADC1 没有这个通道，所以母线电压必须由 ADC2 采，
 *    不能和相电流塞进同一个 ADC。这也是本层要同时驱动 ADC1 和 ADC2 的原因。 */
#define BSP_BOARD_ADC_BUS_PORT                                          (GPIOC)
#define BSP_BOARD_ADC_BUS_PIN                                           (GPIO_PIN_4)    /* J_MCU_R.12 */
#define BSP_BOARD_ADC_BUS_CHANNEL                                       (ADC_CHANNEL_5)

/* ADC 满量程（mV）。VREF_CORE（J_MCU_R.8）同时接到 DRV8353 的 VREF 脚，
 * 是电流放大器的基准，也是 ADC 满量程。
 * ⚠️ 板子上没有任何器件主动产生 VREF_CORE，它依赖核心板把 P2-8 内部接到 3V3。
 *    上电第一件事：万用表量 P2-8 对 GND，确认 3.3 V 再谈电流标定。
 *    若实测不是 3300 mV，改这里，否则所有电流/电压换算都会按比例偏。 */
#define BSP_BOARD_ADC_VREF_MV                                           (3300U)

/************************************FDCAN1****************************************************/
/* 本板 CAN 走 PA11/PA12，**不是** PB8/PB9 —— 天然避开了 water_car_motor 那块板的
 * PB8-BOOT0 遗留问题（上电进 System Memory）。这是本板设计上比 water_car_motor 那块板干净的一处。
 *
 * ⚠️ 与稳压板同样的约束：核心板把 PA11/PA12 同时引到 Type-C 的 D−/D+。
 *    接 CAN 时**不得再插 USB 数据线**，否则 USB 收发器会挂到 CAN 总线上当额外负载。
 *    两块板共用同一颗核心板（WeAct Normal QFN48），这条风险相同；稳压工程已写，此处补齐。 */
#define BSP_BOARD_CAN_RX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_RX_PIN                                            (GPIO_PIN_11)   /* J_MCU_L.13 */
#define BSP_BOARD_CAN_TX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_TX_PIN                                            (GPIO_PIN_12)   /* J_MCU_L.14 */
#define BSP_BOARD_CAN_AF                                               (GPIO_AF9_FDCAN1)

#ifdef _cplusplus
}
#endif

#endif /* __BSP_BOARD_H__ */
