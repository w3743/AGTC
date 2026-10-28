#ifndef __BSP_H__
#define __BSP_H__

#include "type.h"
#include "bsp_callback.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * FOC 驱动模块 —— 板级支持包总入口
 *
 * 这块板到底有哪些东西（依据 当前原理图连接.json，169 器件 / 506 脚 / 102 网）：
 *
 *   ┌──────────────────────────────────────────────────────────────────────────┐
 *   │ 功率级   DRV8353 三相栅极驱动 + 6×IPT015N10N5 三相桥                     │
 *   │          三只 0.5 mΩ 低边采样电阻（RSU/RSV/RSW）                          │
 *   │ 母线     48 V 输入，母线电压采样 180k/10k 分压 → BUS_ADC                 │
 *   │ 泄放     DUMP 电阻 + 比较器过热许可 → DUMP_THERM_OK_GPIO                 │
 *   │ 反馈     三相霍尔（HALL_A/B/C，4.7k 上拉或 5V 分压，跳线选择）            │
 *   │ 温度     NTC ×2（MOS / 电机）                                            │
 *   │ 通信     TCAN1042HGV（500 kbit/s）+ BREAK_N 硬件刹车链                   │
 *   │ 供电     5V_BUCK → 核心板 → MCU_3V3 / VREF_CORE（同时是 CSA 基准）       │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * ── 分层的边界 ──────────────────────────────────────────────────────────────
 *   BSP（本层）只做三件事：**配引脚、配外设、给出安全可用的原语**。
 *   它不知道什么是电流环、什么是 SVPWM、什么是换相表；
 *   它也不做任何闭环、不做标定换算、不做保护策略。
 *   凡是"该不该开""开到多大""出事了怎么办"的决定，都在 DL/BL 层。
 *
 * ── 文件组织 ────────────────────────────────────────────────────────────────
 *   Bsp/Inc/bsp_board.h        全工程**唯一**的引脚 / 复用号 / ADC 通道定义处
 *   Bsp/Inc|Src/Gpio           bsp_gpio      DRV_ENABLE、CAN_STB、泄放热许可、BREAK_N 读回
 *   Bsp/Inc|Src/Inverter       bsp_inverter  三相逆变器：TIM1 六路互补 PWM、死区、刹车、ADC 触发
 *   Bsp/Inc|Src/Adc            bsp_adc       注入组（相电流）+ 规则组（母线/温度/安全）
 *   Bsp/Inc|Src/Hall           bsp_hall      三路霍尔电平读取
 *   Bsp/Inc|Src/Spi            bsp_spi       SPI1 访问 DRV8353 寄存器
 *   Bsp/Inc|Src/Can            bsp_can       FDCAN1 + 收发器待机控制 + 接收回调
 *   Bsp/Inc|Src/Callback       bsp_callback  回调注册表（BSP 唯一的上行通道）
 *
 *   每个子模块自带时钟使能和引脚配置，互不依赖，单独拿出来也能编；
 *   但**引脚实体只允许定义在 bsp_board.h**，子模块里最多写短别名。
 *   好处是：查问题时可以只怀疑一个文件；改板时只改 bsp_board.h 一个文件。
 *   坏处是：初始化顺序有隐含约束，必须由 bsp_init() 统一编排 —— 见下面 bsp_init 的说明。
 *
 * ── 与 water_car_motor 工程的命名对照 ───────────────────────────────────────
 *   本工程          bsp_inverter   三相逆变器（无刷）  6 路互补 PWM + 霍尔 + 刹车
 *   water_car_motor bsp_hbridge    H 桥（直流有刷）    1 路 PWM + 1 路方向
 *   两者同层不同物，刻意不同名，避免看代码时把"PWM"当成同一件事。
 *
 * ── 上电默认状态（重要）─────────────────────────────────────────────────────
 *   bsp_init() 执行完毕后，板子处于"静默但已就绪"：
 *     · 功率级关闭   —— DRV_ENABLE = 低，六路栅极强制关断，绕组开路，电机自由滑行
 *                       （注意这跟"占空比 0"不是一回事，占空比 0 是下桥全开 = 电制动）
 *     · 总线待机     —— CAN_STB = 高，收发器不驱动总线
 *     · PWM 计数器在跑，但 MOE = 0，六路输出无效
 *     · ADC 已在连续采样（母线/温度）与注入采样（相电流，跟着 PWM 走）
 *     · DRV8353 寄存器已探测，但**没有写入任何配置**（保持芯片自己的复位默认值）
 *   这个状态下板子不会动、也不会干扰总线，可以放心用调试器挂上去看寄存器。
 ********************************************************************************************/

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_get_tick
* Descriptions  :   Get the HAL tick in milliseconds
*                   Thin wrapper so that upper layers do not include stm32g4xx_hal.h just
*                   for a timestamp.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   Milliseconds since HAL_Init()
*********************************************************************************************/
extern U32 bsp_get_tick
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_init
* Descriptions  :   Bring the whole board up in a safe order and leave it silent-but-ready.
*
*                   Order is safety-critical and must not be rearranged:
*                     1. GPIO  —— first, so the gate driver is disabled and the
*                                 transceiver is in standby before anything can toggle
*                     2. PWM   —— TIM1 configured, counter running, MOE still 0
*                     3. ADC   —— needs the running counter, see note below
*                     4. HALL  —— plain inputs, no bias
*                     5. SPI   —— only then can the DRV8353 be probed
*                     6. CAN   —— configure last, transceiver STILL in standby
*
*                   ⚠️ Step 3 must come after step 2. The injected ADC group is triggered
*                      by TIM1_TRGO (OC4REF). If the ADC is armed while TIM1 is not
*                      counting, JEOC never fires and bsp_adc_get_phase_raw() blocks
*                      forever waiting for a flag that will never come.
*
*                   ⚠️ This function does NOT release the transceiver and does NOT enable
*                      the power stage. Both are decisions for the layer above:
*                        bsp_can_start()              —— join the bus
*                        bsp_power_stage_enable(TRUE) —— apply voltage to the motor
*
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS : everything configured, DRV8353 answered on SPI
*                   TYPE_RET_FAIL    : DRV8353 did not answer (SPI dead or chip absent)
*                                      ⚠️ The rest of the board is still configured even
*                                      when FAIL is returned, on purpose: a board with a
*                                      dead gate driver is still usable for hall tests and
*                                      still reachable over CAN for diagnosis. Do not treat
*                                      FAIL as "board unusable".
*********************************************************************************************/
extern TYPE_RET_E bsp_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_power_stage_enable
* Descriptions  :   Turn the three-phase power stage on or off through the safe sequence.
*
*                   ON  (enable = TRUE):
*                     1. read DRV8353 FAULT, refuse to start if a fault is latched
*                     2. set all three duties to 50 %  -> zero voltage vector
*                     3. MOE = 1                      -> MCU side starts switching
*                     4. DRV_ENABLE = high            -> gate driver follows the inputs
*
*                   Why 50 % and not 0 %: in 6x PWM mode duty 0 means "low side always
*                   on", i.e. the three windings are shorted together (electrical brake).
*                   At 50 % all three phases sit at VBUS/2 on average, so the line-to-line
*                   voltage is zero and no current flows -- the motor is free, not braked.
*                   Starting from 0 % would slam a full brake on at the first enable.
*
*                   Why the enable pin goes last: while DRV_ENABLE is low the gate driver
*                   ignores INHx/INLx completely and holds all six gates off. So the MCU
*                   can settle MOE and the compare values in a state where nothing can
*                   switch, and the very first switching cycle the chip ever sees is
*                   already a balanced zero vector.
*
*                   OFF (enable = FALSE):
*                     1. DRV_ENABLE = low  -> hardware kill, fastest path, no software
*                                             timing involved
*                     2. MOE = 0           -> MCU side stops too
*                     3. duties back to 50 % so the next enable starts from zero vector
*
*                   ⚠️ This does not clear a latched TIM1 break. If BREAK_N was pulled low,
*                      MOE cannot be set again until bsp_inverter_clear_break() is called.
*
* Input         :
*                   enable : TRUE = apply voltage to the motor, FALSE = hard off
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS : state applied
*                   TYPE_RET_FAIL    : refused, DRV8353 reports a fault (only for TRUE)
*********************************************************************************************/
extern TYPE_RET_E bsp_power_stage_enable
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bsp_probe_drv8353
* Descriptions  :   Verify that the DRV8353 really answers on SPI1.
*
*                   Method: write 0x0001 to register 0x7 (Driver Configuration), read it
*                   back, expect 0x0001, then restore 0x0000.
*
*                   Why register 0x7 is the right one for a read-back test (verified
*                   against SLVSDY6A §8.6.2.6, Table 19):
*                     · reset value is 0x0000 and only bit 0 (CAL_MODE) is implemented,
*                       so the test pattern is unambiguous;
*                     · CAL_MODE = 1 only selects *which* calibration routine the
*                       CSA_CAL_x bits trigger -- it does not start a calibration by
*                       itself (§8.3.4.3), so a brief 1 -> 0 leaves no side effect;
*                     · unlike 0x3/0x4/0x5 it has no LOCK bit that would lock the
*                       register map against later legitimate writes.
*
*                   Why a read-only test would not be enough: a floating or stuck-low
*                   SDO returns 0x000, which is exactly the reset value of most
*                   registers. Only a write/read-back round trip distinguishes
*                   "chip is answering" from "line is stuck".
*
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE  : SPI link is alive
*                   FALSE : no answer, or 0x7 did not hold the pattern
*********************************************************************************************/
extern BOOL bsp_probe_drv8353
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif
