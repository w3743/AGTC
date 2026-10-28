#ifndef __BSP_INVERTER_H__
#define __BSP_INVERTER_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 三相逆变器 —— 无刷电机功率级（TIM1 六路互补 PWM）
 *
 * 为什么叫 inverter 而不是 pwm：
 *   "PWM" 说的是**手段**，"逆变器"说的是**对象**。本模块管的是一台三相无刷电机的
 *   功率级：六路互补输出、死区、硬件刹车、ADC 同步触发。它和 water_car_motor 工程里的
 *   bsp_hbridge（H 桥，直流有刷，1 路 PWM + 1 路方向）是同层不同物，
 *   名字上先分开，看代码的人就不会把两者搞混。
 *
 * 引脚（依据 当前原理图连接.json + DS12589 Rev6 Table 12，全部集中在 bsp_board.h）：
 *   PWM_UH  PA8   TIM1_CH1    AF6  → DRV8353 INHA（第 34 脚）
 *   PWM_VH  PA9   TIM1_CH2    AF6  → DRV8353 INHB（第 36 脚）
 *   PWM_WH  PA10  TIM1_CH3    AF6  → DRV8353 INHC（第 38 脚）
 *   PWM_UL  PB13  TIM1_CH1N   AF6  → DRV8353 INLA（第 35 脚）
 *   PWM_VL  PB14  TIM1_CH2N   AF6  → DRV8353 INLB（第 37 脚）
 *   PWM_WL  PB15  TIM1_CH3N   AF6  → DRV8353 INLC（第 39 脚）
 *   BREAK_N PB12  TIM1_BKIN   AF6  ← DRV8353 nFAULT + U3 比较器，线与，2.2k 上拉
 *
 * ⚠️ 三个关键前提，改代码前先看这里
 *
 * 1) DRV8353 工作在 6x PWM 模式（PWM_MODE=00b，也是寄存器默认值），真值表（手册 Table 1）：
 *        INL  INH | GL  GH | SH
 *         0    0  | L   L  | Hi-Z     ← 双关，电机自由滑行
 *         0    1  | L   H  | H        ← 上桥导通
 *         1    0  | H   L  | L        ← 下桥导通
 *         1    1  | L   L  | Hi-Z     ← 双关（注意：这里也不会直通）
 *    所以：INH=CHx、INL=CHxN，MCU 侧只要是互补输出就能对上；即使上下同时为 1，
 *    DRV 也会自己进 Hi-Z，不会烧桥。真正的直通风险来自死区不足，不来自电平组合。
 *
 * 2) 占空比 0 的含义是"下桥常通"= 电制动，不是断电。
 *    CCRx=0 → CHx 恒低 → CHxN 恒高 → INL=1,INH=0 → 三相下桥全开，绕组被短路。
 *    要真正断电请用 bsp_gpio_drv_set_enable(FALSE)（ENABLE 拉低，六路栅极强制关断），
 *    或者 bsp_inverter_set_output_enable(FALSE)（MOE=0，本层把六路全部拉低 → DRV 进 Hi-Z）。
 *
 * 3) BREAK_N 是硬件刹车。它一旦被拉低，MOE 由硬件清零、六路输出立刻进入 off-state，
 *    不需要软件参与。本层把 AOE 关掉（AutomaticOutput=DISABLE），所以刹车后必须由软件
 *    调 bsp_inverter_clear_break() + bsp_inverter_start() 才能恢复 —— 这是故意的，避免故障
 *    反复自动重启。off-state 的引脚电平由 OSSI/OSSR + OISx 决定，本层全部设为"低"，
 *    即六路 INH/INL 同时为低 → DRV 进 Hi-Z。
 *
 * ⚠️ PA6（SPI1_MISO）与 PA11（FDCAN1_RX）在别的复用功能里分别是 TIM1_BKIN 与 TIM1_BKIN2。
 *    本模块把 Break2State 写死 DISABLE，也不把 BKIN 源配到 PA6：
 *    否则 SPI 每传一帧、CAN 每收一帧都可能触发刹车。
 ********************************************************************************************/

/* 下方三个宏保留原有名字，避免上层已经引用的常量一次性全改；
 * 真正的数值定义已上移到 bsp_board.h，这里只是别名。 */
#define BSP_INVERTER_ARR                                                (BSP_BOARD_INV_PWM_ARR)
#define BSP_INVERTER_FREQ_HZ                                            (BSP_BOARD_INV_PWM_FREQ_HZ)
#define BSP_INVERTER_DEADTIME_NS                                        (BSP_BOARD_INV_DEADTIME_NS)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_inverter_init
* Descriptions  :   Configure TIM1 as a 20 kHz center-aligned 3-phase complementary PWM
*                   with dead time, and wire BREAK_N to TIM1_BKIN as a hardware trip.
*                   Outputs stay in the off state (MOE = 0) until bsp_inverter_start().
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_inverter_start
* Descriptions  :   Enable the six outputs (set MOE). The duty already programmed by
*                   bsp_inverter_set_duty() takes effect immediately, so set the duty BEFORE
*                   calling this function if you do not want to start from 0 % (brake).
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_start
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_inverter_stop
* Descriptions  :   Clear MOE so all six driver inputs are driven low (DRV8353 -> Hi-Z).
*                   The counter keeps running so the ADC trigger stays alive.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_stop
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_inverter_set_duty
* Descriptions  :   Set the three phase duties in one shot
*                   duty = 1.0 -> high side fully on, 0.0 -> low side fully on (brake)
*                   Values are clamped to [0, 1].
* Input         :
*                   duty_u / duty_v / duty_w: 0.0 ~ 1.0
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_set_duty
(
    IN  F32     duty_u,
    IN  F32     duty_v,
    IN  F32     duty_w
);

/********************************************************************************************
* Function name :   bsp_inverter_set_output_enable
* Descriptions  :   Switch the six outputs on/off without touching the programmed duty
*                   TRUE  -> MOE = 1, outputs follow the duty
*                   FALSE -> MOE = 0, all six inputs low, DRV8353 goes Hi-Z (coast)
* Input         :
*                   enable: TRUE = run, FALSE = coast
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_set_output_enable
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bsp_inverter_break_occurred
* Descriptions  :   Tell whether TIM1 has been tripped by BREAK_N since the last clear
*                   The flag is latched by hardware, so a short fault is not lost.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE  : break was triggered
*                   FALSE : no break event
*********************************************************************************************/
extern BOOL bsp_inverter_break_occurred
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_inverter_clear_break
* Descriptions  :   Clear the latched break flag. MOE is NOT restored by this call.
*                   Re-arm with bsp_inverter_start() only after the fault source is gone,
*                   otherwise the driver trips again on the next PWM edge.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_inverter_clear_break
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_inverter_get_dtg
* Descriptions  :   Convert a dead time in nanoseconds into the TIM1 BDTR DTG field.
*                   Exposed so the value can be checked on a scope without recompiling.
* Input         :
*                   ns: requested dead time in nanoseconds
* Output        :
*                   none
* Return        :
*                   DTG[7:0] encoded value (see RM0440 dead-time generator table)
*********************************************************************************************/
extern U32 bsp_inverter_get_dtg
(
    IN  U32     ns
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_INVERTER_H__ */
