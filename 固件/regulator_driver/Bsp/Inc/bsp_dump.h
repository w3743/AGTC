#ifndef __BSP_DUMP_H__
#define __BSP_DUMP_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 泄放（dump）—— 把 BUS 上多余的制动回馈能量烧在外置电阻上
 *
 * 信号链（依据 接口与采样核对.md 的「UCC27511A 全引脚与 BUS dump 驱动外围」一节）：
 *   PB7 / TIM4_CH2 ── 100Ω 串阻 ──► UCC27511A IN+（IN− 固定接 GND，非反相）
 *                                    │
 *                                    OUTH/OUTL 各经 10Ω ──► DUMP_FET_GATE
 *                                                             │
 *                                                        Q_DUMP (IRLB4030PBF)
 *                                                             │
 *                                                     外置泄放电阻（J_DUMP）
 *
 * 关键前提：
 *   1) 板上 **没有** 独立的硬件 BUS_OV 泄放触发。这条路径完全由软件决定，
 *      MCU 死机、PWM 卡住、GPIO 卡高，都会导致"该泄放时不泄放"或"一直泄放"。
 *      原理图准备文档原文：「IWDG 不保证覆盖这些故障」。
 *   2) PB7 上有 10 kΩ 下拉、UCC27511A IN+ 上也有 10 kΩ 下拉
 *      → MCU 复位期间栅极天然关断，泄放不动作。这是失效安全方向。
 *   3) 泄放功率近似 Pavg ≈ D × V_bus² / R（只算电阻导通损耗，不含开关损耗）。
 *      **本层不做任何功率/热限幅** —— 占空比给多少就输出多少，
 *      限幅与限时是 BL 层的事。
 *
 * 为什么用 TIM4_CH2 而不是再找一个 TIM1 通道：
 *   TIM1 已经全部分给半桥（CH1/CH1N），而且泄放和半桥是**两个独立的时间尺度**：
 *   半桥要精确的死区与互补，泄放只是一个可调的占空比，放同一个定时器反而互相牵制。
 ********************************************************************************************/

#define BSP_DUMP_ARR                                                    (BSP_BOARD_DUMP_PWM_ARR)
#define BSP_DUMP_FREQ_HZ                                                (BSP_BOARD_DUMP_PWM_FREQ_HZ)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_dump_init
* Descriptions  :   Configure TIM4 CH2 on PB7 as a plain edge-aligned PWM, duty forced to 0.
*                   The counter starts immediately; nothing is dissipated until
*                   bsp_dump_set_duty() is given a non-zero value.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_dump_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_dump_set_duty
* Descriptions  :   Set the dissipation duty cycle.
*                   0.0 = fully off, 1.0 = resistor permanently across the bus (will cook it).
*                   Values are clamped to 0.0 ~ 1.0.
* Input         :
*                   duty: 0.0 ~ 1.0
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_dump_set_duty
(
    IN  F32     duty
);

/********************************************************************************************
* Function name :   bsp_dump_get_duty
* Descriptions  :   Read back the value last written
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   0.0 ~ 1.0
*********************************************************************************************/
extern F32 bsp_dump_get_duty
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_DUMP_H__ */
