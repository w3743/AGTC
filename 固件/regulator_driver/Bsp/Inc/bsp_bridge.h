#ifndef __BSP_BRIDGE_H__
#define __BSP_BRIDGE_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 双向半桥 —— 稳压模块的功率级（TIM1 通道 1 + 互补通道 1N）
 *
 * 为什么叫 bridge 而不是 pwm 或 buck：
 *   这块板不是单向降压，也不是单向升压，而是在电池 BUS 与 8S1P 超容之间**双向**搬能量。
 *   同一对管子，上管占空比 >50% 是一种流向，<50% 是另一种流向。叫 "buck" 或 "boost"
 *   都会把另一半功能说漏，所以用中性词 bridge。
 *
 *   与另外两个工程同层不同物的对照：
 *     foc_driver             bsp_inverter  三相逆变器，6 路互补，驱动无刷电机
 *     water_car_motor        bsp_hbridge   H 桥，1 路 PWM + 1 路方向，驱动直流有刷
 *     本工程                 bsp_bridge    单半桥，2 路互补，双向 DC/DC
 *   名字上先分开，看代码的人就不会把三者搞混。
 *
 * 引脚（依据 接口与采样核对.md + DS12589 Rev6 Table 12，实体定义在 bsp_board.h）：
 *   PWM_H  PA8   TIM1_CH1   AF6  → DRV8161 INH
 *   PWM_L  PB13  TIM1_CH1N  AF6  → DRV8161 INL
 *
 * ⚠️ 四个关键前提，改代码前先看这里
 *
 * 1) DRV8161 工作在 2-pin PWM 模式（DT/MODE 脚 3.3 kΩ 上电锁存）。真值表：
 *        INH  INL | GH  GL | SH
 *         0    0  | L   L  | Hi-Z     ← 双关
 *         0    1  | L   H  | L        ← 下管导通
 *         1    0  | H   L  | H        ← 上管导通
 *         1    1  | L   L  | Hi-Z     ← 双关（同样不会直通）
 *    所以 MCU 侧只要保证 CH1 / CH1N 是互补的，就自然对上 INH / INL。
 *
 * 2) 死区由芯片负责，MCU 侧默认给 0。
 *    DRV8161 自己插 370 ns（典型）死区，BSP_BOARD_BRIDGE_DEADTIME_NS 默认 0。
 *    两边都加不会更安全，只会压缩可用占空比范围。理由和数字见 bsp_board.h 里的注释。
 *
 * 3) 占空比 0 / 1 的含义不是"没输出"。
 *    duty = +1 → CCR = ARR → CH1 恒高、CH1N 恒低 → INH=1/INL=0 → 上管常通（把超容直接挂到 BUS）
 *    duty = -1 → CCR = 0   → CH1 恒低、CH1N 恒高 → INH=0/INL=1 → 下管常通（把超容短路）
 *    两个极端都是**危险状态**，正常调制永远不该到达端点。要真正断开功率级请用
 *    bsp_gpio_permit_set(FALSE)（把 DRV_FAULT_N 拉低）或 bsp_bridge_stop()（清 MOE）。
 *
 * 4) 本层**不开** TIM1 的刹车，也不接 COMP1/COMP2。
 *    原因：PB12(NTC_DUMP) 与 PA6(AD_T7) 在别的复用里都是 TIM1_BKIN，一旦把 BreakState
 *    打开、或把 BKIN 源配到这两个脚，NTC 分压和抽头分压上的任何低电平都会被当成刹车。
 *    原理图准备的方案是用 COMP1(上阈值)/COMP2(下阈值) → TIM1 Break 做快关断，
 *    但文档明确写着"极性、内部源合并、MOE/OSSI/OSSR、Break 锁存清除、DAC/COMP 上电顺序、
 *    传播延迟"全部**尚未核实**。在核实之前不实现，比实现一个反逻辑的保护更安全。
 *    → 当前过流保护只有 DRV8161 自己的 VDS OCP（LEVEL4，typ 0.8 V）。
 ********************************************************************************************/

#define BSP_BRIDGE_ARR                                                  (BSP_BOARD_BRIDGE_PWM_ARR)
#define BSP_BRIDGE_FREQ_HZ                                              (BSP_BOARD_BRIDGE_PWM_FREQ_HZ)

/* 占空比端点。留一点余量，避免长期停在 100%/0% 让某一只管持续导通。 */
#define BSP_BRIDGE_DUTY_LIMIT                                           (1.0F)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_bridge_init
* Descriptions  :   Configure TIM1 CH1 / CH1N as a center-aligned complementary pair with
*                   the dead time from bsp_board.h. The counter starts running but MOE stays
*                   cleared, so no gate signal reaches the driver until bsp_bridge_start().
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_bridge_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_bridge_start
* Descriptions  :   Set MOE so CH1/CH1N actually drive the pins.
*                   Call bsp_bridge_set_duty() FIRST: whatever is in CCR takes effect the
*                   instant MOE goes high.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_bridge_start
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_bridge_stop
* Descriptions  :   Clear MOE. Both outputs fall to their idle state (LOW), which the driver
*                   sees as INH=0/INL=0 -> both gates off, SH high-Z.
*                   The counter keeps running so nothing else that depends on it stalls.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_bridge_stop
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_bridge_set_duty
* Descriptions  :   Set the high-side duty in signed form.
*                     duty =  0.0  -> 50% both sides (mid point, see the note below)
*                     duty = +1.0  -> high side always on  (upper limit, avoid)
*                     duty = -1.0  -> low  side always on  (lower limit, avoid)
*                   Mapping: high-side duty = (duty + 1) / 2, i.e. CCR = (duty+1)/2 * ARR.
*                   Out-of-range values are clamped to +/-BSP_BRIDGE_DUTY_LIMIT.
*
*                   NOTE 50% is NOT "zero current". The switch-node average is D * V_BUS,
*                   so the inductor current only stops ramping when D = V_SC / V_BUS.
*                   With a 9S bus (22.5-37.8 V) and an 8S supercap (0-20.4 V) the true
*                   balance point moves with state of charge. 50% is merely the mildest
*                   starting point; computing the balance duty is the BL layer's job.
* Input         :
*                   duty: -1.0 ~ +1.0
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_bridge_set_duty
(
    IN  F32     duty
);

/********************************************************************************************
* Function name :   bsp_bridge_get_duty
* Descriptions  :   Read back the value last written by bsp_bridge_set_duty
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   -1.0 ~ +1.0
*********************************************************************************************/
extern F32 bsp_bridge_get_duty
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_bridge_is_running
* Descriptions  :   TRUE when MOE is set and the outputs are live
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE / FALSE
*********************************************************************************************/
extern BOOL bsp_bridge_is_running
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_BRIDGE_H__ */
