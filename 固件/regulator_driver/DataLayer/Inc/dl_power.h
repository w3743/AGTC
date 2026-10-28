#ifndef __DL_POWER_H__
#define __DL_POWER_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 电流与泄放功率 —— 只做测量，不做保护
 *
 * ── 电流 ────────────────────────────────────────────────────────────────────
 *   R_SC_SHUNT 0.5 mΩ 6 W 1%（HoLRS2512-6W-0.5mR-1%，四端开尔文）
 *   → DRV8161 CSA（CSAGAIN 30 kΩ → 20 V/V，CSAREF 板载 3.3 V）
 *   → SO ≈ 1.65 V + 0.010 V/A × I_SC（放电为正）
 *
 *   换算本身在 BSP 里（bsp_adc_get_so_ma），本层只是把它转成 A 并缓存。
 *
 *   ⚠️ 三个必须记住的限制：
 *     1) TI 给的 SO 线性输出范围是 0.25 V ~ CSAREF−0.25 V，
 *        所以可用量程约 ±140 A，不是纸面的 ±165 A；
 *     2) 输入偏移 ±1.94 mV（25 °C，gain=20）等效 ±3.88 A ——
 *        也就是说"读到 0 A"只代表绝对值小于约 3.9 A；
 *     3) 增益误差、shunt 公差与温漂、参考误差还会叠加。
 *   ⟹ 这个电流可以拿来算功率、做趋势、算 SOC 增量，**不能**拿来当 80 A 跳闸判据。
 *      80 A 的保护是 DRV_SO → COMP1/COMP2 → TIM1 Break 的硬件路径（尚未实现）。
 *
 * ── 泄放功率 ────────────────────────────────────────────────────────────────
 *   Pavg ≈ D × V_bus² / R_dump
 *   只算电阻的导通损耗，不含开关损耗和电感残余能量，是**近似值**。
 *
 *   ⚠️ R_dump 是**外置**的，接在 J_DUMP 上，原理图里只有连接器没有具体型号，
 *      所以本层不能写死一个电阻值 —— 调用者必须自己传 R_mohm 进来。
 *      传错这个数，功率估算就跟着错，而错误的功率估算会直接烧电阻。
 ********************************************************************************************/

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_power_get_current_ma
* Descriptions  :   Signed supercapacitor current in mA, positive = discharging.
*                   See the caveats above; this is a measurement, not a trip point.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   current in mA
*********************************************************************************************/
extern S32 dl_power_get_current_ma
(
    VOID
);

/********************************************************************************************
* Function name :   dl_power_get_current_a
* Descriptions  :   Same value expressed in amperes, for readability in float math
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   current in A
*********************************************************************************************/
extern F32 dl_power_get_current_a
(
    VOID
);

/********************************************************************************************
* Function name :   dl_power_estimate_dump_mw
* Descriptions  :   Estimate the average power burnt by the external dump resistor.
*                   Pavg = duty * V_bus^2 / R  (conduction loss only).
* Input         :
*                   bus_mv  : bus voltage in mV
*                   duty    : dump PWM duty, 0.0 ~ 1.0
*                   r_mohm  : external dump resistance in milliohm (caller supplied!)
* Output        :
*                   none
* Return        :
*                   estimated average power in mW, 0 when r_mohm is 0
*********************************************************************************************/
extern U32 dl_power_estimate_dump_mw
(
    IN  U32     bus_mv,
    IN  F32     duty,
    IN  U32     r_mohm
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_POWER_H__ */
