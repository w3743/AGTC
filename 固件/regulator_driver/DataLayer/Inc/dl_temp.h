#ifndef __DL_TEMP_H__
#define __DL_TEMP_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 3 路 NTC —— 温度换算
 *
 * ── 硬件（BOM 已核对）──────────────────────────────────────────────────────
 *   每路都是：3.3 V ── R_NTC_x_PU 10 kΩ ±1% ──┬── ADC
 *                                             └── NTC ── GND_SYS
 *   并联 100 nF 滤波（C_NTC_MAIN / C_NTC_SC / C_NTC_DUMP）。
 *   所以 NTC 在上臂、ADC 在中间、10 kΩ 在上方：
 *       V_adc = 3.3 × R_ntc / (10k + R_ntc)
 *       R_ntc = 10k × V_adc / (3.3 − V_adc)
 *
 *   这跟"10k 上拉 + NTC 下拉"是同一个意思，但方向要想清楚：
 *   **温度升高 → R_ntc 变小 → V_adc 变小**。读数下降才是变热，
 *   看到"电压低"就以为"温度低"是最常见的接反错误。
 *
 * ── NTC 型号 ───────────────────────────────────────────────────────────────
 *   TH_SC / TH_MAIN : 时恒 MF52A103F3950 (A1)，10 kΩ ±1%，B25/50 = 3950 K
 *   TH_DUMP         : ATI ATH10K1R8B3950K，10 kΩ，B25/50 = 3950 K
 *   三路都用 10 kΩ / B3950，所以一组常数就够了。
 *   ⚠️ 原理图准备文档提到"下一原理图修订计划只将 TH_SC、TH_MAIN 改为 MF52A103F3950(A1)"，
 *      也就是说当前 BOM 里 TH_SC/TH_MAIN 与 TH_DUMP 的料号可能还不一致，
 *      B 值相同所以算法不受影响，但**精度和公差**不一样，做温度保护门限时要留够余量。
 *
 * ── 换算公式（Beta 方程）────────────────────────────────────────────────────
 *   1/T = 1/T25 + (1/B) × ln(R/R25)，  T25 = 298.15 K (25 °C)
 *   T(°C) = 1/T − 273.15
 *
 *   这里用的是两点式 Beta，25 °C 附近准，远离 25 °C 会偏。
 *   B3950 的 NTC 在 0 °C 附近误差约 1 °C、在 80 °C 附近误差可到 3–5 °C。
 *   对"过温保护"够用，对"精确测温"不够 —— 需要更准就换 Steinhart-Hart 三系数模型，
 *   但要先有实测标定数据，本层不编造系数。
 *
 * ── 已知未闭合项 ───────────────────────────────────────────────────────────
 *   · NTC 是外置的（OFF_BOARD_EXTERNAL），引线电阻、接插件接触电阻都会叠加到读数上；
 *   · 热安装方式（贴哪里、有没有导热胶）未定，测到的是"探头温度"不是"器件温度"；
 *   · 响应时间未实测，做快速过温保护时不能只靠它。
 ********************************************************************************************/

/************************************Type****************************************************/
typedef enum _DL_TEMP_E_
{
    DL_TEMP_MAIN                                                    = 0,    /* NTC_MAIN  P2.6  PB11 主功率级 */
    DL_TEMP_SC                                                      = 1,    /* NTC_SC    P2.12 PC4  超容 */
    DL_TEMP_DUMP                                                    = 2,    /* NTC_DUMP  P1.5  PB12 泄放电阻 */
    DL_TEMP_NUM                                                     = 3
}DL_TEMP_E;

/* NTC 参数。换料号只改这几行。 */
#define DL_TEMP_NTC_R25_OHM                                             (10000.0F)
#define DL_TEMP_NTC_B                                                   (3950.0F)
#define DL_TEMP_PULLUP_OHM                                              (10000.0F)

/* 无效读数。超出 −55 ~ 150 °C 一律按无处理，避免把开路(≈3.3 V)当成 −273 °C。 */
#define DL_TEMP_INVALID_C                                               (-273.15F)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_temp_get_c
* Descriptions  :   Read one NTC in degrees Celsius via the Beta equation.
* Input         :
*                   which: channel selector
* Output        :
*                   none
* Return        :
*                   temperature in degC, DL_TEMP_INVALID_C when the reading is not plausible
*********************************************************************************************/
extern F32 dl_temp_get_c
(
    IN  DL_TEMP_E   which
);

/********************************************************************************************
* Function name :   dl_temp_get_raw_mv
* Descriptions  :   Raw NTC pin voltage, for diagnostics. Remember: hotter reads LOWER.
* Input         :
*                   which: channel selector
* Output        :
*                   none
* Return        :
*                   pin voltage in mV
*********************************************************************************************/
extern U32 dl_temp_get_raw_mv
(
    IN  DL_TEMP_E   which
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_TEMP_H__ */
