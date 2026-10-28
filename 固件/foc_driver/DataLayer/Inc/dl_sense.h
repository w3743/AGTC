#ifndef __DL_SENSE_H__
#define __DL_SENSE_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 采样数据层 —— 三相无刷 + DRV8353
 *
 * 换算链路（依据 FOC 原理图网表 + DS12589 Rev6 Table 12）：
 *
 *   三相电流：0.5 mΩ 低边采样（RSU/RSV/RSW，四端开尔文接法）
 *             → DRV8353 CSA（增益 20 V/V，CSA Control 复位默认 10b）
 *             → SO_U / SO_V / SO_W → PA1 / PA3 / PA0（ADC1 注入组，与 PWM 顶点同步采样）
 *
 *             零点  = VREF/2（CSA 双向模式，VREF_DIV = 1b 复位默认）
 *             灵敏度 = R_shunt × GAIN = 0.5 mΩ × 20 = **10 mV/A**
 *             满量程 = ±(VREF/2) / 10 mV/A ≈ **±165 A** @ VREF = 3.3 V
 *
 *             ⟹ 这段换算本身在 bsp_adc_phase_raw_to_ma() 里做，本层只负责
 *                取三相、保持符号约定、必要时滤波，不再重复算一遍系数。
 *
 *   母线电压：180k / 10k 分压 → PC4（ADC2 规则组 rank1）
 *             分压比 = 10 / (180 + 10) = **1/19**
 *             48 V → 2.526 V；100 V → 5.263 V ⚠️ 超过 3.3 V 量程，见下方上限说明
 *
 *   温度：NTC ×2（MOS 一颗、电机一颗），10k 上拉 + B 值公式
 *         ⚠️ 两颗 NTC 的标称阻值与 B 值尚未从原理图确认，先给占位值。
 *            实测后只改 DL_SENSE_NTC_* 这几行，不动算法。
 *
 * ── 母线量程上限（上板前必须确认，这是会烧板子的事）────────────────────────
 *   分压比 1/19 配 3.3 V 满量程 → 理论可测 62.7 V。
 *   标称 48 V 母线留了约 30% 余量，正常；但**再生制动/泄放失效时母线会被顶高**，
 *   而 ADC 引脚超过 VDDA + 0.3 V 就开始走钳位二极管。
 *   上板第一步：用万用表量一次满母线下的 BUS_ADC 引脚电压，确认 < 3.0 V。
 *   若大于，必须在硬件上改分压比（BSP 侧改不了，这是电阻的事）。
 *
 * 对 BL 的承诺：
 *   - 返回值已带物理单位，BL 不再做任何分压/线性换算
 *   - 单次调用耗时恒定，无阻塞
 *   - 未标定的通道返回 0 并置 DL_SENSE_STATUS_UNCALIBRATED
 ********************************************************************************************/

/************************************Macro***************************************************/
/* 标定参数占位：拿到实测数据后只改这里，不改算法 */

/* 母线分压比（V_pin / V_bus）。180k 上、10k 下。
 * 实测方法：万用表量实际母线电压 V_bus 与 PC4 对地电压 V_pin，填 V_pin/V_bus。 */
#define DL_SENSE_BUS_DIVIDER                                            (1.0F / 19.0F)

/* 母线读数的一阶低通系数（0~1，越小越平滑）。
 * 母线本身有大电容，噪声主要来自开关耦合，0.1 对应约 10 个控制周期的时间常数。 */
#define DL_SENSE_BUS_LPF_ALPHA                                          (0.1F)

/* NTC 分压：上拉电阻接 3V3，NTC 对地。 */
#define DL_SENSE_NTC_PULLUP_OHM                                         (10000.0F)
#define DL_SENSE_NTC_NOMINAL_OHM                                        (10000.0F)   /* R25，待实测确认 */
#define DL_SENSE_NTC_NOMINAL_K                                          (298.15F)    /* 25 degC */
#define DL_SENSE_NTC_B_COEFFICIENT                                      (3950.0F)    /* B25/85，待实测确认 */

/* NTC 读数超出物理可能范围时的返回值（明显错误，方便上层直接判） */
#define DL_SENSE_TEMP_INVALID_C                                         (-273.15F)

/************************************Type****************************************************/
typedef enum _DL_SENSE_STATUS_E_
{
    DL_SENSE_STATUS_OK                                              = 0,  /* 已标定 */
    DL_SENSE_STATUS_UNCALIBRATED,                                         /* 参数未标定 */
    DL_SENSE_STATUS_NUM
}DL_SENSE_STATUS_E;

/* 相序。数值必须与 bsp_adc_get_phase_raw() 的输出顺序一致：
 * 0 = U（PA1/ADC1_IN2，注入 rank1）、1 = V（PA3/IN4，rank2）、2 = W（PA0/IN1，rank3）。 */
typedef enum _DL_SENSE_PHASE_E_
{
    DL_SENSE_PHASE_U                                                = 0,
    DL_SENSE_PHASE_V,
    DL_SENSE_PHASE_W,
    DL_SENSE_PHASE_NUM
}DL_SENSE_PHASE_E;

/* 温度通道：本板有两颗 NTC，必须分开取，不能像单电机板那样只给一个 get_temp_c()。 */
typedef enum _DL_SENSE_TEMP_E_
{
    DL_SENSE_TEMP_MOS                                               = 0,  /* 功率级散热器 */
    DL_SENSE_TEMP_MOTOR,                                                  /* 电机绕组附近 */
    DL_SENSE_TEMP_NUM
}DL_SENSE_TEMP_E;

/************************************Fcunction***********************************************/
/********************************************************************************************
* Function name :   dl_sense_init
* Descriptions  :   Init sense data layer
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_sense_init
(
    VOID
);

/********************************************************************************************
* Function name :   dl_sense_update
* Descriptions  :   Refresh all engineering values from the latest raw samples
*                   Call once per control tick before reading any getter
*                   The injected results are latched by hardware, so this never waits
*                   for a conversion; it just copies and filters.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_sense_update
(
    VOID
);

/********************************************************************************************
* Function name :   dl_sense_get_phase_current_ma
* Descriptions  :   Phase current in milliamps, signed
*                   Positive = current flowing from the bridge output into the low-side
*                   shunt (i.e. out of the motor terminal).
* Input         :
*                   phase: DL_SENSE_PHASE_U / V / W
* Output        :
*                   none
* Return        :
*                   signed current in mA, 0 when the phase index is invalid
*********************************************************************************************/
extern S32 dl_sense_get_phase_current_ma
(
    IN  DL_SENSE_PHASE_E    phase
);

/********************************************************************************************
* Function name :   dl_sense_get_bus_v
* Descriptions  :   Bus voltage in volt, low-pass filtered
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   bus voltage, V
*********************************************************************************************/
extern F32 dl_sense_get_bus_v
(
    VOID
);

/********************************************************************************************
* Function name :   dl_sense_get_temp_c
* Descriptions  :   NTC temperature in celsius
*                   Returns DL_SENSE_TEMP_INVALID_C when the reading is physically
*                   impossible (open or shorted NTC), so the caller can distinguish
*                   "sensor broken" from "really cold".
* Input         :
*                   which: DL_SENSE_TEMP_MOS / DL_SENSE_TEMP_MOTOR
* Output        :
*                   none
* Return        :
*                   temperature in degC
*********************************************************************************************/
extern F32 dl_sense_get_temp_c
(
    IN  DL_SENSE_TEMP_E     which
);

/********************************************************************************************
* Function name :   dl_sense_get_status
* Descriptions  :   Get calibration status
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   status
*********************************************************************************************/
extern DL_SENSE_STATUS_E dl_sense_get_status
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif
