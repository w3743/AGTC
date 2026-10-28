/************************************包含头文件*************************************************/
#include <math.h>
#include "type.h"
#include "bsp_adc.h"
#include "dl_temp.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 绝对零度偏移与 25 °C 的开氏温度 */
#define DL_TEMP_KELVIN_OFFSET                                           (273.15F)
#define DL_TEMP_T25_K                                                   (298.15F)

/* 合理的读数区间（°C）。超出这个区间一律返回 DL_TEMP_INVALID_C。
 * 下限 −55：外置 NTC 引线开路时 ADC 读到接近满量程（R_ntc 计算值巨大）→ T 很低，
 *           这条限幅顺带把"探头没接"挡住。上限 150：超过这个温度 NTC 本身也不保准。 */
#define DL_TEMP_MIN_C                                                   (-55.0F)
#define DL_TEMP_MAX_C                                                   (150.0F)


/************************************静态函数声明************************************************/
static U16 _dl_temp_get_raw
(
    IN  DL_TEMP_E   which
);


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _dl_temp_get_raw
* 功能描述 :   按通道取 NTC 的 ADC 原始码值
*
*              三路分布：
*                NTC_MAIN → PB11 → ADC2_IN14
*                NTC_SC   → PC4  → ADC2_IN5
*                NTC_DUMP → PB12 → ADC1_IN11
*              注意 DUMP 那一路在 ADC1，另两路在 ADC2 —— 这是原理图定的，别想当然。
*
* 输入参数 :   which: 通道
* 输出参数 :   无
* 返回值   :   0 ~ 4095，非法通道返回 0
*********************************************************************************************/
static U16 _dl_temp_get_raw
(
    IN  DL_TEMP_E   which
)
{
    switch (which)
    {
        case DL_TEMP_MAIN: return bsp_adc_get_raw(2U, BSP_ADC2_IDX_NTC_MAIN);
        case DL_TEMP_SC:   return bsp_adc_get_raw(2U, BSP_ADC2_IDX_NTC_SC);
        case DL_TEMP_DUMP: return bsp_adc_get_raw(1U, BSP_ADC1_IDX_NTC_DUMP);
        default:           return 0U;
    }
}

/********************************************************************************************
* 函数名称 :   dl_temp_get_raw_mv
* 功能描述 :   NTC 引脚电压（mV），纯诊断用
*              记住方向：**越热读数越低**。R_ntc 变小 → 分压变小。
* 输入参数 :   which: 通道
* 输出参数 :   无
* 返回值   :   引脚电压（mV）
*********************************************************************************************/
U32 dl_temp_get_raw_mv
(
    IN  DL_TEMP_E   which
)
{
    return bsp_adc_raw_to_mv(_dl_temp_get_raw(which));
}

/********************************************************************************************
* 函数名称 :   dl_temp_get_c
* 功能描述 :   NTC 温度（°C），Beta 方程
*
*               第一步：由引脚电压反推 NTC 阻值
*                 V_adc = 3.3 × R / (10k + R)
*                 ⟹ R = 10k × V / (3.3 − V)
*               用 mV 直接算可以少一次浮点除法，但这里优先可读性。
*
*               第二步：Beta 方程
*                 1/T = 1/T25 + (1/B) × ln(R / R25)
*
*               ⚠️ 三个会让温度不可信的因素：
*                  1) V 接近 3300 mV 时 R 会爆掉（分母趋零）—— 这就是"探头开路"。
*                     这里用 3200 mV 做硬门限，超过直接判无效，不去算那个除法。
*                  2) Beta 是两点式模型，只在 25 °C 附近准；80 °C 附近误差可到 3–5 °C。
*                     做"过温保护"够用，做"精确测温"不够。
*                  3) NTC 是外置的，引线电阻和接触电阻会给 R 加一个固定偏移，
*                     表现为低温端读数偏高。要准确必须现场标定。
*
* 输入参数 :   which: 通道
* 输出参数 :   无
* 返回值   :   温度（°C），读数不合理时返回 DL_TEMP_INVALID_C
*********************************************************************************************/
F32 dl_temp_get_c
(
    IN  DL_TEMP_E   which
)
{
    U32     mv;
    F32     v_mv;
    F32     r_ntc;
    F32     inv_t;
    F32     temp_c;

    mv = dl_temp_get_raw_mv(which);

    /* 开路 / 未接探头：引脚被 10k 上拉顶到接近 3.3 V。
     * 3200 mV 对应 R ≈ 320 kΩ，已经远超任何正常工作温度下的阻值。 */
    if (mv >= 3200U)
    {
        return DL_TEMP_INVALID_C;
    }

    /* 短路 / 探头短到地：R ≈ 0 → 温度趋向无穷高，也不用算 */
    if (mv <= 20U)
    {
        return DL_TEMP_INVALID_C;
    }

    v_mv = (F32)mv;

    /* R = R_pullup × V / (Vref − V) */
    r_ntc = DL_TEMP_PULLUP_OHM * v_mv / ((F32)BSP_BOARD_ADC_VREF_MV - v_mv);

    /* 1/T = 1/T25 + ln(R/R25)/B */
    inv_t = (1.0F / DL_TEMP_T25_K) + (logf(r_ntc / DL_TEMP_NTC_R25_OHM) / DL_TEMP_NTC_B);

    temp_c = (1.0F / inv_t) - DL_TEMP_KELVIN_OFFSET;

    /* 量程外一律判无效，不把离谱的数字交给上层去做保护判断 */
    if ((temp_c < DL_TEMP_MIN_C) || (temp_c > DL_TEMP_MAX_C))
    {
        return DL_TEMP_INVALID_C;
    }

    return temp_c;
}

#ifdef _cplusplus
}
#endif
