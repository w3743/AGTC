/************************************包含头文件*************************************************/
#include "type.h"
#include "bsp_adc.h"
#include "dl_cell.h"
#include "dl_bus.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************静态函数声明************************************************/
static U32 _dl_cell_get_tap_pin_mv
(
    IN  U8      index
);


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _dl_cell_get_tap_pin_mv
* 功能描述 :   按 tap 序号取对应的 ADC 引脚电压（mV）
*
*               T1–T4 在 ADC1（idx 0–3），T5–T7 在 ADC2（idx 0–2），T8 在 ADC1（idx 4）。
*               这个"跨两个 ADC"的分布是原理图布线定的，软件改不了，
*               所以这张映射表必须留在这里、不能省。
*
* 输入参数 :   index: 1 ~ 8
* 输出参数 :   无
* 返回值   :   引脚电压（mV），越界返回 0
*********************************************************************************************/
static U32 _dl_cell_get_tap_pin_mv
(
    IN  U8      index
)
{
    U16 raw = 0U;

    switch (index)
    {
        case 1U: raw = bsp_adc_get_raw(1U, BSP_ADC1_IDX_T1); break;
        case 2U: raw = bsp_adc_get_raw(1U, BSP_ADC1_IDX_T2); break;
        case 3U: raw = bsp_adc_get_raw(1U, BSP_ADC1_IDX_T3); break;
        case 4U: raw = bsp_adc_get_raw(1U, BSP_ADC1_IDX_T4); break;
        case 5U: raw = bsp_adc_get_raw(2U, BSP_ADC2_IDX_T5); break;
        case 6U: raw = bsp_adc_get_raw(2U, BSP_ADC2_IDX_T6); break;
        case 7U: raw = bsp_adc_get_raw(2U, BSP_ADC2_IDX_T7); break;
        case 8U: raw = bsp_adc_get_raw(1U, BSP_ADC1_IDX_T8); break;
        default: return 0U;
    }

    return bsp_adc_raw_to_mv(raw);
}

/********************************************************************************************
* 函数名称 :   dl_cell_get_test_mv
* 功能描述 :   TEST 网络电压（mV）
*
*               TEST_FB 是 10k/10k 分压 → Vtest = 2 × V(TEST_FB)。
*               正常情况下 TEST 被 PB10 推挽拉低，这里应该读到接近 0。
*               **如果这里读到几百毫伏以上，先别信任何 tap 读数** ——
*               说明 TEST 没被拉低（PB10 没初始化 / 焊点开路 / 有人在做诊断），
*               此时每一节都会被 c × Vtest 拉偏。把本函数的值打出来是最快的排查手段。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TEST 电压（mV）
*********************************************************************************************/
U32 dl_cell_get_test_mv
(
    VOID
)
{
    U16 raw;
    U32 pin_mv;

    raw    = bsp_adc_get_raw(2U, BSP_ADC2_IDX_TEST_FB);
    pin_mv = bsp_adc_raw_to_mv(raw);

    return (U32)(((F32)pin_mv * DL_CELL_TEST_FB_RATIO) + 0.5F);
}

/********************************************************************************************
* 函数名称 :   dl_cell_get_tap_mv
* 功能描述 :   第 j 个抽头的绝对电压（mV），已按 TEST 网络修正
*
*               公式：Vtap = K × Vadc − c × Vtest
*                     K = 1 + 68/10 + 68/47 = 9.2468
*                     c = 68/47             = 1.4468
*
*               推导（以 T1 为例，68k 从 T1 到 ADC，10k 从 ADC 到地，47k 从 ADC 到 TEST）：
*                 节点 KCL：(Vtap − Vadc)/68 = Vadc/10 + (Vadc − Vtest)/47
*                 两边乘 68×10×47 整理即得上式。
*
*               T8 没有 47k 支路，是一条纯 68k/10k 分压，所以直接走 7.8 倍，
*               不需要减 Vtest 项（传 c=0 的效果一样）。
*
* 输入参数 :   index: 1 ~ 8
* 输出参数 :   无
* 返回值   :   抽头电压（mV，相对 GND_SYS），越界返回 0
*********************************************************************************************/
U32 dl_cell_get_tap_mv
(
    IN  U8      index
)
{
    F32 pin_mv;
    F32 tap_mv;
    F32 test_mv;

    if ((index < 1U) || (index > DL_CELL_NUM))
    {
        return 0U;
    }

    pin_mv = (F32)_dl_cell_get_tap_pin_mv(index);

    /* T8 是纯分压，没有 TEST 支路 */
    if (index == DL_CELL_NUM)
    {
        tap_mv = pin_mv * DL_CELL_T8_RATIO;
        return (U32)(tap_mv + 0.5F);
    }

    test_mv = (F32)dl_cell_get_test_mv();

    tap_mv = (DL_CELL_TAP_K * pin_mv) - (DL_CELL_TAP_C * test_mv);

    /* 理论上 tap ≥ 0，但 TEST 被拉高时减出来的值可能为负。
     * 这里钳到 0 而不是回绕 —— 回绕会变成一个巨大的正数，比负数更误导人。 */
    if (tap_mv < 0.0F)
    {
        tap_mv = 0.0F;
    }

    return (U32)(tap_mv + 0.5F);
}

/********************************************************************************************
* 函数名称 :   dl_cell_get_cell_mv
* 功能描述 :   第 j 节单体电压（mV，带符号）
*
*               第 2–8 节：直接两个抽头相减，SC− 电位互相抵消，**结果可信**。
*               第 1 节  ：需要 SC− 相对 GND_SYS 的电位，而当前原理图没有这个采样通道。
*                          调用者必须自己把 sc_minus_mv 传进来；
*                          传 0 意味着假设 SC− 与系统地等电位，这在有电流时**不成立**
*                          （0.5 mΩ × 100 A = 50 mV，已经超过单节工作窗余量）。
*
*               ⚠️ 第 8 节 = Vtap8 − Vtap7，两端各带一次分压误差，
*                  而 T8 的比值（7.8）与 T1–T7 的 K（9.2468）不同，
*                  所以这两路的增益误差**不会互相抵消**。做单体过压判断时要单独评估。
*
* 输入参数 :   index      : 1 ~ 8
*               sc_minus_mv: SC− 相对 GND_SYS 的电位（mV）
* 输出参数 :   无
* 返回值   :   带符号单体电压（mV），越界返回 0
*********************************************************************************************/
S32 dl_cell_get_cell_mv
(
    IN  U8      index,
    IN  U32     sc_minus_mv
)
{
    U32 tap_now;
    U32 tap_prev;

    if ((index < 1U) || (index > DL_CELL_NUM))
    {
        return 0;
    }

    tap_now = dl_cell_get_tap_mv(index);

    /* 第 1 节：下限是 SC−，不是 GND，也不是 0 */
    if (index == 1U)
    {
        return (S32)tap_now - (S32)sc_minus_mv;
    }

    tap_prev = dl_cell_get_tap_mv((U8)(index - 1U));

    return (S32)tap_now - (S32)tap_prev;
}

#ifdef _cplusplus
}
#endif
