#ifndef __BSP_ADC_H__
#define __BSP_ADC_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 采样通道分配（依据 当前原理图连接.json + DS12589 Rev6 Table 12 的 ADC 通道列）
 *
 * ── ADC1 · 注入组（与 PWM 同步，一个 PWM 周期采一次）──────────────────────────
 *   注入组硬件上限 4 个通道，这里用 3 个，顺序即转换顺序：
 *     rank1  SO_U_ADC   PA1   ADC1_IN2
 *     rank2  SO_V_ADC   PA3   ADC1_IN4
 *     rank3  SO_W_ADC   PA0   ADC1_IN1
 *   触发源：TIM1_TRGO，由 bsp_inverter 用 CH4(OC4REF) 在计数器顶点打一拍。
 *   为什么必须在顶点采：本板是**低边采样电阻**（RSU/RSV/RSW = 0.5 mΩ，接在 SHx 与 GND 之间），
 *   只有三相下桥同时导通时，三路采样电阻里才有完整相电流。中心对齐 PWM 的顶点 (CNT=ARR)
 *   正好是下桥全通的中心时刻，此刻采到的就是三相电流的真实值。
 *
 * ── ADC1 · 规则组（慢信号，连续转换 + DMA 循环搬运）────────────────────────────
 *     rank1  TEMP_MOS_ADC    PA2   ADC1_IN3
 *     rank2  TEMP_MOTOR_ADC  PB1   ADC1_IN12
 *     rank3  SAFE_ADC        PB0   ADC1_IN15
 *
 * ── ADC2 · 规则组（连续转换 + DMA 循环搬运）──────────────────────────────────
 *     rank1  BUS_ADC        PC4   ADC2_IN5
 *   ⚠️ PC4 在 G431 上**只有 ADC2 通道**（ADC1 没有映射），所以母线电压必须由 ADC2 采，
 *      不能和相电流塞进同一个 ADC。这也是本层要同时驱动 ADC1 和 ADC2 的原因。
 *
 * ── 参考电压 ────────────────────────────────────────────────────────────────
 *   VREF_CORE（J_MCU_R.8 = 核心板 P2-8 = MCU 的 VREF+）同时接到 DRV8353 的 VREF 脚，
 *   是电流放大器的基准。ADC 满量程 = 这个电压，不是想当然的 3.3 V。
 *   ⚠️ 板子上没有任何器件主动产生 VREF_CORE，它依赖核心板把 P2-8 内部接到 3V3。
 *      上电第一件事：万用表量 P2-8 对 GND，确认是 3.3 V 再谈电流标定。
 *      若实测不是 3300 mV，改 BSP_ADC_VREF_MV，否则所有电流/电压换算都会按比例偏。
 ********************************************************************************************/

/* ADC 满量程（mV）。当前按 3.3 V 写，实测 VREF_CORE 后必须复核。 */
#define BSP_ADC_VREF_MV                                                 (3300U)

/* 采样电阻（µΩ）。三只 PSF4NTEBL500F = 0.5 mΩ，四端开尔文接法。 */
#define BSP_ADC_SHUNT_UOHM                                              (500U)

/* DRV8353 电流放大器增益（V/V）。由 CSA Control 寄存器 CSA_GAIN 决定，
 * 复位默认 10b = 20 V/V。改增益必须同步改这里，否则电流标定全错。 */
#define BSP_ADC_CSA_GAIN                                                (20U)

/* 规则组通道数 */
#define BSP_ADC1_REG_NUM                                                (3U)
#define BSP_ADC2_REG_NUM                                                (1U)

/* 规则组缓冲区下标，方便上层按语义取值 */
#define BSP_ADC1_IDX_TEMP_MOS                                           (0U)
#define BSP_ADC1_IDX_TEMP_MOTOR                                         (1U)
#define BSP_ADC1_IDX_SAFE                                               (2U)
#define BSP_ADC2_IDX_BUS                                                (0U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_adc_init
* Descriptions  :   Configure ADC1 (regular + injected) and ADC2 (regular), plus their
*                   DMA channels. Nothing is started yet; call bsp_adc_start() after
*                   bsp_inverter_init() so the injection trigger source already exists.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_adc_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_adc_start
* Descriptions  :   Run offset calibration, then start the two regular DMA streams and
*                   arm the injected group.
*                   Must be called AFTER bsp_inverter_init(), because the injected group waits
*                   for TIM1_TRGO and TIM1 must already be counting.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_adc_start
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_adc_get_phase_raw
* Descriptions  :   Read the latest synchronized phase-current sample (raw 12-bit codes)
*                   The injected results are latched by hardware, so reading them is safe
*                   at any time and always returns the value captured at the last PWM top.
* Input         :
*                   p_u / p_v / p_w: destination pointers, must not be NULL
* Output        :
*                   *p_u / *p_v / *p_w: 0 ~ 4095
* Return        :
*                   TRUE  : at least one injected conversion has completed
*                   FALSE : no sample yet (TIM1 not running or ADC not armed)
*********************************************************************************************/
extern BOOL bsp_adc_get_phase_raw
(
    OUT U16     *p_u,
    OUT U16     *p_v,
    OUT U16     *p_w
);

/********************************************************************************************
* Function name :   bsp_adc_get_regular_raw
* Descriptions  :   Read one channel of the free-running regular groups (raw 12-bit code)
* Input         :
*                   adc_index: 1 = ADC1, 2 = ADC2
*                   channel_index: index inside that group (see BSP_ADCx_IDX_*)
* Output        :
*                   none
* Return        :
*                   0 ~ 4095, or 0 when the index is out of range
*********************************************************************************************/
extern U16 bsp_adc_get_regular_raw
(
    IN  U8      adc_index,
    IN  U8      channel_index
);

/********************************************************************************************
* Function name :   bsp_adc_raw_to_mv
* Descriptions  :   Convert a raw code into millivolts using BSP_ADC_VREF_MV
* Input         :
*                   raw: 0 ~ 4095
* Output        :
*                   none
* Return        :
*                   pin voltage in mV
*********************************************************************************************/
extern U32 bsp_adc_raw_to_mv
(
    IN  U16     raw
);

/********************************************************************************************
* Function name :   bsp_adc_phase_raw_to_ma
* Descriptions  :   Convert a phase-current raw code into milliamps.
*                   Sign convention: the DRV8353 CSA output is biased at VREF/2 when the
*                   shunt current is zero (CSA Control VREF_DIV = 1b, the reset default),
*                   so zero current reads mid-scale. Positive = current from SHx into GND.
*                   I(mA) = (V_pin - VREF/2) / (GAIN * R_shunt)
* Input         :
*                   raw: 0 ~ 4095
* Output        :
*                   none
* Return        :
*                   signed current in mA
*********************************************************************************************/
extern S32 bsp_adc_phase_raw_to_ma
(
    IN  U16     raw
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_ADC_H__ */
