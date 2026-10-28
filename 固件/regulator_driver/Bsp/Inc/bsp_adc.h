#ifndef __BSP_ADC_H__
#define __BSP_ADC_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 采样通道分配（依据 接口与采样核对.md + BOM.csv + DS12589 Rev6 Table 12）
 *
 * 本板一共 14 路模拟量，**分在两个 ADC 上**，这不是设计选择，是原理图的布线决定的：
 *   超容抽头 T1–T4 走 PA0–PA3（只有 ADC1），T5–T7 走 PA4–PA6（只有 ADC2）。
 *   想把 8 节抽头塞进一个 ADC 是不可能的。
 *
 * ── ADC1 · 规则组（8 通道，连续转换 + DMA 循环搬运）──────────────────────────
 *     idx 0  AD_T1       PA0   ADC1_IN1
 *     idx 1  AD_T2       PA1   ADC1_IN2
 *     idx 2  AD_T3       PA2   ADC1_IN3
 *     idx 3  AD_T4       PA3   ADC1_IN4
 *     idx 4  SC_SUM/T8   PB14  ADC1_IN5      ← 与 T8 是同一个读数，不是冗余通道
 *     idx 5  NTC_DUMP    PB12  ADC1_IN11
 *     idx 6  DRV_SO      PB1   ADC1_IN12     ← 同时是 COMP1_INP
 *     idx 7  BUS_ADC     PB0   ADC1_IN15
 *
 * ── ADC2 · 规则组（6 通道）──────────────────────────────────────────────────
 *     idx 0  AD_T5       PA4   ADC2_IN17
 *     idx 1  AD_T6       PA5   ADC2_IN13
 *     idx 2  AD_T7       PA6   ADC2_IN3
 *     idx 3  NTC_MAIN    PB11  ADC2_IN14
 *     idx 4  NTC_SC      PC4   ADC2_IN5
 *     idx 5  TEST_FB     PB15  ADC2_IN15
 *
 * ⚠️ PC4 在 G431 上只有 ADC2_IN5，没有 ADC1 映射；PA4 只有 ADC2_IN17（ADC1 的 IN17
 *    不在 PA4 上）。所以两个 ADC 必须同时驱动，缺一个就少一片采样。
 *
 * ── 采样时间为什么统一给 247.5 周期 ──────────────────────────────────────────
 *   慢通道的源阻抗都不低：
 *     T1–T7   68k ∥ 10k ∥ 47k ≈ 8.1 kΩ
 *     BUS     150k ∥ 10k      ≈ 9.4 kΩ
 *     NTC     10k 上拉         ≈ 10 kΩ
 *   G4 的采样电容约 5 pF，τ ≈ 40–50 ns；247.5 周期在 42.5 MHz 下是 5.82 µs，
 *   相当于 120 个 τ，足够稳。代价是一轮 8 通道要 ~49 µs，即约 20 kSps —— 对温度、
 *   抽头电压、母线电压这些量都远远够用（它们的时间常数是毫秒级）。
 *
 *   DRV_SO 是里面最快的信号（电流纹波在 20 kHz 上是三角波），20 kSps 只能看包络、
 *   看不清纹波形状。**但它本来就不是给保护用的** —— 过流快关断走的是
 *   DRV_SO → COMP1/COMP2 → TIM1 Break 的硬件路径（尚未实现，见 bsp_bridge.c 第 4 条）。
 *   要看纹波请用示波器，不要指望这个通道。
 *
 * ── 参考电压 ────────────────────────────────────────────────────────────────
 *   VREF_CORE（P2.8）同时是 MCU 的 VREF+ 和 DRV8161 的 CSAREF（板载 3.3 V），
 *   两者同源，所以电压换算与电流换算共用 BSP_BOARD_ADC_VREF_MV。
 ********************************************************************************************/

/* 规则组通道数。改这两个数必须同步改 bsp_adc.c 里的 Rank 赋值和 bsp_board.h 的通道宏。 */
#define BSP_ADC1_REG_NUM                                                (8U)
#define BSP_ADC2_REG_NUM                                                (6U)

/* 规则组缓冲区下标，方便上层按语义取值 */
#define BSP_ADC1_IDX_T1                                                 (0U)
#define BSP_ADC1_IDX_T2                                                 (1U)
#define BSP_ADC1_IDX_T3                                                 (2U)
#define BSP_ADC1_IDX_T4                                                 (3U)
#define BSP_ADC1_IDX_T8                                                 (4U)
#define BSP_ADC1_IDX_NTC_DUMP                                           (5U)
#define BSP_ADC1_IDX_DRV_SO                                             (6U)
#define BSP_ADC1_IDX_BUS                                                (7U)

#define BSP_ADC2_IDX_T5                                                 (0U)
#define BSP_ADC2_IDX_T6                                                 (1U)
#define BSP_ADC2_IDX_T7                                                 (2U)
#define BSP_ADC2_IDX_NTC_MAIN                                           (3U)
#define BSP_ADC2_IDX_NTC_SC                                             (4U)
#define BSP_ADC2_IDX_TEST_FB                                            (5U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_adc_init
* Descriptions  :   Configure ADC1 (8 channels) and ADC2 (6 channels) as free-running
*                   scan sequences with circular DMA, plus their analog input pins.
*                   Nothing is started yet; call bsp_adc_start().
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
* Descriptions  :   Run offset calibration on both ADCs, then start the two DMA streams.
*                   Must be called after bsp_adc_init().
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
* Function name :   bsp_adc_get_raw
* Descriptions  :   Read one channel of the free-running groups (raw 12-bit code).
*                   DMA moves the samples in the background, so this is just a memory read.
*                   The two ADCs are not sampled in lockstep: a value from ADC1 and one from
*                   ADC2 may come from slightly different instants. For temperature, tap
*                   voltages and bus voltage that is irrelevant.
* Input         :
*                   adc_index: 1 = ADC1, 2 = ADC2
*                   channel_index: index inside that group (see BSP_ADCx_IDX_*)
* Output        :
*                   none
* Return        :
*                   0 ~ 4095, or 0 when the index is out of range
*********************************************************************************************/
extern U16 bsp_adc_get_raw
(
    IN  U8      adc_index,
    IN  U8      channel_index
);

/********************************************************************************************
* Function name :   bsp_adc_raw_to_mv
* Descriptions  :   Convert a raw code into millivolts using BSP_BOARD_ADC_VREF_MV
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
* Function name :   bsp_adc_get_so_ma
* Descriptions  :   Convert the DRV_SO channel into a signed bus/inductor current.
*                   Chain: pin mV -> subtract VREF/2 -> divide by (GCSA x Rshunt).
*                   With GCSA = 20 V/V and Rshunt = 0.5 mOhm the sensitivity is 10 mV/A,
*                   so the full +/-1.65 V swing maps to roughly +/-165 A on paper. The
*                   datasheet limits the SO output to 0.25 V .. VCSAREF-0.25 V, so the
*                   usable range is closer to +/-140 A, and the input offset alone is
*                   worth about +/-3.9 A. Treat this as a measurement, not a trip point.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   signed current in mA, positive = discharging the supercapacitor
*********************************************************************************************/
extern S32 bsp_adc_get_so_ma
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_ADC_H__ */
