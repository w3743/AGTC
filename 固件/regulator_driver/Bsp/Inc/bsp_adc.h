#ifndef __BSP_ADC_H__
#define __BSP_ADC_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 采样通道分配（依据 2026-10-30 最终逐针连接.csv + DS12589 Table 12）
 *
 * 本板一共 7 路模拟量，**分在两个 ADC 上**，这不是设计选择，是引脚决定的：
 *   PB14 只有 ADC1_IN5，PB15 只有 ADC2_IN15 —— 8 节抽头被拆成两半。
 *
 * ── ADC1 · 规则组（4 通道，连续转换 + DMA 循环搬运）──────────────────────────
 *     idx 0  BUS_ADC     PB0   ADC1_IN15     母线电压（150k/10k 分压）
 *     idx 1  IMON        PB1   ADC1_IN12     DRV8161 CSA 输出（兼 COMP1_INP）
 *     idx 2  NTC_DUMP    PA3   ADC1_IN4      泄放电阻温度
 *     idx 3  SC_ADC_A    PB14  ADC1_IN5      MUX1 公共端（T1–T4，一次只开一路）
 *
 * ── ADC2 · 规则组（3 通道）──────────────────────────────────────────────────
 *     idx 0  NTC_MAIN    PB11  ADC2_IN14
 *     idx 1  NTC_SC      PC4   ADC2_IN5
 *     idx 2  SC_ADC_B    PB15  ADC2_IN15     MUX2 公共端（T5–T8）
 *
 * ⚠️ 2026-10-30 之前是 ADC1 八通道 / ADC2 六通道，抽头直连。改版后抽头全部经 TMUX1511
 *    选通到 SC_ADC_A/B 两个节点，**每次只有一节能被采到**，扫描逻辑在 dl_cell.c，
 *    选通在 bsp_mux.c。别再按旧下标找 T1…T8。
 *
 * ── 采样时间为什么统一给 247.5 周期 ──────────────────────────────────────────
 *   慢通道的源阻抗都不低：
 *     抽头（选通后） 68k ∥ 10k ∥ 10k ≈ 4.66 kΩ
 *     BUS           150k ∥ 10k      ≈ 9.4 kΩ
 *     NTC           10k 上拉         ≈ 10 kΩ
 *   G4 的采样电容约 5 pF，τ ≈ 40–50 ns；247.5 周期在 42.5 MHz 下是 5.82 µs，
 *   相当于 120 个 τ，足够稳。代价是一轮 4 通道约 23 µs，约 43 kSps —— 对温度、
 *   抽头电压、母线电压这些量远远够用（它们的时间常数是毫秒级）。
 *
 *   IMON 是里面最快的信号（20 kHz 三角波），43 kSps 只能看包络、看不清纹波形状。
 *   它本来也不是给保护用的 —— 过流快关断走 DRV_FAULT_N 硬件链路。
 *
 * ── 参考电压 ────────────────────────────────────────────────────────────────
 *   VREF_CORE（P2.8）同时是 MCU 的 VREF+ 和 DRV8161 的 CSAREF（板载 3.3 V），
 *   两者同源，所以电压换算与电流换算共用 BSP_BOARD_ADC_VREF_MV。
 ********************************************************************************************/

/* 规则组通道数。改这两个数必须同步改 bsp_adc.c 里的 Rank 赋值和 bsp_adc.h 的下标宏。 */
#define BSP_ADC1_REG_NUM                                                (4U)
#define BSP_ADC2_REG_NUM                                                (3U)

/* 规则组缓冲区下标，方便上层按语义取值 */
#define BSP_ADC1_IDX_BUS                                                (0U)
#define BSP_ADC1_IDX_IMON                                               (1U)
#define BSP_ADC1_IDX_NTC_DUMP                                           (2U)
#define BSP_ADC1_IDX_SC_A                                               (3U)

#define BSP_ADC2_IDX_NTC_MAIN                                           (0U)
#define BSP_ADC2_IDX_NTC_SC                                             (1U)
#define BSP_ADC2_IDX_SC_B                                               (2U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_adc_init
* Descriptions  :   Configure ADC1 (4 channels) and ADC2 (3 channels) as free-running
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
*                   ADC2 may come from slightly different instants.
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
* Descriptions  :   Convert the IMON channel into a signed bus/inductor current.
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
