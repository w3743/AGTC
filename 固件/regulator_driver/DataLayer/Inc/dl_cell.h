#ifndef __DL_CELL_H__
#define __DL_CELL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 超容 8 节累积抽头 —— 选通采样、换算与它的前提
 *
 * ── 硬件（2026-10-30 改版后）─────────────────────────────────────────────────
 *   T0 是超容负极（SC−），T1–T8 依次往上抽头，T8 = SC+ 顶端，共 9 根进 J_SC_TAPS。
 *   每节的电阻网络现在只有两只：
 *       Tj ── R_Tj_HI 68 kΩ ──┬── AD_Tj ── U_CELL_MUX(开关) ── SC_ADC_x ── ADC
 *                             └── R_Tj_LO   10 kΩ ── GND_SYS
 *   ADC 节点上另有 R_ADC_x_PD 10 kΩ 下拉（与 R_Tj_LO 并联）和 1 nF 滤波电容。
 *
 *   于是选通后的分压比：
 *       K = 1 + 68k / (10k ∥ 10k) = 1 + 13.6 = 14.6
 *   **8 路完全一样**，T8 也不例外。
 *
 *   ⚠️ 旧的 47k→TEST 支路已随 TEST 网络一起删除，所以不再有 −c×Vtest 修正项。
 *      不要再按 7.8 / 9.2468 换算 —— 那会把读数放大 1.87 倍 / 1.58 倍。
 *
 * ── 采样方式 ────────────────────────────────────────────────────────────────
 *   两颗 TMUX1511 各带 4 路开关，D 端分别并到 SC_ADC_A（ADC1）与 SC_ADC_B（ADC2）。
 *   所以**同一时刻只能测一节**，8 节靠轮询。dl_cell_scan() 负责整轮扫描并刷新缓存，
 *   dl_cell_get_tap_mv() 只读缓存 —— 否则每读一节就要等一次模拟建立，调用者会被拖死。
 *
 *   扫描周期建议 100 ms 级：抽头电压跟着超容 SOC 走，时间常数是秒级。
 *
 *   ⚠️ 选通后模拟侧需要建立时间：68k ∥ 5k 对 1 nF，τ ≈ 4.7 µs，5τ ≈ 24 µs。
 *      本层用 DL_CELL_SETTLE_LOOPS 的空转等待（远大于 24 µs），宁可慢也不要读错。
 *
 * ── ⚠️ 一个原理图层面的缺口，本层绕不过去 ──────────────────────────────────
 *   所有 tap ADC 都相对 **GND_SYS** 读数，而超容负极（SC−）不是 GND_SYS：
 *   中间隔着 R_SC_SHUNT 0.5 mΩ，而且原理图准备文档明确写着
 *   「不得默认 SC− 与系统地等电位」。
 *
 *   于是：
 *     第 1 节电压  = Vtap1 − V_SCminus      ← 需要 V_SCminus，**当前原理图没有这个通道**
 *     第 2–7 节    = Vtap[j] − Vtap[j−1]    ← 差分，SC− 抵消，**可以直接算**
 *     第 8 节      = Vtap8 − Vtap7          ← 同上，可以直接算
 *     超容总压     = Vtap8 − V_SCminus      ← 同样缺 V_SCminus
 *
 *   这就是为什么 dl_cell_get_cell_mv() 要调用者自己把 sc_minus_mv 传进来。
 *   传 0 意味着假设 SC− 与系统地等电位，在有电流时**不成立**
 *   （0.5 mΩ × 100 A = 50 mV，已经超过单节工作窗余量）。
 ********************************************************************************************/

#define DL_CELL_NUM                                                     (8U)

/* 选通后的分压比：68k / (10k ∥ 10k) + 1 = 14.6（8 路统一） */
#define DL_CELL_TAP_K                                                   (14.6F)

/* 每路选通后的模拟建立等待。用空转循环而不是 HAL_Delay：
 * 这里只需要几十微秒，HAL_Delay 最小 1 ms，会把一次扫描拖到 8 ms 以上。
 * 170 MHz 下每次循环约 3–4 个周期，12000 次约 250–280 µs，是 τ 的 50 倍以上。 */
#define DL_CELL_SETTLE_LOOPS                                            (12000U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_cell_scan
* Descriptions  :   Walk all eight taps: close one switch, wait for the divider to settle,
*                   read the shared ADC node, open the switch. Results are cached.
*                   Blocking, roughly 8 x 280 us = 2.3 ms per round.
*                   Call it from the slow task (100 ms class), not from the control loop.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_cell_scan
(
    VOID
);

/********************************************************************************************
* Function name :   dl_cell_is_scanned
* Descriptions  :   TRUE once dl_cell_scan() has completed at least one full round
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE / FALSE
*********************************************************************************************/
extern BOOL dl_cell_is_scanned
(
    VOID
);

/********************************************************************************************
* Function name :   dl_cell_get_tap_mv
* Descriptions  :   Absolute voltage of tap j (1..8) in mV, from the last scan.
*                   Referenced to GND_SYS, not to SC-.
*                   Returns 0 if dl_cell_scan() has never run.
* Input         :
*                   index: 1..8
* Output        :
*                   none
* Return        :
*                   tap voltage in mV, 0 when the index is out of range or no scan yet
*********************************************************************************************/
extern U32 dl_cell_get_tap_mv
(
    IN  U8      index
);

/********************************************************************************************
* Function name :   dl_cell_get_cell_mv
* Descriptions  :   Voltage across cell j (1..8) in mV.
*                   Cells 2..8 are pure differences of two taps and are therefore independent
*                   of the SC- offset. Cell 1 needs the SC- potential, which has no sampling
*                   path in the current schematic and must be supplied by the caller.
* Input         :
*                   index      : 1..8
*                   sc_minus_mv: potential of SC- above GND_SYS, in mV (pass 0 if unknown,
*                                but then cell 1 carries an unknown offset)
* Output        :
*                   none
* Return        :
*                   signed cell voltage in mV, 0 when the index is out of range
*********************************************************************************************/
extern S32 dl_cell_get_cell_mv
(
    IN  U8      index,
    IN  U32     sc_minus_mv
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_CELL_H__ */
