#ifndef __DL_CELL_H__
#define __DL_CELL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 超容 8 节累积抽头 —— 换算与它的前提
 *
 * ── 硬件 ────────────────────────────────────────────────────────────────────
 *   T0 是超容负极（SC−），T1–T8 依次往上抽头，T8 = SC+ 顶端，共 9 根进 J_SC_TAPS。
 *   T1–T7 每路的候选网络（BOM 已逐只核对）：
 *       Tj ── R_Tj_HI 68 kΩ ──┬── ADCj
 *                             ├── R_Tj_LO   10 kΩ ── GND_SYS
 *                             └── R_Tj_TEST 47 kΩ ── TEST
 *   T8 只有 R_T8_HI 68 kΩ + R_T8_LO 10 kΩ，没有 TEST 支路。
 *
 *   解得：Vtap[j] = K × Vadc[j] − c × Vtest
 *         K = 1 + 68/10 + 68/47 = 9.246808…
 *         c = 68/47             = 1.446808…
 *   T8：Vtap[8] = 7.8 × Vadc[8]，c = 0。
 *
 * ── 为什么公式里要减 c × Vtest，而不是当 TEST 是 0 ────────────────────────
 *   TEST 平时由 PB10 推挽拉低，理论上 Vtest = 0，第二项消失。
 *   但一旦 PB10 没配好、焊点虚了、或者有人在做诊断把它拉高，
 *   这一项就是实打实的误差：TEST 抬高 1 V，每节读数会偏 1.45 V。
 *   所以本层**始终用实测的 TEST_FB 反推出 Vtest**（TEST_FB 是 10k/10k 分压，×2），
 *   而不是假设它为零。多读一个通道，换掉一整类"看起来正常其实全偏"的故障。
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
 *   这就是为什么 dl_cell_get_cell_mv() 要调用者自己把 sc_minus_mv 传进来：
 *   本层不编造一个它测不到的数。真值是 0、是 I×R_shunt、还是别的补偿值，
 *   要等硬件补上 SC− 采样或给出可行的估算方法之后才能定。
 *   文档里也写着「SC−偏移的采集或基于电流/分流器的补偿方法及其误差预算仍待闭合」。
 *
 *   传 0 的后果：第 1 节读数会等于 Vtap1 的全值（可能偏几十毫伏到几百毫伏，
 *   取决于当前电流），总压同理。做单体过压判断时不能接受这种偏差。
 ********************************************************************************************/

#define DL_CELL_NUM                                                     (8U)

/* T1–T7 的换算系数（由 68k / 10k / 47k 三个电阻定出来，不是经验值） */
#define DL_CELL_TAP_K                                                   (9.246808F)
#define DL_CELL_TAP_C                                                   (1.446808F)

/* T8 的换算比：68k / 10k 简单分压 */
#define DL_CELL_T8_RATIO                                                (7.8F)

/* TEST_FB 的分压比：R_TEST_HI 10k / R_TEST_LO 10k → Vtest = 2 × V(TEST_FB) */
#define DL_CELL_TEST_FB_RATIO                                           (2.0F)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_cell_get_tap_mv
* Descriptions  :   Absolute voltage of tap j (1..8) in mV, already corrected for the TEST
*                   network. This is referenced to GND_SYS, not to SC-.
* Input         :
*                   index: 1..8
* Output        :
*                   none
* Return        :
*                   tap voltage in mV, 0 when the index is out of range
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
*                   index     : 1..8
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

/********************************************************************************************
* Function name :   dl_cell_get_test_mv
* Descriptions  :   Reconstructed TEST net voltage in mV (TEST_FB x 2).
*                   Useful as a sanity check before trusting any tap reading.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TEST voltage in mV
*********************************************************************************************/
extern U32 dl_cell_get_test_mv
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_CELL_H__ */
