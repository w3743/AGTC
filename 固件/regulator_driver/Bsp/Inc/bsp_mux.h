#ifndef __BSP_MUX_H__
#define __BSP_MUX_H__

#include "type.h"
#include "bsp_board.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 超容抽头选通 —— 2 颗 TMUX1511（4 路常开开关 ×2）
 *
 * 2026-10-30 改版把“8 路抽头各自直连一个 ADC 脚”换成了“8 路抽头 → 2 颗模拟开关 → 2 路 ADC”。
 * 省下来的引脚给 MUX_SEL，代价是**同一时刻只能测一节**，必须轮询。
 *
 *   U_CELL_MUX1  SEL1..SEL4  →  AD_T1..AD_T4  →  公共 D 端 = SC_ADC_A（PB14 / ADC1_IN5）
 *   U_CELL_MUX2  SEL1..SEL4  →  AD_T5..AD_T8  →  公共 D 端 = SC_ADC_B（PB15 / ADC2_IN15）
 *
 * ⚠️ 必须 one-hot。四路开关的 D 端是连在一起的，同时开两路等于把两节超容经开关直连，
 *    轻则读数全错，重则把开关或抽头电阻烧掉。bsp_mux_select() 内部先全关再开一路，
 *    但仍然要求调用者不要并发调用（本工程只有主循环一处会调）。
 *
 * ⚠️ 上电/复位态：8 个 SEL 全低 = 全断。抽头每路仍挂着自己的 68k/10k 分压，
 *    ADC 节点被 10k 下拉到 0 V —— 读到的 0 是“没选通”，不是“没电”。
 ********************************************************************************************/

#define BSP_MUX_CH_NONE                                                 (0U)
#define BSP_MUX_CH_NUM                                                  (8U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_mux_init
* Descriptions  :   Configure the eight SEL pins as push-pull outputs and leave all switches
*                   open. Safe to call before anything else.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_mux_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_mux_select
* Descriptions  :   Close exactly one switch. channel = 0 opens all of them.
*                   The previous channel is opened first, so there is never a moment with two
*                   switches closed (a few hundred ns of both-open is harmless: the ADC node
*                   just holds its 10k pulldown).
* Input         :
*                   channel: 0 = all off, 1..8 = tap T1..T8
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_mux_select
(
    IN  U8      channel
);

/********************************************************************************************
* Function name :   bsp_mux_get_selected
* Descriptions  :   Channel currently closed (0 = none)
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   0..8
*********************************************************************************************/
extern U8 bsp_mux_get_selected
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_MUX_H__ */
