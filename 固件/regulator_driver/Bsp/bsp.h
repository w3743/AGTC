#ifndef __BSP_H__
#define __BSP_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 板级支持包 —— 稳压模块的对外总入口
 *
 * 分三层，边界不许跨：
 *     BSP  —— 只认寄存器和引脚，认识 HAL，认识芯片手册
 *     DL   —— 只认物理量（伏、安、摄氏度），不 include 任何 HAL 头文件
 *     BL   —— 只认策略（该不该放电、该泄多少），只 include DL
 *
 * 本文件是 BSP 对外的唯一门面：上层只 include "bsp.h"，不用关心下面有八个子模块。
 ********************************************************************************************/

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_get_tick
* Descriptions  :   Millisecond tick since HAL_Init().
*                   Wrapped so that DL/BL never need to include stm32g4xx_hal.h.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   milliseconds
*********************************************************************************************/
extern U32 bsp_get_tick
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_init
* Descriptions  :   Board bring-up in a fixed, safety-ordered sequence.
*                   Leaves the bridge switched off (MOE=0), the gate driver inhibited
*                   (PERMIT low), the BUS P-MOS open and the dump PWM at 0%.
*                   CAN is configured but not started.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS: every block configured
*                   TYPE_RET_FAIL   : at least one block failed to configure
*********************************************************************************************/
extern TYPE_RET_E bsp_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_power_stage_enable
* Descriptions  :   Energise / de-energise the whole power path in a safe order.
*
*                   ENABLE:  midpoint duty -> close BUS P-MOS -> release PERMIT -> set MOE
*                   DISABLE: clear MOE -> assert PERMIT -> midpoint duty -> wait for the
*                            inductor current to decay -> open BUS P-MOS
*
*                   See bsp.c for why each step is where it is.
* Input         :
*                   enable: TRUE = energise, FALSE = shut down
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS: done
*                   TYPE_RET_TIMEOUT: current never decayed, P-MOS left CLOSED on purpose
*********************************************************************************************/
extern TYPE_RET_E bsp_power_stage_enable
(
    IN  BOOL    enable
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_H__ */
