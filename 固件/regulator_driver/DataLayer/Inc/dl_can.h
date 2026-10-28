#ifndef __DL_CAN_H__
#define __DL_CAN_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * CAN 数据层 —— 成组参数下发、状态回报
 *
 * 分工（与整个仓库的约定一致）：
 *   CAN 只负责"参数下发 / 选策略 / 回报状态 / 回报拒绝原因"，
 *   **不承担快速保护**，也不接收运行时代码。
 *   真正的过流快关断必须由 DRV8161 的 VDS OCP 和（将来的）COMP→TIM1 Break 完成。
 *
 * 本层做的事：
 *   · 把 BSP 的原始帧拆成有语义的字段
 *   · 参数组做版本 / CRC / 单位 / 范围校验，校验通过才交给 BL
 *   · 状态帧的打包
 *
 * 本层不做的事：
 *   · 不决定"接受还是拒绝"（那是 BL 的判断，本层只把拒绝原因带上去）
 *   · 不写 Flash（参数持久化由 BL 在收到明确保存请求时发起）
 *
 * ⚠️ 报文 ID 与字段布局**尚未定义**。
 *    整个仓库还没有一份冻结的 CAN 协议表，随便编一套 ID 只会和以后的真协议打架。
 *    所以下面这些宏先留空位，等协议定了再填 —— 宁可编译期少几个常量，
 *    也不要留一堆"看起来能用其实是猜的"定义。
 ********************************************************************************************/

/* ⚠️ 以下 ID 全部是占位符，等整车 CAN 协议表定稿后替换。 */
#define DL_CAN_ID_STATUS                                                (0x180U)    /* 占位：状态回报 */
#define DL_CAN_ID_CMD                                                   (0x200U)    /* 占位：参数/策略下发 */
#define DL_CAN_ID_FAULT                                                 (0x280U)    /* 占位：故障与拒绝原因 */

/************************************Type****************************************************/
/* 一帧原始报文（已从 BSP 回调里拿出来的形态） */
typedef struct _DL_CAN_FRAME_T_
{
    U32     id;
    BOOL    is_ext;
    U8      data[8];
    U8      len;
}DL_CAN_FRAME_T;

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_can_init
* Descriptions  :   Hook the BSP CAN receive callback and clear the pending-frame slot.
*                   The peripheral itself is started by the caller (bsp_can_start), because
*                   when to join the bus is a system-level decision.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_can_init
(
    VOID
);

/********************************************************************************************
* Function name :   dl_can_poll
* Descriptions  :   Take one pending frame out of the interrupt hand-off slot, if any.
*                   Call from the main loop, never from an interrupt.
* Input         :
*                   none
* Output        :
*                   p_frame: filled when a frame was pending, untouched otherwise
* Return        :
*                   TRUE = a frame was returned, FALSE = nothing pending
*********************************************************************************************/
extern BOOL dl_can_poll
(
    OUT DL_CAN_FRAME_T  *p_frame
);

/********************************************************************************************
* Function name :   dl_can_send_status
* Descriptions  :   Pack and queue the periodic status frame.
*                   Layout is a placeholder until the vehicle CAN table is frozen.
* Input         :
*                   bus_mv    : bus voltage
*                   sc_mv     : supercapacitor stack voltage
*                   current_ma: signed supercapacitor current
*                   temp_main_c: main stage temperature
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_can_send_status
(
    IN  U32     bus_mv,
    IN  U32     sc_mv,
    IN  S32     current_ma,
    IN  F32     temp_main_c
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_CAN_H__ */
