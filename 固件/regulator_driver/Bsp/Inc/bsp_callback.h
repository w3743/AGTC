#ifndef __BSP_CALLBACK_H__
#define __BSP_CALLBACK_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/************************************Type****************************************************/
/* FDCAN 接收回调原型：由上层（DL）注册，BSP 在中断上下文里调用。
 * 约束：回调运行在中断里，不得调用阻塞 API，不得做浮点长运算。 */
typedef VOID (* BSP_CALLBACK_CAN_RX_FUNC)(U32 id, BOOL is_ext, const U8 *p_data, U8 len);

/************************************Fcunction***********************************************/
/********************************************************************************************
* Function name :   bsp_callback_can_register
* Descriptions  :   Register CAN rx callback
* Input         :
*                   func: callback, NULL to unregister
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_callback_can_register
(
    IN  BSP_CALLBACK_CAN_RX_FUNC    func
);

/********************************************************************************************
* Function name :   bsp_callback_can_rx
* Descriptions  :   Forward a received frame to the registered callback
*                   Called by BSP CAN driver inside the RX interrupt only
* Input         :
*                   id    : frame id
*                   is_ext: TRUE for extended id
*                   p_data: payload
*                   len   : payload length
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_callback_can_rx
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
);

#ifdef _cplusplus
}
#endif

#endif
