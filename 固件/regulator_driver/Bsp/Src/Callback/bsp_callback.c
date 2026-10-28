/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_callback.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_CALLBACK_MODULE_T_
{
    BSP_CALLBACK_CAN_RX_FUNC    can_rx;                                 // CAN 接收回调
}BSP_CALLBACK_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_CALLBACK_MODULE_T * _bsp_callback_get_module
(
    VOID
);


/************************************静态变量定义************************************************/
static BSP_CALLBACK_MODULE_T s_bsp_callback_module =
{
    NULL
};


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_callback_get_module
* 功能描述 :   获取回调模块实例指针
* 输入参数 :
*              无
* 输出参数 :
*              无
* 返回值   :
*              回调模块实例指针
*********************************************************************************************/
static BSP_CALLBACK_MODULE_T * _bsp_callback_get_module
(
    VOID
)
{
    return &s_bsp_callback_module;
}

/********************************************************************************************
* 函数名称 :   bsp_callback_can_register
* 功能描述 :   注册 CAN 接收回调
*              只有已启动的 CAN 才会进中断，注册前未能接收的报文会被硬件丢弃
* 输入参数 :
*              func: 回调函数，传 NULL 表示注销
* 输出参数 :
*              无
* 返回值   :
*              无
*********************************************************************************************/
VOID bsp_callback_can_register
(
    IN  BSP_CALLBACK_CAN_RX_FUNC    func
)
{
    _bsp_callback_get_module()->can_rx = func;
}

/********************************************************************************************
* 函数名称 :   bsp_callback_can_rx
* 功能描述 :   把收到的报文转交给已注册的回调
*              未注册时直接丢弃，不缓存、不重试
* 输入参数 :
*              id    : 标识符
*              is_ext: 是否扩展帧
*              p_data: 数据
*              len   : 长度
* 输出参数 :
*              无
* 返回值   :
*              无
*********************************************************************************************/
VOID bsp_callback_can_rx
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
)
{
    BSP_CALLBACK_MODULE_T *p_mod = _bsp_callback_get_module();

    if (p_mod->can_rx != NULL)
    {
        p_mod->can_rx(id, is_ext, p_data, len);
    }
}

#ifdef _cplusplus
}
#endif
