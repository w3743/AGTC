/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp.h"
#include "dl.h"
#include "dl_motor.h"
#include "dl_sense.h"
#include "dl_can.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   dl_init
* 功能描述 :   数据层初始化
*
*              顺序有依赖，不能换：
*                1. dl_motor_init() —— 先把功率级确认在关断状态。哪怕前面 bsp_init()
*                   已经保证过，这里再确认一次的成本是几次寄存器写，收益是"DL 自己
*                   保证了自己工作时的初始条件"，不依赖调用者的行为。
*                2. dl_sense_init() —— 清采样缓存。放第二步是因为它只清内存，
*                   不影响硬件，而第 1 步会动硬件。
*
*              **不在这里**做的：dl_can_init()。
*                注册 CAN 接收回调属于"要不要接入总线"这件事的一部分，和
*                bsp_can_start() 是一对，统一由 bl_init() 负责。
*                这样 DL 层的初始化只关心"测量链路"，不掺通信的启停决策。
*                （regulator_driver 工程是同样的分工，两边可以对照着看。）
*
*              也不调 dl_sense_update()：此刻 bl 还没起来，没有任何人在等这些数，
*              调了也只是把一帧上电瞬间的乱数塞进缓存。第一次有效采样由 BL 的
*              第一个控制节拍取。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E dl_init
(
    VOID
)
{
    TYPE_RET_E result = TYPE_RET_SUCCESS;

    if (dl_motor_init() != TYPE_RET_SUCCESS)
    {
        result = TYPE_RET_FAIL;
    }

    if (dl_sense_init() != TYPE_RET_SUCCESS)
    {
        result = TYPE_RET_FAIL;
    }

    return result;
}

#ifdef _cplusplus
}
#endif
