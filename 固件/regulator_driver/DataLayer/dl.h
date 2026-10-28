#ifndef __DL_H__
#define __DL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 数据层（Data Layer）总入口 —— 稳压模块
 *
 * 职责：
 *   1. 把 BSP 的原始码值换算成有物理意义的工程量
 *      （母线 V、抽头 V、单体 V、温度 degC、电流 A、功率 W）
 *   2. 屏蔽分压比、上拉电阻、B 值、CSA 增益这些板级细节
 *   3. 向上提供稳定的类型化接口，BL 拿到手就能用
 *
 * 边界：
 *   - 只调用 BSP 接口，不直接碰寄存器
 *   - 不做控制决策（"该不该放电"是 BL 的事）
 *   - 不做保护判断（门限属于 BL，硬保护属于 DRV8161/TLV6700）
 *
 * 本层由下面几个子模块组成，统一从 dl.h 暴露：
 *   dl_bus    母线电压、超容总压
 *   dl_cell   8 节累积抽头的换算（含 TEST 网络修正）
 *   dl_temp   3 路 NTC
 *   dl_power  电流与泄放功率估算
 *   dl_can    CAN 收发封装
 *
 * ⚠️ 本层有一处**原理图层面的缺口**，必须在 BL 之前知道：
 *   超容负极（SC−）不是系统地，而当前原理图**没有给 SC− 任何采样通道**。
 *   少了这个参考，第 1 节电压算不出来，超容总压也只能算到 T8 为止。
 *   详见 dl_cell.h 顶部的说明。
 ********************************************************************************************/

#include "dl_bus.h"
#include "dl_cell.h"
#include "dl_temp.h"
#include "dl_power.h"
#include "dl_can.h"

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_init
* Descriptions  :   Init the data layer. Must be called after bsp_init().
*                   Currently this only clears the cached values; every getter reads the
*                   BSP buffers on demand, so there is no long-running state to start.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_init
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __DL_H__ */
