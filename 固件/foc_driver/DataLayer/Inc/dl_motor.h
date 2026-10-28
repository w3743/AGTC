#ifndef __DL_MOTOR_H__
#define __DL_MOTOR_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 电机数据层 —— 三相无刷电机 + DRV8353 三相逆变器
 *
 * ⚠️ 本文件原先是从 water_car_motor（水泵，直流有刷 + DRV8262）抄过来的，
 *    里面的"IN1/IN2 反相""nSLEEP""duty = 正向占空比 0~1"全部对不上本工程。
 *    已按实际外设重写：**三相、互补、带死区、有硬件刹车**。
 *
 * 与 BSP 的关系（全部由 bsp_inverter / bsp_gpio 提供）：
 *   bsp_gpio_drv_set_enable()      -> 总闸，DRV8353 ENABLE
 *   bsp_inverter_set_duty(u,v,w)   -> 三相占空比
 *   bsp_inverter_set_output_enable -> MOE（六路输出放行/收回）
 *   bsp_inverter_break_occurred()  -> 硬件刹车是否触发过（锁存）
 *   bsp_inverter_clear_break()     -> 清刹车锁存（不清 MOE）
 *   bsp_gpio_read_break_n()        -> BREAK_N 当前电平（诊断用）
 *
 * ── 三种"停"的区别（这一节最容易搞混，实现 DL 前必须先分清）─────────────────
 *   1. COAST（自由滑行）
 *        bsp_gpio_drv_set_enable(FALSE) + MOE=0
 *        六路栅极全关，绕组开路，电机靠惯性转，无反电动势制动。
 *        ⟹ 正常停机、故障后安全隔离都用这个。
 *
 *   2. ZERO VECTOR（零矢量 / 有源短路）
 *        三相占空比全写 0（或全写 1）
 *        三相下桥全通，绕组被短接，电流在电机内部环流。
 *        ⟹ FOC 里"输出零电压"就是它，不是 coast。会有制动转矩，且 MOS 上要耗掉
 *           I²R 与反电动势产生的能量。高速时突然切零矢量会把母线顶高。
 *
 *   3. BRAKE（主动电制动）
 *        与零矢量是同一件事，只是语义上强调"我是故意在刹车"。
 *        本层提供 dl_motor_brake() 作为别名，方便 BL 表达意图。
 *
 * ── 上电/使能的正确顺序（顺序错了会有一瞬间的冲击电流）──────────────────────
 *      dl_motor_init()                  功率级关断，MOE=0，CCR 摆零点
 *      dl_motor_enable(TRUE)            BSP 内部强制零矢量 → MOE 开 → ENABLE 高
 *      dl_motor_set_duty(u, v, w)       由 BL 按斜坡一点点把电压加上去
 *
 *   ⚠️ **不能**反过来"先 set_duty 再 enable"。原因不是寄存器时序，而是电气：
 *      第一次通电的瞬间如果占空比不是零矢量，三相绕组会立刻承受阶跃电压，
 *      电流按绕组电阻在极短时间内冲上去。这台电机绕组电阻只有零点几欧，
 *      48 V 下的冲击电流是几十安量级。
 *      （早期版本这里写的是"先 set_duty 再 enable"，那句话是错的，已改。）
 *
 *   而且 enable(TRUE) 内部会把 CCR 强制拉回零点，之前 set_duty 写的值会被覆盖 ——
 *   这不是实现偷懒，是 bsp_power_stage_enable() 故意设计成"绝不带载使能"。
 *
 * 对 BL 的承诺：
 *   - 三相占空比各自钳位到 [0.0, 1.0]，越界不报错
 *   - 硬件刹车一旦触发，状态立刻变成 FAULT，且 duty 被记成 0
 *   - 接口不阻塞，可在 1 ms / 20 kHz 控制节拍内安全调用
 *
 * 需要 BL 自行保证：
 *   - 调制策略（SPWM / SVPWM / 过调制）—— 本层只接收三相占空比，不管怎么算出来
 *   - 加减速斜率、弱磁、堵转判定
 ********************************************************************************************/

/************************************Type****************************************************/
typedef enum _DL_MOTOR_STATE_E_
{
    DL_MOTOR_STATE_OFF                                              = 0,  /* ENABLE 低，功率级全关 */
    DL_MOTOR_STATE_READY,                                                 /* ENABLE 高、MOE 已开，但占空比停在零电压矢量，电机不发力 */
    DL_MOTOR_STATE_RUN,                                                   /* MOE=1，且已下发非零矢量，正在输出转矩 */
    DL_MOTOR_STATE_BRAKE,                                                 /* 零矢量，主动电制动 */
    DL_MOTOR_STATE_FAULT,                                                 /* 硬件刹车触发 / DRV 报故障 */
    DL_MOTOR_STATE_NUM
}DL_MOTOR_STATE_E;

/************************************Fcunction***********************************************/
/********************************************************************************************
* Function name :   dl_motor_init
* Descriptions  :   Init the motor data layer, leaves the power stage fully off
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_motor_init
(
    VOID
);

/********************************************************************************************
* Function name :   dl_motor_enable
* Descriptions  :   Enable or disable the power stage
*                   TRUE  -> DRV8353 ENABLE high, then MOE on (six outputs follow the duty)
*                   FALSE -> MOE off first, then ENABLE low (gates forced off = coast)
*                   The order matters: cutting the gate driver before MOE guarantees the
*                   bridge is already quiet when the outputs go to their idle state.
* Input         :
*                   enable: TRUE = power stage live, FALSE = coast
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_motor_enable
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   dl_motor_set_duty
* Descriptions  :   Set the three phase duties in one shot, each clamped to [0.0, 1.0]
*                   1.0 = high side fully on, 0.0 = low side fully on
*                   ⚠️ All-zero is a zero vector (active short), NOT coast.
* Input         :
*                   duty_u / duty_v / duty_w: 0.0 ~ 1.0
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_motor_set_duty
(
    IN  F32     duty_u,
    IN  F32     duty_v,
    IN  F32     duty_w
);

/********************************************************************************************
* Function name :   dl_motor_get_duty
* Descriptions  :   Read back the last commanded duty of one phase
* Input         :
*                   phase: 0 = U, 1 = V, 2 = W (out of range returns 0.0)
* Output        :
*                   none
* Return        :
*                   duty 0.0 ~ 1.0
*********************************************************************************************/
extern F32 dl_motor_get_duty
(
    IN  U8      phase
);

/********************************************************************************************
* Function name :   dl_motor_coast
* Descriptions  :   Free-wheel: MOE off + ENABLE low. Windings open, no braking torque.
*                   This is the "real stop". Use it for normal shutdown and for isolating
*                   the bridge after a fault.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_motor_coast
(
    VOID
);

/********************************************************************************************
* Function name :   dl_motor_brake
* Descriptions  :   Active short: three low-side switches on (zero vector, duty = 0).
*                   ⚠️ This produces braking torque and burns the back-EMF energy inside
*                      the motor and the MOSFETs. At high speed the bus voltage can rise.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_motor_brake
(
    VOID
);

/********************************************************************************************
* Function name :   dl_motor_get_fault
* Descriptions  :   Report whether the power stage is in a fault condition
*                   True when TIM1's break flag is latched, or BREAK_N is currently low.
*                   Use dl_motor_clear_fault() to re-arm; that does NOT restore the output.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE = faulted, FALSE = healthy
*********************************************************************************************/
extern BOOL dl_motor_get_fault
(
    VOID
);

/********************************************************************************************
* Function name :   dl_motor_clear_fault
* Descriptions  :   Clear the latched hardware break flag
*                   The output stays off afterwards (MOE is not restored).
*                   Only call this once the fault source is gone, then re-enable explicitly.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID dl_motor_clear_fault
(
    VOID
);

/********************************************************************************************
* Function name :   dl_motor_get_state
* Descriptions  :   Get the motor state, refreshes the fault condition internally
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   state
*********************************************************************************************/
extern DL_MOTOR_STATE_E dl_motor_get_state
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif
