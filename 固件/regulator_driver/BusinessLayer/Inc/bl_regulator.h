#ifndef __BL_REGULATOR_H__
#define __BL_REGULATOR_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 稳压策略 —— 状态机 + 安全联锁
 *
 * ── 这块板想做什么 ──────────────────────────────────────────────────────────
 *   BMS 给的功率额度、母线电压、超容电压都摆在这里，板子决定"往哪边搬多少能量"。
 *   首版策略（原理图准备文档定义的四种）：
 *     电压死区缓冲    e = Vref − Vbus；|e| ≤ ΔV 时不动，超出才请求充/放
 *     可信功率差额补偿 只有 BMS 可用功率和负载功率都有效时才补差
 *     受控储能恢复   按逐节电压、目标储能、温度逐步恢复
 *     待机           清除功率请求，仍监测母线与温度，保留软件 dump
 *
 * ── 本文件实际实现了什么（别高估）──────────────────────────────────────────
 *   实现了：状态机、安全联锁、前馈工作点、泄放的基本限幅、故障退出。
 *   **没实现**：闭环调节。
 *
 *   为什么不做闭环：原理图准备文档写得很明确 ——
 *     「软件电流/功率请求到 DRV8161 CSA、G431 ADC/COMP 与 PWM 的映射、
 *       双向零点/符号、量程、调制和环路稳定性待联合定型」。
 *   在这个前提下写一组 PI 系数出来，只是把没有依据的数字写进代码，
 *   上板之后要么不收敛、要么直接振荡炸管。所以 BL_ACTIVE 状态下只做一件事：
 *   按实测两端电压算出让电感电流**维持不变**的占空比（前馈）。
 *   这已经能让板子安全地挂在那里，而不会往任何方向猛冲。
 *
 *   等两件事齐了再补 PI：① 环路实测（阶跃响应、增益/相位）；
 *   ② COMP/DAC/TIM1 Break 的极性核实（那是硬保护，不是软件环）。
 ********************************************************************************************/

/************************************Type****************************************************/
typedef enum _BL_REG_STATE_E_
{
    BL_REG_STATE_INIT                                               = 0,    /* 上电，未初始化完 */
    BL_REG_STATE_IDLE                                               = 1,    /* 待机：功率级断电，只监测 */
    BL_REG_STATE_ARMED                                              = 2,    /* 功率级已上电，占空比停在前馈点 */
    BL_REG_STATE_ACTIVE                                             = 3,    /* 按策略搬能量（当前=前馈，无闭环） */
    BL_REG_STATE_FAULT                                              = 4,    /* 联锁触发，已断电，等清除 */
    BL_REG_STATE_NUM                                                = 5
}BL_REG_STATE_E;

/* 故障位（可按位或）。BL 自己判断出来的都记在这里，供状态帧上报。 */
#define BL_REG_FAULT_NONE                                               (0x0000U)
#define BL_REG_FAULT_BUS_OV                                             (0x0001U)   /* 母线过压 */
#define BL_REG_FAULT_BUS_UV                                             (0x0002U)   /* 母线欠压 */
#define BL_REG_FAULT_SC_UV                                              (0x0004U)   /* 超容欠压 */
#define BL_REG_FAULT_SC_OV                                              (0x0008U)   /* 超容过压 */
#define BL_REG_FAULT_SC_ABOVE_BUS                                       (0x0010U)   /* 超容高于母线，禁止继续搬 */
#define BL_REG_FAULT_TEMP_MAIN                                          (0x0020U)   /* 主功率级过温 */
#define BL_REG_FAULT_TEMP_SC                                            (0x0040U)   /* 超容过温 */
#define BL_REG_FAULT_TEMP_DUMP                                          (0x0080U)   /* 泄放电阻过温 */
#define BL_REG_FAULT_TEMP_INVALID                                       (0x0100U)   /* 温度读数无效 */
#define BL_REG_FAULT_POWER_OFF_TIMEOUT                                  (0x0200U)   /* 断电时电流未衰减 */

/************************************Macro***************************************************/
/* ── 母线工作窗 ──────────────────────────────────────────────────────────────
 * 9S4P INR18650/30P 直连 BUS：22.5–37.8 V，标称 32.4 V。
 * 上下各留 2 V 余量，避免在电池端点反复进出联锁。
 * ⚠️ 这不是保护阈值，只是"允不允许搬能量"的窗口。
 *    真正的过压保护靠 TVS 和 BMS，本板不承担。 */
#define BL_REG_BUS_MIN_MV                                               (20500U)
#define BL_REG_BUS_MAX_MV                                               (39800U)

/* ── 超容工作窗 ──────────────────────────────────────────────────────────────
 * 8S1P，BW6101 选 2.45 V 档 → 均衡启动点 8 × 2.45 = 19.6 V。
 * 器件绝对最大值 8 × 2.7 = 21.6 V，但那只是"不立刻炸"，不是工作点。
 * 所以软件上限按 19.6 V 算，下限留 0（超容可以从 0 V 开始充）。 */
#define BL_REG_SC_MIN_MV                                                (0U)
#define BL_REG_SC_MAX_MV                                                (19600U)

/* ── 温度门限 ────────────────────────────────────────────────────────────────
 * 全部是**首版候选值**，没有热仿真依据。NTC 本身在 80 °C 附近就有 3–5 °C 误差，
 * 所以门限要留够余量，宁可早报也不要晚报。 */
#define BL_REG_TEMP_MAIN_MAX_C                                          (75.0F)
#define BL_REG_TEMP_SC_MAX_C                                           (55.0F)
#define BL_REG_TEMP_DUMP_MAX_C                                         (200.0F)   /* 泄放电阻是功率电阻，允许更高 */

/* ── 前馈占空比限幅 ──────────────────────────────────────────────────────────
 * 前馈算出来的平衡占空比理论上在 0~1 之间，但两端接近相等时数值会很敏感。
 * 限到 ±0.90 相当于 high_side 占空比 5%~95%，永远不贴到端点。
 * 贴到端点意味着某一只管子长期常通 —— 那是故障状态，不是工作状态。 */
#define BL_REG_DUTY_LIMIT                                               (0.90F)

/* ── 泄放默认限幅 ────────────────────────────────────────────────────────────
 * 外置泄放电阻的功率/热参数**尚未确定**，所以默认占空比上限只给 20%，
 * 并且要求调用者显式给出电阻阻值后才允许提高。
 * ⚠️ 这里只是软件限幅，不构成热保护 —— 电阻会不会烧取决于它的实际额定功率。 */
#define BL_REG_DUMP_DUTY_MAX                                            (0.20F)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bl_regulator_init
* Descriptions  :   Reset the state machine to IDLE with the power stage off
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS
*********************************************************************************************/
extern TYPE_RET_E bl_regulator_init
(
    VOID
);

/********************************************************************************************
* Function name :   bl_regulator_task_1ms
* Descriptions  :   One step of the state machine. Refreshes the measurement cache, evaluates
*                   the interlocks and updates the bridge duty / dump duty.
*                   WARNING: this may call bsp_power_stage_enable(), which blocks on HAL_Delay
*                   while waiting for the inductor current to decay. Never call it from an
*                   interrupt context - run it from the main loop paced by bsp_get_tick().
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_regulator_task_1ms
(
    VOID
);

/********************************************************************************************
* Function name :   bl_regulator_request_enable
* Descriptions  :   Ask for the power stage to be energised (TRUE) or shut down (FALSE).
*                   This is a request, not a command: the interlocks still have to pass for
*                   the state machine to reach ARMED / ACTIVE.
* Input         :
*                   enable: TRUE = request run, FALSE = request stop
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL (fault active)
*********************************************************************************************/
extern TYPE_RET_E bl_regulator_request_enable
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bl_regulator_set_bus_ref_mv
* Descriptions  :   Set the bus reference used by the dead-band strategy.
*                   Stored only; the dead band itself is not implemented yet.
* Input         :
*                   ref_mv: target bus voltage in mV
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
extern TYPE_RET_E bl_regulator_set_bus_ref_mv
(
    IN  U32     ref_mv
);

/********************************************************************************************
* Function name :   bl_regulator_clear_fault
* Descriptions  :   Clear a latched fault. Only allowed when every interlock now reads healthy,
*                   so a real fault cannot be cleared by simply asking twice.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS: fault cleared, state moved to IDLE
*                   TYPE_RET_FAIL   : an interlock is still tripped
*********************************************************************************************/
extern TYPE_RET_E bl_regulator_clear_fault
(
    VOID
);

/********************************************************************************************
* Function name :   bl_regulator_get_state
* Descriptions  :   Current state of the state machine
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   BL_REG_STATE_E
*********************************************************************************************/
extern BL_REG_STATE_E bl_regulator_get_state
(
    VOID
);

/********************************************************************************************
* Function name :   bl_regulator_get_fault
* Descriptions  :   Latched fault bits (BL_REG_FAULT_*)
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   bit mask
*********************************************************************************************/
extern U16 bl_regulator_get_fault
(
    VOID
);

/********************************************************************************************
* Function name :   bl_regulator_get_bus_mv / get_sc_mv / get_current_ma
* Descriptions  :   Cached measurements, refreshed once per 1 ms tick
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   cached value
*********************************************************************************************/
extern U32 bl_regulator_get_bus_mv
(
    VOID
);

extern U32 bl_regulator_get_sc_mv
(
    VOID
);

extern S32 bl_regulator_get_current_ma
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BL_REGULATOR_H__ */
