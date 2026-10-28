#ifndef __BL_MOTOR_H__
#define __BL_MOTOR_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 电机策略 —— 状态机 + 安全联锁 + 电压斜坡
 *
 * ── 这一层负责"该不该开、开多大、出事了怎么办" ──────────────────────────────
 *   测量的活全在 DL（母线、三相电流、两路温度、霍尔），
 *   调制的活全在 bl_foc（Clarke/Park/SVPWM），
 *   本文件只做决策：
 *     · 什么时候允许使能（联锁全部通过）
 *     · 使能之后电压怎么加上去（斜坡，不能一步到位）
 *     · 出现异常时先做什么后做什么（先降额、再断电、最后锁存）
 *
 * ── 本文件实际实现了什么（别高估）──────────────────────────────────────────
 *   实现了：5 状态机、9 个故障位、母线/电流/温度联锁、电压斜坡、软停。
 *   **没实现**：转速环、转矩指令、弱磁、堵转判定、自启动策略。
 *
 *   最关键的缺口是**自启动**：本层用的是霍尔扇区中心角（±30° 误差），
 *   静止时扇区不变化，定子磁场也就停在原地 —— 转子可能被拉过去一点就停住，
 *   也可能因为误差太大而根本拉不动。要可靠自启动，至少还需要下面其中一项：
 *     ① 角度细分（TIM2 霍尔接口模式或输入捕获插值）；
 *     ② 电流环 + 编码器；
 *     ③ 开环强拖（I/F 或 V/f）启动，转到一定速度再切到霍尔。
 *   在补齐之前，本层的能力边界是：**验证相序、方向、驱动链路、母线电流采样**，
 *   以及在空载/轻载条件下让电机转起来看现象。
 *
 * ── 与稳压模块工程的关系 ───────────────────────────────────────────────────
 *   regulator_driver 里的 bl_regulator 是同一套骨架（状态机 + 联锁 + 前馈 + 不闭环），
 *   两个工程共用同一颗 G431，但插不同的功率板、跑不同的固件。
 *   对照着看能省很多时间 —— 那边的注释里写了"为什么不做闭环"，理由和这边一样：
 *   参数没有实测依据时不编系数。
 ********************************************************************************************/

/************************************Macro***************************************************/
/* ── 温度保护总开关 ──────────────────────────────────────────────────────────
 * 两路 NTC 的 R25 与 B 值目前是**占位值**（10k / 3950，见 dl_sense.h）。
 * 参数不对时，温度读数可能是"看着像个数"的假数，会误报过温把板子锁在 FAULT。
 *
 * 首次上板建议：先设 0，把电机转起来、确认相序和电流采样都对，
 * 再用万用表+温度计标定 NTC，标完之后设回 1。
 * ⚠️ 设 0 期间必须有人盯着，且不要让电机长时间堵转 —— 那时没有任何温度兜底。 */
#define BL_MOTOR_TEMP_PROTECT_ENABLE                                    (1)

/* ── 母线工作窗 ──────────────────────────────────────────────────────────────
 * 本板标称 48 V 母线。上下限是**首版兜底门限**，不是精确的保护值：
 *   上限 58 V —— 留出再生制动时母线被顶高的余量，同时远低于 ADC 满量程（62.7 V）。
 *               超过 62.7 V 时 ADC 会削顶，读数反而开始"变小"，那才是最危险的状态。
 *   下限 30 V —— 低于此认为母线掉电或严重欠压，此时栅极驱动的 UVLO 也可能动作。
 * ⚠️ 实际电池串数确认后必须重算这两个值（13S 锂电是 39 ~ 54.6 V）。 */
#define BL_MOTOR_BUS_MIN_V                                              (30.0F)
#define BL_MOTOR_BUS_MAX_V                                              (58.0F)

/* ── 相电流过流门限 ──────────────────────────────────────────────────────────
 * 30 A 是**首版保守值**，没有依据 —— 这台电机的额定/峰值电流还没给。
 * 采样链路本身能测 ±165 A（0.5 mΩ × 20 V/V，零点 VREF/2），所以门限完全由电机决定。
 *
 * 这个值直接决定"会不会烧管"。第一次上板建议先调到 5 A，把电机转起来看波形，
 * 确认电流方向、零点、噪声都正常之后再放到真实值。
 * 用 bl_motor_set_limits() 可以在运行时改，不用重烧。 */
#define BL_MOTOR_PHASE_OC_MA                                            (30000)

/* ── 温度门限 ────────────────────────────────────────────────────────────────
 * 首版候选值，没有热仿真依据。NTC 本身在 80 °C 附近就有 3~5 °C 误差，
 * 门限要留够余量，宁可早报也不要晚报。
 * MOS 门限量的是散热器，不是结温 —— 结温比它高得多，瞬态过流靠硬件 OCP 兜。 */
#define BL_MOTOR_TEMP_MOS_MAX_C                                         (85.0F)
#define BL_MOTOR_TEMP_MOTOR_MAX_C                                       (110.0F)

/* ── 电压指令上限 ────────────────────────────────────────────────────────────
 * 开环电压模式下，这是"人能给出去的最大电压"，与母线无关（母线限幅在 bl_foc 里另做）。
 * 48 V 母线的 SVPWM 线性区上限是 48/√3 ≈ 27.7 V，所以 20 V 已经接近 72% 调制比。
 * 第一次上板请从 2~3 V 起，用 bl_motor_set_vq_v() 慢慢加。 */
#define BL_MOTOR_VOLTAGE_MAX_V                                          (20.0F)

/* ── 电压斜坡 ────────────────────────────────────────────────────────────────
 * 20 V/s：从 0 加到 20 V 要 1 秒。慢是故意的 ——
 * 开环电压模式下没有电流环限流，唯一挡住浪涌的就是这条斜坡。
 * 觉得太慢可以直接改，但改之前想清楚：斜坡每快一倍，启动电流冲击也大一倍。 */
#define BL_MOTOR_RAMP_V_PER_S                                           (20.0F)

/* 电压低于这个值就认为"已经停住了"，可以从 ARMED 退回 IDLE（或执行软停收尾）。
 * 不能用 0.0F 判断：浮点斜坡是逐步逼近的，可能永远到不了精确的 0。 */
#define BL_MOTOR_VOLTAGE_ZERO_V                                         (0.05F)

/* ── 软停超时 ────────────────────────────────────────────────────────────────
 * 收到停止请求后先把电压斜坡降到 0 再断电，最多等这么久。
 * 超时就直接断电（宁可让电流硬断，也不能因为等不到 0 而不停机）。 */
#define BL_MOTOR_STOP_TIMEOUT_MS                                        (2000U)

/* ── DRV8353 探测重试周期 ────────────────────────────────────────────────────
 * bsp_init() 探测失败不代表芯片坏了（可能是上电顺序、SPI 接线松动）。
 * 在 IDLE 状态下每 500 ms 重探一次，探通了自动放行，不用人工复位整个系统。 */
#define BL_MOTOR_DRV_RETRY_MS                                           (500U)

/************************************Type****************************************************/
typedef enum _BL_MOTOR_STATE_E_
{
    BL_MOTOR_STATE_INIT                                             = 0,    /* 上电，还没初始化完 */
    BL_MOTOR_STATE_IDLE                                             = 1,    /* 待机：功率级断电，只监测 */
    BL_MOTOR_STATE_ARMED                                            = 2,    /* 功率级已使能，电压在零点附近 */
    BL_MOTOR_STATE_RUN                                              = 3,    /* 正在输出电压 */
    BL_MOTOR_STATE_FAULT                                            = 4,    /* 联锁触发，已断电，等清除 */
    BL_MOTOR_STATE_NUM                                              = 5
}BL_MOTOR_STATE_E;

/* 故障位（可按位或）。
 *
 * 分成"锁存"与"非锁存"两类，这个区分很重要：
 *   锁存的（LATCHED）：过流、过压、过温、硬件刹车、霍尔故障、DRV 通信故障
 *     —— 这些都需要人看一眼现场再决定，自动恢复只会让故障反复出现。
 *   非锁存的（TRANSIENT）：母线欠压
 *     —— 电池放电、上电爬升、负载突加都会让它短暂成立，锁存只会让人反复手动复位。
 *     它只阻止"从 IDLE 进 ARMED"，并会把 RUN 退回 IDLE，母线回来自动恢复。 */
#define BL_MOTOR_FAULT_NONE                                             (0x0000U)
#define BL_MOTOR_FAULT_HW_BREAK                                         (0x0001U)   /* 锁存：TIM1 刹车 / BREAK_N 低 */
#define BL_MOTOR_FAULT_BUS_OV                                           (0x0002U)   /* 锁存：母线过压 */
#define BL_MOTOR_FAULT_BUS_UV                                           (0x0004U)   /* 瞬态：母线欠压 */
#define BL_MOTOR_FAULT_PHASE_OC                                         (0x0008U)   /* 锁存：相电流过流 */
#define BL_MOTOR_FAULT_TEMP_MOS                                         (0x0010U)   /* 锁存：MOS 散热器过温 */
#define BL_MOTOR_FAULT_TEMP_MOTOR                                       (0x0020U)   /* 锁存：电机绕组过温 */
#define BL_MOTOR_FAULT_TEMP_INVALID                                     (0x0040U)   /* 锁存：NTC 读数无效 */
#define BL_MOTOR_FAULT_DRV_COMM                                         (0x0080U)   /* 锁存：DRV8353 SPI 无应答 */
#define BL_MOTOR_FAULT_HALL_INVALID                                     (0x0100U)   /* 锁存：霍尔码无效 */

/* 锁存掩码：clear_fault() 只清这些；瞬态的靠自己恢复 */
#define BL_MOTOR_FAULT_LATCHED_MASK                                     \
    (BL_MOTOR_FAULT_HW_BREAK | BL_MOTOR_FAULT_BUS_OV | BL_MOTOR_FAULT_PHASE_OC \
   | BL_MOTOR_FAULT_TEMP_MOS | BL_MOTOR_FAULT_TEMP_MOTOR                     \
   | BL_MOTOR_FAULT_TEMP_INVALID | BL_MOTOR_FAULT_DRV_COMM | BL_MOTOR_FAULT_HALL_INVALID)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bl_motor_init
* Descriptions  :   Init the business layer. Must be called after dl_init().
*                   Leaves the power stage DE-ENERGISED: nothing is enabled just because
*                   the firmware booted.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E bl_motor_init
(
    VOID
);

/********************************************************************************************
* Function name :   bl_motor_task_1ms
* Descriptions  :   One step of the state machine. Refreshes the measurement cache, evaluates
*                   the interlocks, ramps the voltage and runs the modulation chain.
*
*                   ⚠️ 必须由主循环按 1 ms 节拍调用，**不能放进 SysTick_Handler**。
*                      它内部切功率级时会走 bsp_power_stage_enable()，那个函数用 HAL_Delay()
*                      轮询 DRV8353 的故障位，而 HAL_Delay() 依赖 SysTick 计数 ——
*                      放进 SysTick 里会直接死锁（等一个永远不会推进的 tick）。
*
* 输入参数 :   none
* 输出参数 :   none
* 返回值   :   none
*********************************************************************************************/
extern VOID bl_motor_task_1ms
(
    VOID
);

/********************************************************************************************
* Function name :   bl_motor_request_enable
* Descriptions  :   Ask for the power stage to be energised (TRUE) or shut down (FALSE).
*                   This is a request, not a command: the interlocks still have to pass.
*                   A FALSE request triggers a soft stop (ramp down, then cut power).
* Input         :
*                   enable: TRUE = request run, FALSE = request stop
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL (a latched fault blocks the request)
*********************************************************************************************/
extern TYPE_RET_E bl_motor_request_enable
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bl_motor_set_vq_v
* Descriptions  :   Set the open-loop q-axis voltage command, in volts.
*                   Clamped to +-BL_MOTOR_VOLTAGE_MAX_V.
*                   This is the "throttle" when the current loop is off (the default).
*
*                   ⚠️ 开环电压模式下，给多少伏就是多少伏，没有电流限制 ——
*                      唯一挡住浪涌的是电压斜坡和过流联锁。第一次上板从 2~3 V 起。
* Input         :
*                   v_q: q 轴电压，V，带符号（正负决定转向）
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_motor_set_vq_v
(
    IN  F32     v_q
);

/********************************************************************************************
* Function name :   bl_motor_set_vd_v
* Descriptions  :   Set the d-axis voltage command, in volts. Normally stays at 0 for a
*                   surface-mounted PM machine. Only useful for experiments (field weakening,
*                   hall angle calibration).
* Input         :
*                   v_d: d 轴电压，V
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_motor_set_vd_v
(
    IN  F32     v_d
);

/********************************************************************************************
* Function name :   bl_motor_set_limits
* Descriptions  :   Override the over-current / over-temperature thresholds at runtime
*                   (no reflash). Pass 0 to keep the current value.
* Input         :
*                   phase_oc_ma   : 过流门限，mA
*                   temp_mos_max_c: MOS 过温门限，degC
*                   temp_motor_max_c: 电机过温门限，degC
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_motor_set_limits
(
    IN  S32     phase_oc_ma,
    IN  F32     temp_mos_max_c,
    IN  F32     temp_motor_max_c
);

/********************************************************************************************
* Function name :   bl_motor_clear_fault
* Descriptions  :   Clear the latched fault bits.
*                   Only allowed when every interlock now reads healthy, so a real fault
*                   cannot be cleared by simply asking twice.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS: cleared, state moved to IDLE
*                   TYPE_RET_FAIL   : an interlock is still tripped
*********************************************************************************************/
extern TYPE_RET_E bl_motor_clear_fault
(
    VOID
);

/********************************************************************************************
* Function name :   bl_motor_get_state / get_fault / get_bus_v / get_phase_ma / get_temp_c
* Descriptions  :   Cached state and measurements, refreshed once per 1 ms tick
* Input         :
*                   (get_phase_ma) phase: 0 = U, 1 = V, 2 = W
*                   (get_temp_c)   which: 0 = MOS, 1 = motor
* Output        :
*                   none
* Return        :
*                   cached value
*********************************************************************************************/
extern BL_MOTOR_STATE_E bl_motor_get_state
(
    VOID
);

extern U16 bl_motor_get_fault
(
    VOID
);

extern F32 bl_motor_get_bus_v
(
    VOID
);

extern S32 bl_motor_get_phase_ma
(
    IN  U8      phase
);

extern F32 bl_motor_get_temp_c
(
    IN  U8      which
);

/********************************************************************************************
* Function name :   bl_motor_get_vq_applied_v
* Descriptions  :   The voltage actually being commanded right now (after the ramp), volts
*                   During a soft stop this is what the ramp is bringing down to zero.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   applied q-axis voltage, V
*********************************************************************************************/
extern F32 bl_motor_get_vq_applied_v
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BL_MOTOR_H__ */
