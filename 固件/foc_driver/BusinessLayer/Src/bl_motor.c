/************************************包含头文件*************************************************/
#include <stddef.h>
#include "type.h"
#include "bsp.h"
#include "dl_motor.h"
#include "dl_sense.h"
#include "bl_foc.h"
#include "bl_motor.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 相数，只为下标好读 */
#define BL_MOTOR_PHASE_NUM                                              (3U)

/* 电压"是否已经归零"的判据，带上斜坡步长的余量。
 * 直接用 BL_MOTOR_VOLTAGE_ZERO_V 会有个隐患：如果斜坡步长比它大，
 * 电压会在 +step 和 -step 之间来回跨过零点，永远满足不了 |v| < 0.05。
 * 取两者中较大的一个就一定能收敛到 target。 */
#define BL_MOTOR_ZERO_DEADBAND_V                                        \
    ((BL_MOTOR_RAMP_V_PER_S * 0.001F) > BL_MOTOR_VOLTAGE_ZERO_V         \
        ? (BL_MOTOR_RAMP_V_PER_S * 0.001F) : BL_MOTOR_VOLTAGE_ZERO_V)


/************************************数据类型定义************************************************/
typedef struct _BL_MOTOR_MODULE_T_
{
    BL_MOTOR_STATE_E    state;                                          /* 状态机 */
    U16                 fault;                                          /* 故障位 */

    /* 指令 */
    BOOL                run_request;                                    /* 上层请求运行 */
    F32                 vq_cmd;                                         /* q 轴电压指令，V */
    F32                 vd_cmd;                                         /* d 轴电压指令，V */

    /* 斜坡后的实际值 */
    F32                 vq_applied;                                     /* 实际下发的 q 轴电压 */
    F32                 vd_applied;                                     /* 实际下发的 d 轴电压 */

    /* 采样缓存 */
    F32                 bus_v;
    S32                 phase_ma[BL_MOTOR_PHASE_NUM];
    F32                 temp_c[2];

    /* 可运行时调整的门限 */
    S32                 oc_ma;
    F32                 temp_mos_max_c;
    F32                 temp_motor_max_c;

    /* 计时 */
    U32                 last_tick;
    U32                 stop_start_tick;
    U32                 last_probe_tick;
    BOOL                drv_ok;                                         /* DRV8353 SPI 是否应答 */
}BL_MOTOR_MODULE_T;


/************************************静态函数声明************************************************/
static BL_MOTOR_MODULE_T * _bl_motor_get_module
(
    VOID
);

static VOID _bl_motor_refresh_sample
(
    VOID
);

static VOID _bl_motor_eval_interlocks
(
    VOID
);

static BOOL _bl_motor_latched_ok
(
    VOID
);

static BOOL _bl_motor_transient_ok
(
    VOID
);

static F32 _bl_motor_ramp
(
    IN  F32     current,
    IN  F32     target,
    IN  U32     dt_ms
);

static VOID _bl_motor_enter_fault
(
    VOID
);

static VOID _bl_motor_task_idle
(
    IN  U32     now
);

static VOID _bl_motor_task_active
(
    IN  U32     now,
    IN  U32     dt_ms
);


/************************************静态变量定义************************************************/
static BL_MOTOR_MODULE_T s_bl_motor_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bl_motor_get_module
* 功能描述 :   获取电机业务模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BL_MOTOR_MODULE_T * _bl_motor_get_module
(
    VOID
)
{
    return &s_bl_motor_module;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_refresh_sample
* 功能描述 :   把本拍的测量值抓进缓存
*
*              为什么要缓存而不是每次用的时候现取：
*                bl_motor_task_1ms 里同一拍会多次用到母线电压（联锁一次、调制一次），
*                现取的话两次之间 DMA 可能已经刷新了缓冲，联锁判的和调制用的是两个值。
*                缓存之后，"这一拍的决策基于同一组数"这件事才有保证。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_motor_refresh_sample
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    /* 先让 DL 把原始采样换算成本拍的一组工程量 */
    dl_sense_update();

    p_mod->bus_v = dl_sense_get_bus_v();

    p_mod->phase_ma[0] = dl_sense_get_phase_current_ma(DL_SENSE_PHASE_U);
    p_mod->phase_ma[1] = dl_sense_get_phase_current_ma(DL_SENSE_PHASE_V);
    p_mod->phase_ma[2] = dl_sense_get_phase_current_ma(DL_SENSE_PHASE_W);

    p_mod->temp_c[0] = dl_sense_get_temp_c(DL_SENSE_TEMP_MOS);
    p_mod->temp_c[1] = dl_sense_get_temp_c(DL_SENSE_TEMP_MOTOR);
}

/********************************************************************************************
* 函数名称 :   _bl_motor_eval_interlocks
* 功能描述 :   评估所有联锁，把成立的条件写进故障位
*
*              只置位、不清位 —— 清位由各自的恢复条件负责（见 bl_motor_task_1ms 里的
*              瞬态项处理）。这样"曾经发生过"和"现在正在发生"能分开看：
*              故障位里既有锁存的也有瞬态的，锁存位只能由 clear_fault() 清。
*
*              过流的取法：三相取绝对值最大者。用 max(|Iu|,|Iv|,|Iw|) 而不是只看两相，
*              是因为 Iw 有可能单独最大（比如某一相下桥短路），只看 Iu/Iv 会漏。
*              Iw 由基尔霍夫定律反算，不需要额外采样 —— 星形无中线，三者和恒为 0。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_motor_eval_interlocks
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();
    S32                i_w;
    S32                i_abs_max;
    S32                i_abs;
    U8                 i;

    /* ── 硬件刹车 / DRV 报故障 ─────────────────────────────────────────────── */
    if (dl_motor_get_state() == DL_MOTOR_STATE_FAULT)
    {
        p_mod->fault |= BL_MOTOR_FAULT_HW_BREAK;
    }

    /* ── 母线电压 ─────────────────────────────────────────────────────────── */
    if (p_mod->bus_v > BL_MOTOR_BUS_MAX_V)
    {
        p_mod->fault |= BL_MOTOR_FAULT_BUS_OV;
    }

    if (p_mod->bus_v < BL_MOTOR_BUS_MIN_V)
    {
        p_mod->fault |= BL_MOTOR_FAULT_BUS_UV;
    }

    /* ── 相电流 ───────────────────────────────────────────────────────────── */
    i_w = -(p_mod->phase_ma[0] + p_mod->phase_ma[1]);

    i_abs_max = p_mod->phase_ma[0];
    if (i_abs_max < 0) { i_abs_max = -i_abs_max; }

    i_abs = p_mod->phase_ma[1];
    if (i_abs < 0) { i_abs = -i_abs; }
    if (i_abs > i_abs_max) { i_abs_max = i_abs; }

    i_abs = i_w;
    if (i_abs < 0) { i_abs = -i_abs; }
    if (i_abs > i_abs_max) { i_abs_max = i_abs; }

    if (i_abs_max > p_mod->oc_ma)
    {
        p_mod->fault |= BL_MOTOR_FAULT_PHASE_OC;
    }

    /* ── 温度 ─────────────────────────────────────────────────────────────── */
#if (BL_MOTOR_TEMP_PROTECT_ENABLE == 1)
    for (i = 0U; i < 2U; i++)
    {
        /* 无效读数（开路/短路 NTC）单独一位：它和"真的很热"是两件事，
         * 但处置方式一样 —— 都不能继续跑。 */
        if (p_mod->temp_c[i] <= DL_SENSE_TEMP_INVALID_C)
        {
            p_mod->fault |= BL_MOTOR_FAULT_TEMP_INVALID;
        }
    }

    if (p_mod->temp_c[0] > p_mod->temp_mos_max_c)
    {
        p_mod->fault |= BL_MOTOR_FAULT_TEMP_MOS;
    }

    if (p_mod->temp_c[1] > p_mod->temp_motor_max_c)
    {
        p_mod->fault |= BL_MOTOR_FAULT_TEMP_MOTOR;
    }
#else
    /* 温度保护被编译开关关掉时，什么都不判。
     * 保留 (VOID) 引用是为了让 temp_c 在关掉保护后也不触发 unused 告警 ——
     * 这个变量在别处（getter、CAN 上报）还有用。 */
    (VOID)i;
#endif

    /* ── 霍尔 / DRV 通信 ──────────────────────────────────────────────────── */
    if (bl_foc_get_sector() == BL_FOC_HALL_SECTOR_DEFAULT_INVALID)
    {
        /* 扇区无效意味着"不知道该往哪个方向加电压"，这时候继续输出就是在瞎赌。
         * 只在**已经在运行**时才算故障：待机时霍尔没插、电机没装都是正常的，
         * 那时候置故障位会让整机上电就锁在 FAULT。 */
        if ((p_mod->state == BL_MOTOR_STATE_ARMED) || (p_mod->state == BL_MOTOR_STATE_RUN))
        {
            p_mod->fault |= BL_MOTOR_FAULT_HALL_INVALID;
        }
    }

    if (p_mod->drv_ok != TRUE)
    {
        p_mod->fault |= BL_MOTOR_FAULT_DRV_COMM;
    }
}

/********************************************************************************************
* 函数名称 :   _bl_motor_latched_ok
* 功能描述 :   锁存类故障是否全部为 0
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 无锁存故障
*********************************************************************************************/
static BOOL _bl_motor_latched_ok
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if ((p_mod->fault & BL_MOTOR_FAULT_LATCHED_MASK) == 0U)
    {
        return TRUE;
    }

    return FALSE;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_transient_ok
* 功能描述 :   瞬态类条件是否满足（当前只有母线欠压）
*
*              ⚠️ 这里判的是"当前母线电压"，不是故障位。
*                 因为故障位只置不清，用它判会导致"欠压过一次就永远进不去"。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 母线在工作窗内
*********************************************************************************************/
static BOOL _bl_motor_transient_ok
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (p_mod->bus_v < BL_MOTOR_BUS_MIN_V)
    {
        return FALSE;
    }

    return TRUE;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_ramp
* 功能描述 :   把 current 按固定斜率朝 target 推进一步
*
*              用真实经过时间 dt_ms 而不是"假设每次都是 1 ms"：
*              主循环偶尔被 CAN 处理或串口打印卡住时，dt 会大于 1，
*              按 dt=1 算会让斜坡实际变慢（电压跟不上指令，看起来像"带载能力不足"）。
*              用真实时间就跟卡顿解耦了。
*
* 输入参数 :   current: 当前值；target: 目标值；dt_ms: 距上次调用的毫秒数
* 输出参数 :   无
* 返回值   :   推进后的值
*********************************************************************************************/
static F32 _bl_motor_ramp
(
    IN  F32     current,
    IN  F32     target,
    IN  U32     dt_ms
)
{
    F32 step;
    F32 delta;

    step  = BL_MOTOR_RAMP_V_PER_S * (((F32)dt_ms) / 1000.0F);
    delta = target - current;

    if (delta > step)
    {
        return current + step;
    }

    if (delta < -step)
    {
        return current - step;
    }

    return target;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_enter_fault
* 功能描述 :   进入故障态的收尾动作：断电、清指令、状态迁移
*
*              顺序：先把电压指令清 0、再断电，而不是反过来。
*              理由很实际：断电后 CCR 里还留着上一次的值，如果哪天有人加了
*              "重新使能"的路径而忘了重置指令，就会带着旧电压直接上电。
*              在这里清掉，等于给下次使能留一个干净的起点。
*
*              断电走 dl_motor_enable(FALSE)（= COAST），不走 dl_motor_brake()：
*              故障时不知道电机在什么状态，主动电制动会把反电动势能量全灌进绕组和 MOS。
*              让绕组开路、电流经体二极管回流到母线，是逆变器故障时的标准动作。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_motor_enter_fault
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    p_mod->run_request = FALSE;
    p_mod->vq_cmd      = 0.0F;
    p_mod->vd_cmd      = 0.0F;
    p_mod->vq_applied  = 0.0F;
    p_mod->vd_applied  = 0.0F;

    (VOID)dl_motor_enable(FALSE);

    p_mod->state = BL_MOTOR_STATE_FAULT;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_task_idle
* 功能描述 :   IDLE 状态的周期处理：重探 DRV8353，条件满足就使能
* 输入参数 :   now: 当前 tick
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_motor_task_idle
(
    IN  U32     now
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    /* ── 1. DRV8353 通信重试 ──────────────────────────────────────────────────
     * bsp_init() 那次探测失败可能只是上电时序问题（VM 还没爬上来、SPI 还没同步）。
     * 这里每 500 ms 重探一次，探通了就自动清掉 DRV_COMM 位 —— 不用人工复位整机。 */
    if (p_mod->drv_ok != TRUE)
    {
        if ((now - p_mod->last_probe_tick) >= BL_MOTOR_DRV_RETRY_MS)
        {
            p_mod->last_probe_tick = now;

            if (bsp_probe_drv8353() == TRUE)
            {
                p_mod->drv_ok = TRUE;
                p_mod->fault &= (U16)(~BL_MOTOR_FAULT_DRV_COMM);
            }
        }
    }

    /* ── 2. 允许使能吗 ────────────────────────────────────────────────────── */
    if (p_mod->run_request != TRUE)
    {
        return;
    }

    if ((_bl_motor_latched_ok() != TRUE) || (_bl_motor_transient_ok() != TRUE))
    {
        return;
    }

    /* ── 3. 使能 ────────────────────────────────────────────────────────────
     * dl_motor_enable(TRUE) 内部会：查 DRV 故障 → 强制零矢量 → MOE → ENABLE。
     * 失败就留在 IDLE 下一拍再试（可能是 VM 还没稳），不置故障 —— 理由见 dl_motor.c。 */
    if (dl_motor_enable(TRUE) != TYPE_RET_SUCCESS)
    {
        return;
    }

    p_mod->vq_applied = 0.0F;
    p_mod->vd_applied = 0.0F;
    p_mod->state      = BL_MOTOR_STATE_ARMED;
}

/********************************************************************************************
* 函数名称 :   _bl_motor_task_active
* 功能描述 :   ARMED / RUN 状态的周期处理：斜坡 + 软停 + 调制
*
*              ARMED 与 RUN 共用同一段代码，只靠电压幅值区分状态 ——
*              这两个状态的**动作**是一样的（都在按斜坡跟随指令），
*              区别只在语义（有没有输出转矩）和对外上报的状态字。
*              拆成两段几乎一样的代码只会让人以为它们真的有区别。
*
* 输入参数 :   now: 当前 tick；dt_ms: 距上次调用的毫秒数
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_motor_task_active
(
    IN  U32     now,
    IN  U32     dt_ms
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();
    F32                target;

    /* ── 1. 决定目标电压 ──────────────────────────────────────────────────── */
    if (p_mod->run_request == TRUE)
    {
        target = p_mod->vq_cmd;
        p_mod->stop_start_tick = now;
    }
    else
    {
        /* 软停：先把电压斜坡降到 0，再断电。
         * 不直接 coast 是因为：带着电流硬断，绕组电感会把 di/dt 变成一个电压尖峰
         * 顶到母线上（虽然体二极管能吸收，但母线上会有一次明显的过冲，
         * 长时间这么干对母线电容不好）。斜坡降下来，电流自然就小了。 */
        target = 0.0F;
    }

    /* ── 2. 斜坡 ──────────────────────────────────────────────────────────── */
    p_mod->vq_applied = _bl_motor_ramp(p_mod->vq_applied, target, dt_ms);
    p_mod->vd_applied = _bl_motor_ramp(p_mod->vd_applied, p_mod->vd_cmd, dt_ms);

    /* ── 3. 软停是否已经完成 ──────────────────────────────────────────────── */
    if (p_mod->run_request != TRUE)
    {
        if ((!((p_mod->vq_applied > BL_MOTOR_ZERO_DEADBAND_V)
            || (p_mod->vq_applied < -BL_MOTOR_ZERO_DEADBAND_V)))
         || ((now - p_mod->stop_start_tick) >= BL_MOTOR_STOP_TIMEOUT_MS))
        {
            /* 到位了，或者等太久了（宁可硬断也不能不停机） */
            p_mod->vq_applied = 0.0F;
            p_mod->vd_applied = 0.0F;

            (VOID)dl_motor_enable(FALSE);

            p_mod->state = BL_MOTOR_STATE_IDLE;
            return;
        }
    }

    /* ── 4. 跑一步调制链 ────────────────────────────────────────────────────
     * 注意顺序：先斜坡后调制。反过来的话，这一拍用的还是上一拍的值，
     * 相当于凭空多了一拍延迟 —— 在 1 ms 的节拍里就是 1 ms 的相位滞后。 */
    (VOID)bl_foc_run(p_mod->vd_applied, p_mod->vq_applied);

    /* ── 5. ARMED ↔ RUN 迁移 ──────────────────────────────────────────────── */
    if (p_mod->run_request == TRUE)
    {
        if ((p_mod->vq_applied > BL_MOTOR_ZERO_DEADBAND_V)
         || (p_mod->vq_applied < -BL_MOTOR_ZERO_DEADBAND_V)
         || (p_mod->vd_applied > BL_MOTOR_ZERO_DEADBAND_V)
         || (p_mod->vd_applied < -BL_MOTOR_ZERO_DEADBAND_V))
        {
            p_mod->state = BL_MOTOR_STATE_RUN;
        }
        else
        {
            p_mod->state = BL_MOTOR_STATE_ARMED;
        }
    }
}

/********************************************************************************************
* 函数名称 :   bl_motor_init
* 功能描述 :   初始化电机业务层，把功率级留在完全关断的状态
*
*              这里调一次 bsp_probe_drv8353()，和 bsp_init() 里那次重复。
*              重复是故意的：DL/BL 不应该依赖"调用者已经检查过"，
*              自己确认一次的成本是几帧 SPI，收益是自己知道自己脚下的地是实的。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_motor_init
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    (VOID)bl_foc_init();

    p_mod->state           = BL_MOTOR_STATE_INIT;
    p_mod->fault           = BL_MOTOR_FAULT_NONE;
    p_mod->run_request     = FALSE;
    p_mod->vq_cmd          = 0.0F;
    p_mod->vd_cmd          = 0.0F;
    p_mod->vq_applied      = 0.0F;
    p_mod->vd_applied      = 0.0F;
    p_mod->bus_v           = 0.0F;
    p_mod->last_tick       = bsp_get_tick();
    p_mod->stop_start_tick = p_mod->last_tick;
    p_mod->last_probe_tick = p_mod->last_tick;

    p_mod->phase_ma[0] = 0;
    p_mod->phase_ma[1] = 0;
    p_mod->phase_ma[2] = 0;
    p_mod->temp_c[0]   = DL_SENSE_TEMP_INVALID_C;
    p_mod->temp_c[1]   = DL_SENSE_TEMP_INVALID_C;

    p_mod->oc_ma            = BL_MOTOR_PHASE_OC_MA;
    p_mod->temp_mos_max_c   = BL_MOTOR_TEMP_MOS_MAX_C;
    p_mod->temp_motor_max_c = BL_MOTOR_TEMP_MOTOR_MAX_C;

    p_mod->drv_ok = bsp_probe_drv8353();

    if (p_mod->drv_ok != TRUE)
    {
        p_mod->fault |= BL_MOTOR_FAULT_DRV_COMM;
    }

    /* 确认功率级是关的。dl_init() 里已经做过一次，这里是第二道 ——
     * 初始化函数自己保证自己的初始条件，比依赖调用顺序可靠。 */
    (VOID)dl_motor_enable(FALSE);

    p_mod->state = BL_MOTOR_STATE_IDLE;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_motor_task_1ms
* 功能描述 :   状态机一步
*
*              整体顺序：采样 → 联锁 → 状态处理。这个顺序不能换：
*              联锁用的是本拍的数，状态处理可能据此断电，中间不能插别的。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_motor_task_1ms
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();
    U32                now;
    U32                dt_ms;

    now = bsp_get_tick();

    /* 同一毫秒内被调两次时 dt 会是 0，斜坡就白走了。
     * 按 1 ms 算，让斜坡至少推进一个最小步长，而不是原地踏步。 */
    dt_ms = now - p_mod->last_tick;
    if (dt_ms == 0U)
    {
        dt_ms = 1U;
    }
    p_mod->last_tick = now;

    /* ── 1. 采样 ──────────────────────────────────────────────────────────── */
    _bl_motor_refresh_sample();

    /* ── 2. 联锁 ──────────────────────────────────────────────────────────── */
    _bl_motor_eval_interlocks();

    /* 瞬态项（母线欠压）母线上来就自动清掉，不需要人工复位 */
    if (_bl_motor_transient_ok() == TRUE)
    {
        p_mod->fault &= (U16)(~BL_MOTOR_FAULT_BUS_UV);
    }

    /* ── 3. 故障优先于一切 ────────────────────────────────────────────────────
     * 只在"还在出力"的状态下才需要执行断电动作。IDLE 本来就没电，
     * FAULT 已经收过尾了 —— 反复调 dl_motor_enable(FALSE) 没有坏处但也没意义，
     * 而且会把 dl_motor 内部的状态反复重置，掩盖"是谁先发现故障的"。
     *
     * 待机时成立的锁存故障（DRV 通信失败、霍尔断线之类）不会进 FAULT，
     * 它们只是让 _bl_motor_task_idle 里的联锁检查永远过不去，不允许使能。 */
    if ((_bl_motor_latched_ok() != TRUE)
     && ((p_mod->state == BL_MOTOR_STATE_ARMED) || (p_mod->state == BL_MOTOR_STATE_RUN)))
    {
        _bl_motor_enter_fault();
        return;
    }

    /* ── 4. 状态处理 ──────────────────────────────────────────────────────── */
    switch (p_mod->state)
    {
        case BL_MOTOR_STATE_INIT:
            /* init 里已经把状态推到 IDLE 了，正常不会走到这里。
             * 真走到这里说明有人直接改了 state 变量，退回 IDLE 重新走流程。 */
            p_mod->state = BL_MOTOR_STATE_IDLE;
            break;

        case BL_MOTOR_STATE_IDLE:
            _bl_motor_task_idle(now);
            break;

        case BL_MOTOR_STATE_ARMED:
        case BL_MOTOR_STATE_RUN:
            _bl_motor_task_active(now, dt_ms);
            break;

        case BL_MOTOR_STATE_FAULT:
            /* 什么都不做，等 bl_motor_clear_fault()。
             * 这里故意不自动重试：故障自动恢复 = 故障反复出现，
             * 而且人永远看不到它发生过。 */
            break;

        default:
            p_mod->state = BL_MOTOR_STATE_IDLE;
            break;
    }
}

/********************************************************************************************
* 函数名称 :   bl_motor_request_enable
* 功能描述 :   请求运行 / 停止
*
*              TRUE  只是"把请求挂上去"，真正使能要等 IDLE 里联锁通过。
*              FALSE 会触发软停（斜坡降到 0 再断电），不是立刻断。
*                    想立刻断请直接调 dl_motor_enable(FALSE) —— 但那是绕过 BL 的，
*                    正常路径不该这么用。
*
* 输入参数 :   enable: TRUE = 请求运行，FALSE = 请求停止
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL（有锁存故障时拒绝）
*********************************************************************************************/
TYPE_RET_E bl_motor_request_enable
(
    IN  BOOL    enable
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (enable != TRUE)
    {
        p_mod->run_request = FALSE;
        return TYPE_RET_SUCCESS;
    }

    if (_bl_motor_latched_ok() != TRUE)
    {
        return TYPE_RET_FAIL;
    }

    p_mod->run_request = TRUE;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_motor_set_vq_v
* 功能描述 :   设定开环 q 轴电压指令
* 输入参数 :   v_q: 电压，V，带符号
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_motor_set_vq_v
(
    IN  F32     v_q
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    /* NaN 会被这两个比较同时放过，所以先单独拦一次 */
    if (!(v_q == v_q))
    {
        p_mod->vq_cmd = 0.0F;
        return;
    }

    if (v_q > BL_MOTOR_VOLTAGE_MAX_V)
    {
        v_q = BL_MOTOR_VOLTAGE_MAX_V;
    }
    else if (v_q < -BL_MOTOR_VOLTAGE_MAX_V)
    {
        v_q = -BL_MOTOR_VOLTAGE_MAX_V;
    }

    p_mod->vq_cmd = v_q;
}

/********************************************************************************************
* 函数名称 :   bl_motor_set_vd_v
* 功能描述 :   设定开环 d 轴电压指令（正常工况保持 0）
* 输入参数 :   v_d: 电压，V
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_motor_set_vd_v
(
    IN  F32     v_d
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (!(v_d == v_d))
    {
        p_mod->vd_cmd = 0.0F;
        return;
    }

    if (v_d > BL_MOTOR_VOLTAGE_MAX_V)
    {
        v_d = BL_MOTOR_VOLTAGE_MAX_V;
    }
    else if (v_d < -BL_MOTOR_VOLTAGE_MAX_V)
    {
        v_d = -BL_MOTOR_VOLTAGE_MAX_V;
    }

    p_mod->vd_cmd = v_d;
}

/********************************************************************************************
* 函数名称 :   bl_motor_set_limits
* 功能描述 :   运行时修改过流 / 过温门限（不用重烧）
* 输入参数 :   phase_oc_ma / temp_mos_max_c / temp_motor_max_c，传 0 表示保持原值
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_motor_set_limits
(
    IN  S32     phase_oc_ma,
    IN  F32     temp_mos_max_c,
    IN  F32     temp_motor_max_c
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (phase_oc_ma > 0)
    {
        p_mod->oc_ma = phase_oc_ma;
    }

    /* 温度门限的合理范围：0 表示不改；负数或超 200 °C 认为是误输入，忽略。
     * 上界取 200 是因为再高就超出 NTC 的可用量程，门限设在那里等于没有保护。 */
    if ((temp_mos_max_c > 0.0F) && (temp_mos_max_c < 200.0F))
    {
        p_mod->temp_mos_max_c = temp_mos_max_c;
    }

    if ((temp_motor_max_c > 0.0F) && (temp_motor_max_c < 200.0F))
    {
        p_mod->temp_motor_max_c = temp_motor_max_c;
    }
}

/********************************************************************************************
* 函数名称 :   bl_motor_clear_fault
* 功能描述 :   清除锁存的故障位
*
*              要求"当前所有联锁都健康"才允许清 —— 否则连按两次就能绕过保护，
*              那是真实故障会被当成误报放过去。
*
*              清完还要显式请求运行才会重新使能；本函数不恢复输出。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_motor_clear_fault
(
    VOID
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (p_mod->state != BL_MOTOR_STATE_FAULT)
    {
        /* 不在故障态就没有可清的；不清 DRV_COMM 这类"待机期也成立"的位，
         * 因为它们反映的是硬件现状，不是历史事件。 */
        if (p_mod->drv_ok != TRUE)
        {
            return TYPE_RET_FAIL;
        }

        return TYPE_RET_SUCCESS;
    }

    /* 先清硬件刹车锁存（TIM1 的 BIF），否则 MOE 永远置不起来 */
    dl_motor_clear_fault();

    /* 重新评估一次联锁，看现在到底还成不成立。
     * 注意这一步必须在 dl_motor_clear_fault() 之后：清之前 HW_BREAK 位一定是置着的，
     * 那时候评估永远过不了，等于这个函数永远返回 FAIL。 */
    _bl_motor_refresh_sample();
    p_mod->fault &= (U16)(~BL_MOTOR_FAULT_LATCHED_MASK);
    _bl_motor_eval_interlocks();

    if (_bl_motor_latched_ok() != TRUE)
    {
        return TYPE_RET_FAIL;
    }

    p_mod->fault       = BL_MOTOR_FAULT_NONE;
    p_mod->vq_applied  = 0.0F;
    p_mod->vd_applied  = 0.0F;
    p_mod->run_request = FALSE;
    p_mod->state       = BL_MOTOR_STATE_IDLE;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_state
* 功能描述 :   读状态机状态
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   BL_MOTOR_STATE_E
*********************************************************************************************/
BL_MOTOR_STATE_E bl_motor_get_state
(
    VOID
)
{
    return _bl_motor_get_module()->state;
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_fault
* 功能描述 :   读故障位
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   位掩码
*********************************************************************************************/
U16 bl_motor_get_fault
(
    VOID
)
{
    return _bl_motor_get_module()->fault;
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_bus_v
* 功能描述 :   读缓存的本拍母线电压
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   V
*********************************************************************************************/
F32 bl_motor_get_bus_v
(
    VOID
)
{
    return _bl_motor_get_module()->bus_v;
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_phase_ma
* 功能描述 :   读本拍某一相电流
* 输入参数 :   phase: 0=U, 1=V, 2=W
* 输出参数 :   无
* 返回值   :   mA，带符号；越界返回 0
*********************************************************************************************/
S32 bl_motor_get_phase_ma
(
    IN  U8      phase
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (phase >= (U8)BL_MOTOR_PHASE_NUM)
    {
        return 0;
    }

    return p_mod->phase_ma[phase];
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_temp_c
* 功能描述 :   读本拍某一路温度
* 输入参数 :   which: 0 = MOS，1 = 电机
* 输出参数 :   无
* 返回值   :   degC；越界返回 DL_SENSE_TEMP_INVALID_C
*********************************************************************************************/
F32 bl_motor_get_temp_c
(
    IN  U8      which
)
{
    BL_MOTOR_MODULE_T *p_mod = _bl_motor_get_module();

    if (which >= 2U)
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    return p_mod->temp_c[which];
}

/********************************************************************************************
* 函数名称 :   bl_motor_get_vq_applied_v
* 功能描述 :   读斜坡之后的实际 q 轴电压
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   V
*********************************************************************************************/
F32 bl_motor_get_vq_applied_v
(
    VOID
)
{
    return _bl_motor_get_module()->vq_applied;
}

#ifdef _cplusplus
}
#endif
