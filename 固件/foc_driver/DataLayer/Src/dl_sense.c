/************************************包含头文件*************************************************/
#include <stddef.h>
#include <math.h>
#include "type.h"
#include "bsp_adc.h"
#include "bsp_board.h"
#include "dl_sense.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* NTC 判据的边界（mV）。
 *
 * 开路时引脚被上拉到 VREF，短路时被拉到 0。留 20 mV 的余量是因为
 *   · ADC 自身有 ±4 LSB 左右的失调与噪声；
 *   · VREF 与 "3.3 V" 之间有几十 mV 的差。
 * 20 mV ≈ 25 LSB，足够把"真的到了端点"和"读数偏一点"分开。
 *
 * 为什么要在端点直接返回无效值而不是硬算：
 *   R = R_pull × V / (VREF − V)，V 越接近 VREF，分母越小，算出来的 R 会冲到几百 kΩ，
 *   按 B 值公式反推的温度会跑到 -100 °C 以下 —— 那是个"看起来像数"的假数，
 *   上层拿它做判断会得出荒谬结论。宁可给一个明显的哨兵值。 */
#define DL_SENSE_ADC_RAIL_MARGIN_MV                                     (20U)

/* 绝对零度（degC），用于把开尔文转摄氏 */
#define DL_SENSE_KELVIN_OFFSET                                          (273.15F)

/* 三相电流索引，只为代码好读；数值必须与 dl_sense.h 的 DL_SENSE_PHASE_E 一致 */
#define DL_SENSE_PHASE_IDX_U                                            (0U)
#define DL_SENSE_PHASE_IDX_V                                            (1U)
#define DL_SENSE_PHASE_IDX_W                                            (2U)
#define DL_SENSE_PHASE_IDX_NUM                                           (3U)


/************************************数据类型定义************************************************/
typedef struct _DL_SENSE_MODULE_T_
{
    S32                 phase_ma[DL_SENSE_PHASE_IDX_NUM];               /* 三相电流，带符号，mA */
    F32                 bus_v;                                          /* 母线电压，V，已滤波 */
    F32                 temp_c[DL_SENSE_TEMP_NUM];                      /* 两路 NTC，degC */
    BOOL                bus_seeded;                                     /* 母线滤波器是否已播种 */
    BOOL                phase_valid;                                    /* 注入组是否出过数 */
    DL_SENSE_STATUS_E   status;                                         /* 采样链路状态 */
}DL_SENSE_MODULE_T;


/************************************静态函数声明************************************************/
static DL_SENSE_MODULE_T * _dl_sense_get_module
(
    VOID
);

static F32 _dl_sense_mv_to_c
(
    IN  U32     pin_mv
);

static F32 _dl_sense_raw_to_c
(
    IN  U16     raw
);


/************************************静态变量定义************************************************/
static DL_SENSE_MODULE_T s_dl_sense_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _dl_sense_get_module
* 功能描述 :   获取采样模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static DL_SENSE_MODULE_T * _dl_sense_get_module
(
    VOID
)
{
    return &s_dl_sense_module;
}

/********************************************************************************************
* 函数名称 :   _dl_sense_mv_to_c
* 功能描述 :   把 NTC 分压点的电压换算成温度
*
*              ── 电路 ────────────────────────────────────────────────────────────
*               3V3 ── R_pull(10k) ──┬── ADC 引脚
*                                     └── NTC ── GND
*
*              所以 V_pin = VREF × R_ntc / (R_pull + R_ntc)，反解得
*                R_ntc = R_pull × V / (VREF − V)
*
*              ── B 值公式 ────────────────────────────────────────────────────────
*               1/T = 1/T25 + ln(R/R25) / B
*               T 是开尔文，最后减 273.15 转摄氏。
*               B 值公式是单点标定的近似，在全温区有 1~3 °C 的误差，对"判断有没有
*              过热"够用；要更准就得用 Steinhart-Hart 三系数，那需要三点实测标定。
*
*              用的是 logf 而不是查表：本函数只在 1 ms 节拍里调两次，开销可以忽略，
*              换来的是任意 NTC 参数都能直接改宏就生效，不用重新生成表。
*
* 输入参数 :   pin_mv: 引脚电压，mV
* 输出参数 :   无
* 返回值   :   摄氏温度；读数在电气上不可能时返回 DL_SENSE_TEMP_INVALID_C
*********************************************************************************************/
static F32 _dl_sense_mv_to_c
(
    IN  U32     pin_mv
)
{
    F32 v_pin;
    F32 v_ref;
    F32 r_ntc;
    F32 inv_t;

    v_ref = ((F32)BSP_ADC_VREF_MV) / 1000.0F;

    /* 开路：引脚被上拉到 VREF 顶死 */
    if (pin_mv >= (BSP_ADC_VREF_MV - DL_SENSE_ADC_RAIL_MARGIN_MV))
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    /* 短路：引脚被拉到地 */
    if (pin_mv <= DL_SENSE_ADC_RAIL_MARGIN_MV)
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    v_pin = ((F32)pin_mv) / 1000.0F;

    r_ntc = DL_SENSE_NTC_PULLUP_OHM * v_pin / (v_ref - v_pin);

    if (!(r_ntc > 0.0F))
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    inv_t = (1.0F / DL_SENSE_NTC_NOMINAL_K)
          + (logf(r_ntc / DL_SENSE_NTC_NOMINAL_OHM) / DL_SENSE_NTC_B_COEFFICIENT);

    /* inv_t ≤ 0 意味着算出来的温度低于绝对零度，只可能是参数或读数错了 */
    if (!(inv_t > 0.0F))
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    return (1.0F / inv_t) - DL_SENSE_KELVIN_OFFSET;
}

/********************************************************************************************
* 函数名称 :   _dl_sense_raw_to_c
* 功能描述 :   ADC 原始码 → 温度，两步合成一步
* 输入参数 :   raw: 0 ~ 4095
* 输出参数 :   无
* 返回值   :   摄氏温度
*********************************************************************************************/
static F32 _dl_sense_raw_to_c
(
    IN  U16     raw
)
{
    return _dl_sense_mv_to_c(bsp_adc_raw_to_mv(raw));
}

/********************************************************************************************
* 函数名称 :   dl_sense_init
* 功能描述 :   初始化采样数据层
*
*              只清缓存，不启动任何外设 —— ADC 已经由 bsp_init() 里的 bsp_adc_start()
*              启动并连续转换了。这里如果再去 start 一次，会打断正在跑的注入组同步。
*
*              状态先记 UNCALIBRATED：此刻一帧同步采样都还没拿到，任何读数都不可信。
*              第一次 dl_sense_update() 成功读到注入组后才会转成 OK。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS
*********************************************************************************************/
TYPE_RET_E dl_sense_init
(
    VOID
)
{
    DL_SENSE_MODULE_T *p_mod = _dl_sense_get_module();
    U8                 i;

    for (i = 0U; i < (U8)DL_SENSE_PHASE_IDX_NUM; i++)
    {
        p_mod->phase_ma[i] = 0;
    }

    for (i = 0U; i < (U8)DL_SENSE_TEMP_NUM; i++)
    {
        p_mod->temp_c[i] = DL_SENSE_TEMP_INVALID_C;
    }

    p_mod->bus_v        = 0.0F;
    p_mod->bus_seeded   = FALSE;
    p_mod->phase_valid  = FALSE;
    p_mod->status       = DL_SENSE_STATUS_UNCALIBRATED;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   dl_sense_update
* 功能描述 :   把最新一轮原始采样换算成工程量，刷新整个缓存
*
*              调用位置：每个控制节拍（bl_task_handler）里，在读取任何 getter 之前调一次。
*              调完之后本拍内所有 getter 拿到的都是同一时刻的值 —— 这一点很重要：
*              如果 getter 每次自己去读 ADC，快速变化的相电流会在一次计算里前后不一致，
*              算出来的矢量角就带了一个和调用顺序有关的抖动。
*
*              ── 相电流不做滤波（有意）────────────────────────────────────────────
*              相电流是电流环的反馈量。任何低通都会引入相位滞后，直接吃掉环路相位裕度；
*              而且注入组本来就是"每个 PWM 周期在计数器顶点采一次"，采到的是三角波
*              纹波的中心值，本身已经是干净的。噪声要压的话，应该压采样时刻的抖动，
*              不是在软件里加一阶滞后。
*              母线电压不一样：它变化慢，加滤波只有好处（`DL_SENSE_BUS_LPF_ALPHA`）。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID dl_sense_update
(
    VOID
)
{
    DL_SENSE_MODULE_T *p_mod = _dl_sense_get_module();
    U16                raw_u = 0U;
    U16                raw_v = 0U;
    U16                raw_w = 0U;
    U16                raw_bus;
    F32                bus_inst;

    /* ── 1. 相电流：ADC1 注入组，与 PWM 顶点同步 ────────────────────────────
     * 返回值 TRUE 表示至少完成过一次转换。为 FALSE 时保持上一次的值不动 ——
     * 把缓存清成 0 反而危险：BL 会以为"电流真的没了"。 */
    if (bsp_adc_get_phase_raw(&raw_u, &raw_v, &raw_w) == TRUE)
    {
        p_mod->phase_ma[DL_SENSE_PHASE_IDX_U] = bsp_adc_phase_raw_to_ma(raw_u);
        p_mod->phase_ma[DL_SENSE_PHASE_IDX_V] = bsp_adc_phase_raw_to_ma(raw_v);
        p_mod->phase_ma[DL_SENSE_PHASE_IDX_W] = bsp_adc_phase_raw_to_ma(raw_w);

        p_mod->phase_valid = TRUE;
        p_mod->status      = DL_SENSE_STATUS_OK;
    }

    /* ── 2. 母线电压：ADC2 规则组，连续转换 + DMA ────────────────────────────
     * 分压比在 dl_sense.h 里定义为 (V_pin / V_bus)，所以是除不是乘。
     * 先算引脚电压（V），再除以比值得到母线电压（V）。 */
    raw_bus  = bsp_adc_get_regular_raw(2U, BSP_ADC2_IDX_BUS);
    bus_inst = (((F32)bsp_adc_raw_to_mv(raw_bus)) / 1000.0F) / DL_SENSE_BUS_DIVIDER;

    if (p_mod->bus_seeded == TRUE)
    {
        p_mod->bus_v += DL_SENSE_BUS_LPF_ALPHA * (bus_inst - p_mod->bus_v);
    }
    else
    {
        /* 第一拍直接把瞬时值当作初值，否则滤波器要从 0 慢慢爬上来，
         * 上电后头几十毫秒读到的母线电压都是偏低的，容易误判欠压。 */
        p_mod->bus_v      = bus_inst;
        p_mod->bus_seeded = TRUE;
    }

    /* ── 3. 温度：ADC1 规则组的两路 NTC ────────────────────────────────────
     * 这里刻意写成两条独立语句而不是循环，并且直接用 BSP 的通道下标常量：
     *   DL_SENSE_TEMP_MOS=0 / DL_SENSE_TEMP_MOTOR=1
     *   BSP_ADC1_IDX_TEMP_MOS=0 / BSP_ADC1_IDX_TEMP_MOTOR=1
     * 两组枚举**碰巧**同序，如果写成 `for (i) ... raw(1, i)`，哪天 BSP 侧插一个
     * 通道进规则组把顺序换了，编译照过、温度对着换 —— 这种错最难查。 */
    p_mod->temp_c[DL_SENSE_TEMP_MOS] =
        _dl_sense_raw_to_c(bsp_adc_get_regular_raw(1U, BSP_ADC1_IDX_TEMP_MOS));

    p_mod->temp_c[DL_SENSE_TEMP_MOTOR] =
        _dl_sense_raw_to_c(bsp_adc_get_regular_raw(1U, BSP_ADC1_IDX_TEMP_MOTOR));
}

/********************************************************************************************
* 函数名称 :   dl_sense_get_phase_current_ma
* 功能描述 :   读某一相电流，带符号，毫安
*
*              符号约定见 dl_sense.h：正值 = 电流从桥臂输出端流向低边采样电阻。
*              换算系数（0.5 mΩ × 20 V/V = 10 mV/A、零点 VREF/2）全部在
*              bsp_adc_phase_raw_to_ma() 里，本层不重复算 —— 只做索引校验。
*
* 输入参数 :   phase: DL_SENSE_PHASE_U / V / W
* 输出参数 :   无
* 返回值   :   带符号电流 mA；phase 越界返回 0
*********************************************************************************************/
S32 dl_sense_get_phase_current_ma
(
    IN  DL_SENSE_PHASE_E    phase
)
{
    DL_SENSE_MODULE_T *p_mod = _dl_sense_get_module();

    if ((U8)phase >= (U8)DL_SENSE_PHASE_IDX_NUM)
    {
        return 0;
    }

    return p_mod->phase_ma[(U8)phase];
}

/********************************************************************************************
* 函数名称 :   dl_sense_get_bus_v
* 功能描述 :   读母线电压（已低通滤波），单位 V
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   母线电压，V
*********************************************************************************************/
F32 dl_sense_get_bus_v
(
    VOID
)
{
    return _dl_sense_get_module()->bus_v;
}

/********************************************************************************************
* 函数名称 :   dl_sense_get_temp_c
* 功能描述 :   读某一路 NTC 温度
*
*              ⚠️ 两路 NTC 的 R25 与 B 值**目前是占位值**（10k / 3950），
*                 见 dl_sense.h 里的 DL_SENSE_NTC_* 宏。实测确认后改那几个宏即可，
*                 算法不用动。在标定之前，温度读数只能当"有没有明显异常"来用，
*                 不能拿去做精确的热模型。
*
* 输入参数 :   which: DL_SENSE_TEMP_MOS / DL_SENSE_TEMP_MOTOR
* 输出参数 :   无
* 返回值   :   摄氏温度；读数不可能时返回 DL_SENSE_TEMP_INVALID_C
*********************************************************************************************/
F32 dl_sense_get_temp_c
(
    IN  DL_SENSE_TEMP_E     which
)
{
    DL_SENSE_MODULE_T *p_mod = _dl_sense_get_module();

    if ((U8)which >= (U8)DL_SENSE_TEMP_NUM)
    {
        return DL_SENSE_TEMP_INVALID_C;
    }

    return p_mod->temp_c[(U8)which];
}

/********************************************************************************************
* 函数名称 :   dl_sense_get_status
* 功能描述 :   读采样链路状态
*
*              ⚠️ 这里的"标定"指**采样链路是否已经建立**（注入组出过数），
*                 不是"NTC 参数是否已用万用表核对过"。后者是编译期常量，
*                 运行期无法判断，只能靠人去看 dl_sense.h 里的注释。
*                 之所以借这个枚举表达"链路未建立"，是因为它在语义上确实属于
*                 "现在的数还不能信"，和"未标定"是一回事。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   DL_SENSE_STATUS_E
*********************************************************************************************/
DL_SENSE_STATUS_E dl_sense_get_status
(
    VOID
)
{
    return _dl_sense_get_module()->status;
}

#ifdef _cplusplus
}
#endif
