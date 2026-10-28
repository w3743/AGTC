/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_spi.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 引脚实体定义已上移到 bsp_board.h（全工程唯一来源），这里只是短别名。 */
#define BSP_SPI_CS_PORT                                                 (BSP_BOARD_SPI_CS_PORT)
#define BSP_SPI_CS_PIN                                                  (BSP_BOARD_SPI_CS_PIN)

/* nSCS 高电平保持时间下限 400 ns（手册 §8.5.1.1）。
 * 170 MHz 下一条空循环指令约 6 ns，跑 100 次 ≈ 600 ns，留足余量。
 * 用 volatile 计数器防止编译器把这层循环优化掉。 */
#define BSP_SPI_CS_GUARD_LOOP                                           (100U)


/************************************数据类型定义************************************************/
typedef struct _BSP_SPI_MODULE_T_
{
    SPI_HandleTypeDef   handle;                                         // SPI1 句柄
}BSP_SPI_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_SPI_MODULE_T * _bsp_spi_get_module
(
    VOID
);

static VOID _bsp_spi_cs_guard
(
    VOID
);

static BOOL _bsp_spi_xfer16
(
    IN  U16     tx,
    OUT U16     *p_rx
);


/************************************静态变量定义************************************************/
static BSP_SPI_MODULE_T s_bsp_spi_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_spi_get_module
* 功能描述 :   获取 SPI 模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_SPI_MODULE_T * _bsp_spi_get_module
(
    VOID
)
{
    return &s_bsp_spi_module;
}

/********************************************************************************************
* 函数名称 :   _bsp_spi_cs_guard
* 功能描述 :   片选拉高后的最短间隔
*               DRV8353 要求两次 16 位帧之间 nSCS 至少高 400 ns，否则芯片可能把两帧
*               当成一帧（帧错误，整帧被丢弃）。这里用空循环凑时间，不占用定时器。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bsp_spi_cs_guard
(
    VOID
)
{
    volatile U32 i;

    for (i = 0U; i < BSP_SPI_CS_GUARD_LOOP; i++)
    {
        __NOP();
    }
}

/********************************************************************************************
* 函数名称 :   _bsp_spi_xfer16
* 功能描述 :   收发一帧 16 位（片选由本函数负责拉低/拉高）
*
*               完整时序：
*                 1. nSCS 拉低（此时 SCLK 必须已经是低电平 —— 模式 1 的空闲电平就是低）；
*                 2. 16 个时钟，MSB 先行；
*                 3. nSCS 拉高，SDO 变高阻；
*                 4. 保持 400 ns 以上再发下一帧。
*
*               HAL_SPI_TransmitReceive 的 Size 参数在 16 位数据宽度下是"帧数"，不是字节数。
*
* 输入参数 :   tx: 要发送的 16 位字
* 输出参数 :   p_rx: 收到的 16 位字，不需要时传 NULL
* 返回值   :   TRUE 成功，FALSE 超时或忙
*********************************************************************************************/
static BOOL _bsp_spi_xfer16
(
    IN  U16     tx,
    OUT U16     *p_rx
)
{
    BSP_SPI_MODULE_T *p_mod = _bsp_spi_get_module();
    U16 rx = 0U;
    BOOL ok;

    HAL_GPIO_WritePin(BSP_SPI_CS_PORT, BSP_SPI_CS_PIN, GPIO_PIN_RESET);

    ok = (HAL_SPI_TransmitReceive(&p_mod->handle, (U8 *)&tx, (U8 *)&rx, 1U, 10U) == HAL_OK)
         ? TRUE : FALSE;

    HAL_GPIO_WritePin(BSP_SPI_CS_PORT, BSP_SPI_CS_PIN, GPIO_PIN_SET);
    _bsp_spi_cs_guard();

    if (p_rx != NULL)
    {
        *p_rx = rx;
    }

    return ok;
}

/********************************************************************************************
* 函数名称 :   bsp_spi_init
* 功能描述 :   初始化 SPI1 作为 DRV8353 的配置通道
*
*               配置要点：
*                 - 模式 1（CPOL=0, CPHA=2EDGE）：芯片在下降沿采样、上升沿输出；
*                 - 16 位数据宽度，MSB 先行：DRV8353 的帧就是 16 位；
*                 - 软件片选：NSS 设成 SOFTWARE，PA4 自己当普通 GPIO 拉，这样清起
*                   多个帧的时序比硬件 NSS 更可控；
*                 - 时钟 170/32 = 5.31 MHz：芯片 SCLK 上限 10 MHz（tCLK ≥ 100 ns），
*                   5.3 MHz 留了一倍余量，长线也不会振铃。
*
*               ⚠️ MISO 上板载 4.7k 上拉是必须的：DRV8353 的 SDO 是**开漏**输出，
*                  没有上拉读回来永远是 0xFFF（或全 1），看起来像"芯片没响应"。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_spi_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};
    BSP_SPI_MODULE_T *p_mod = _bsp_spi_get_module();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_SPI1_CLK_ENABLE();

    /* 先写锁存再切模式，保证片选一上电就是高（未选中） */
    HAL_GPIO_WritePin(BSP_SPI_CS_PORT, BSP_SPI_CS_PIN, GPIO_PIN_SET);

    /* PA4：软件片选，推挽输出，空闲高 */
    gpio_init.Pin   = BSP_SPI_CS_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(BSP_SPI_CS_PORT, &gpio_init);

    /* PA5/PA6/PA7：SCK / MISO / MOSI，AF5
     *
     * ⚠️ PA6 在 G431 上除了 SPI1_MISO(AF5)，还是 TIM1_BKIN(AF6)。
     *    本板已经用 PB12 当 TIM1_BKIN，所以 PA6 必须老老实实留在 AF5。
     *    谁要是把 PA6 改成 AF6 想"多接一路刹车"，SPI 会立刻收不到任何数据，
     *    而现象看起来像 DRV8353 坏了 —— 实际是把 MISO 抢走了。 */
    gpio_init.Pin       = BSP_BOARD_SPI_AF_MASK;
    gpio_init.Mode      = GPIO_MODE_AF_PP;
    gpio_init.Pull      = GPIO_NOPULL;          /* MISO 板上已有 4.7k 上拉 */
    gpio_init.Speed     = GPIO_SPEED_FREQ_HIGH;
    gpio_init.Alternate = BSP_BOARD_SPI_AF;
    HAL_GPIO_Init(BSP_BOARD_SPI_SCK_PORT, &gpio_init);

    /* SPI1 主机参数
     * ⚠️ G4 的 SPI_InitTypeDef 只有下面这 13 个字段。H7/U5 那套
     *    NSSPolarity / FifoThreshold / MasterSSIdleness / IOSwap 等在这里不存在，
     *    照抄别的系列会直接编译不过 —— 这份结构体是照着
     *    Drivers/STM32G4xx_HAL_Driver/Inc/stm32g4xx_hal_spi.h 第 48~90 行逐个填的。 */
    p_mod->handle.Instance               = SPI1;
    p_mod->handle.Init.Mode              = SPI_MODE_MASTER;
    p_mod->handle.Init.Direction         = SPI_DIRECTION_2LINES;
    p_mod->handle.Init.DataSize          = SPI_DATASIZE_16BIT;         /* DRV8353 是 16 位帧 */
    p_mod->handle.Init.CLKPolarity       = SPI_POLARITY_LOW;           /* CPOL = 0 */
    p_mod->handle.Init.CLKPhase          = SPI_PHASE_2EDGE;            /* CPHA = 1 → 模式 1 */
    p_mod->handle.Init.NSS               = SPI_NSS_SOFT;               /* nSCS 由软件用 PA4 控制 */
    p_mod->handle.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;   /* 170/32 ≈ 5.3 MHz */
    p_mod->handle.Init.FirstBit          = SPI_FIRSTBIT_MSB;           /* 高字节在前 */
    p_mod->handle.Init.TIMode            = SPI_TIMODE_DISABLE;
    p_mod->handle.Init.CRCCalculation    = SPI_CRCCALCULATION_DISABLE; /* DRV8353 不用 SPI CRC */
    p_mod->handle.Init.CRCPolynomial     = 7;                          /* CRC 关掉也要填合法值 */
    p_mod->handle.Init.CRCLength         = SPI_CRC_LENGTH_16BIT;
    p_mod->handle.Init.NSSPMode          = SPI_NSS_PULSE_DISABLE;      /* 不能有 NSS 脉冲，会多出时钟 */

    (VOID)HAL_SPI_Init(&p_mod->handle);
}

/********************************************************************************************
* 函数名称 :   bsp_drv8353_write_reg
* 功能描述 :   写 DRV8353 的一个寄存器字段
*
*               帧格式：B15=0（写），B14~B11=地址，B10~B0=数据。
*               写的时候 SDO 会同时把该寄存器**原来的值**移出来，所以写操作也能顺便
*               验证"刚才那一帧地址对不对"—— 调试时可以把 rx 存下来看。
*
* 输入参数 :   addr: 0x0 ~ 0x7
*               data: 11 位数据，超出部分被截断
* 输出参数 :   无
* 返回值   :   TRUE 成功
*********************************************************************************************/
BOOL bsp_drv8353_write_reg
(
    IN  U8      addr,
    IN  U16     data
)
{
    U16 frame;

    if (addr > BSP_DRV8353_REG_CAL_MODE)
    {
        return FALSE;
    }

    frame = (U16)(((U16)(addr & 0x0FU) << 11) | (data & 0x07FFU));

    return _bsp_spi_xfer16(frame, NULL);
}

/********************************************************************************************
* 函数名称 :   bsp_drv8353_read_reg
* 功能描述 :   读 DRV8353 的一个寄存器
*
*               帧格式：B15=1（读），B14~B11=地址，B10~B0 随便填。
*               返回帧的高 5 位是无关位，真正数据在低 11 位，所以要 & 0x07FF。
*
*               ⚠️ 读回来的值是"上一帧写进去的地址"，这是 DRV8353 的流水线特性：
*                  SDO 在片选拉低后要先移出 5 个无关位，才开始移目标寄存器内容。
*                  本函数一次调用只发一帧，读到的就是刚请求的那个寄存器（芯片在
*                  同一帧内已经把该地址的内容放到 SDO 上），不需要多发一帧。
*
* 输入参数 :   addr: 0x0 ~ 0x7
* 输出参数 :   p_data: 11 位寄存器值
* 返回值   :   TRUE 成功
*********************************************************************************************/
BOOL bsp_drv8353_read_reg
(
    IN  U8      addr,
    OUT U16     *p_data
)
{
    U16 frame;
    U16 rx = 0U;

    if ((addr > BSP_DRV8353_REG_CAL_MODE) || (p_data == NULL))
    {
        return FALSE;
    }

    frame = (U16)(0x8000U | ((U16)(addr & 0x0FU) << 11));

    if (_bsp_spi_xfer16(frame, &rx) != TRUE)
    {
        return FALSE;
    }

    *p_data = (U16)(rx & 0x07FFU);

    return TRUE;
}

/********************************************************************************************
* 函数名称 :   bsp_drv8353_clear_faults
* 功能描述 :   清 DRV8353 的故障锁存
*
*               DRV8353 的故障位是锁存的：过流/过温消失以后 nFAULT 仍然是低，
*               必须写 CLR_FLT=1 才释放。这个位写完自动清零，不需要手动抹。
*
*               与 TIM1_BKIN 的关系：nFAULT 拉低会直接触发硬件刹车（MOE 被清）。
*               所以故障恢复的完整流程是：
*                 1) bsp_inverter_break_occurred() 确认是不是刹车了；
*                 2) bsp_drv8353_read_fault() 看是谁报的故障；
*                 3) bsp_drv8353_clear_faults() 清锁存；
*                 4) bsp_inverter_clear_break() 清 TIM1 标志；
*                 5) bsp_inverter_start() 重新放行 —— 顺序不能反。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE 成功
*********************************************************************************************/
BOOL bsp_drv8353_clear_faults
(
    VOID
)
{
    U16 ctrl = 0U;

    /* 先读回 Driver Control，保留其它位（尤其是 PWM_MODE），只置 CLR_FLT */
    if (bsp_drv8353_read_reg(BSP_DRV8353_REG_DRV_CTRL, &ctrl) != TRUE)
    {
        return FALSE;
    }

    ctrl |= BSP_DRV8353_CTRL_CLR_FLT;

    return bsp_drv8353_write_reg(BSP_DRV8353_REG_DRV_CTRL, ctrl);
}

/********************************************************************************************
* 函数名称 :   bsp_drv8353_read_fault
* 功能描述 :   读故障状态寄存器
*               Fault Status 1 里每一位对应一种故障，位定义见 bsp_spi.h 的 BSP_DRV8353_FLT1_*。
*               VGS Status 2 记录栅极相关的异常（VGS_Hx/VGS_Lx 欠压、GDUV、OTW 过温预警）。
*
*               注意：读这两个寄存器不会清故障，必须显式写 CLR_FLT。
*
* 输入参数 :   无
* 输出参数 :   p_fault1: Fault Status 1 原始值
*               p_vgs2  : VGS Status 2 原始值，读失败时为 0
* 返回值   :   TRUE = Fault Status 1 读取成功
*********************************************************************************************/
BOOL bsp_drv8353_read_fault
(
    OUT U16     *p_fault1,
    OUT U16     *p_vgs2
)
{
    U16 f1 = 0U;
    U16 v2 = 0U;
    BOOL ok;

    if ((p_fault1 == NULL) || (p_vgs2 == NULL))
    {
        return FALSE;
    }

    ok = bsp_drv8353_read_reg(BSP_DRV8353_REG_FAULT1, &f1);

    (VOID)bsp_drv8353_read_reg(BSP_DRV8353_REG_VGS2, &v2);

    *p_fault1 = f1;
    *p_vgs2   = v2;

    return ok;
}

#ifdef _cplusplus
}
#endif
