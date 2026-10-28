/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_gpio.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 引脚实体定义已上移到 bsp_board.h（全工程唯一来源）。这里只保留本模块的短别名，
 * 好处是函数体不用改、读起来短，而改板时仍然只需要动 bsp_board.h 一个文件。
 *
 * 依据：J_MCU_L = 核心板 P1，J_MCU_R = 核心板 P2；
 *       water_car_motor 板走的是 J_MCU_A/J_MCU_B，命名不同但同为 P1/P2，脚号一一对应。 */
#define BSP_GPIO_DRV_EN_PORT                                            (BSP_BOARD_DRV_EN_PORT)
#define BSP_GPIO_DRV_EN_PIN                                             (BSP_BOARD_DRV_EN_PIN)

#define BSP_GPIO_CAN_STB_PORT                                           (BSP_BOARD_CAN_STB_PORT)
#define BSP_GPIO_CAN_STB_PIN                                            (BSP_BOARD_CAN_STB_PIN)

#define BSP_GPIO_THERM_OK_PORT                                          (BSP_BOARD_DUMP_THERM_OK_PORT)
#define BSP_GPIO_THERM_OK_PIN                                           (BSP_BOARD_DUMP_THERM_OK_PIN)

/* BREAK_N 由 bsp_inverter 配成 AF 复用输入；本模块只读 IDR，不改它的模式。 */
#define BSP_GPIO_BREAK_N_PORT                                           (BSP_BOARD_INV_BKIN_PORT)
#define BSP_GPIO_BREAK_N_PIN                                            (BSP_BOARD_INV_BKIN_PIN)


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bsp_gpio_init
* 功能描述 :   初始化本模块负责的 GPIO，并把板子停在安全态
*
*               安全态三件事（顺序无所谓，都是"把功率通路关掉"）：
*                 1. DRV_ENABLE = 低  → DRV8353 的 ENABLE 拉低，六路栅极被强制关断，
*                                        不管 PWM 引脚此刻是什么电平都不会有输出；
*                 2. CAN_STB    = 高  → 收发器待机，总线不驱动（STB 内部还有上拉到 VIO）；
*                 3. BREAK_N 不配置 → 保持复位后的输入态，TIM1 的 BKIN 由 bsp_inverter 接管。
*
*               板上冗余说明（这点很重要，决定了"固件没跑起来时板子是什么状态"）：
*                 R_DRV_EN_PD 10k 下拉 → MCU 复位期间 GPIO 高阻，ENABLE 已被拉低；
*                 R_CAN_STB_PU 10k 上拉 → 复位期间 STB 已为高，收发器已在待机。
*               也就是说：即使固件一行没执行，功率级也不会动。本函数只是把这两个电平
*               用推挽输出"钉死"，避免长线或干扰把它抬起来。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};

    /* GPIOB 上集中了 ENABLE / STB / THERM_OK / BREAK_N 四个脚 */
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* 先把输出锁存写成安全电平，再切输出模式，避免切换瞬间出现毛刺。
     * 注意：HAL_GPIO_WritePin 写的是 ODR，引脚还是输入时也会记住这个值，
     *       一旦切成输出就立刻呈现该电平，所以这一句必须在前。 */
    HAL_GPIO_WritePin(BSP_GPIO_DRV_EN_PORT,  BSP_GPIO_DRV_EN_PIN,  GPIO_PIN_RESET);
    HAL_GPIO_WritePin(BSP_GPIO_CAN_STB_PORT, BSP_GPIO_CAN_STB_PIN, GPIO_PIN_SET);

    /* DRV_ENABLE：推挽输出，低速（只在启停时翻转一次，不快） */
    gpio_init.Pin   = BSP_GPIO_DRV_EN_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;      /* 板上已有 10k 下拉，不需要内部再并一个 */
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_DRV_EN_PORT, &gpio_init);

    /* CAN_STB：推挽输出，低速 */
    gpio_init.Pin   = BSP_GPIO_CAN_STB_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;      /* 板上已有 10k 上拉 */
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_CAN_STB_PORT, &gpio_init);

    /* DUMP_THERM_OK：普通输入。
     * 信号从比较器 U4 的推挽输出出来，经 100k 串联到 MCU，属于"慢边沿"信号，
     * 这里不加内部上/下拉：加了只会和比较器对着拉，反而让电平不确定。
     * 100k 串联的作用是限流保护，不影响读取（输入电流只有 nA 级）。 */
    gpio_init.Pin  = BSP_GPIO_THERM_OK_PIN;
    gpio_init.Mode = GPIO_MODE_INPUT;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_GPIO_THERM_OK_PORT, &gpio_init);

    /* BREAK_N（PB12）故意不在这里初始化：
     * 它要作为 TIM1_BKIN 复用输入，AF 配置归 bsp_inverter 管，两处都配会互相覆盖。
     * bsp_inverter_init() 之前读它，PB12 是复位后的输入态，同样能读到真实电平。 */
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_drv_set_enable
* 功能描述 :   控制 DRV8353 的 ENABLE（芯片手册里就是叫 ENABLE，不是 nENABLE，高有效）
*
*               与 PWM 的关系：
*                 ENABLE=0 → 六路栅极全部关断，OUT 高阻，电机自由滑行；
*                 ENABLE=1 → 栅极听 PWM 的。
*               所以 ENABLE 是比 PWM 更"硬"的一道闸，急停优先用它，而不是把占空比写 0
*               —— 写占空比 0 在 PWM 模式下是"持续导通下桥"，属于电制动，不是断电。
*
* 输入参数 :   level: TRUE = 放行栅极驱动，FALSE = 强制关断
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_drv_set_enable
(
    IN  BOOL    level
)
{
    HAL_GPIO_WritePin(BSP_GPIO_DRV_EN_PORT, BSP_GPIO_DRV_EN_PIN,
                      (level == TRUE) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_can_set_stb
* 功能描述 :   控制 TCAN1042 的 STB（standby，高有效）
*               待机时收发器只留低功耗接收机监测总线，RXD 只在检测到唤醒图样时动作，
*               也就是说：待机状态下收不到正常报文，正常通信前必须先拉低 STB。
* 输入参数 :   level: TRUE = 待机，FALSE = 正常通信
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_can_set_stb
(
    IN  BOOL    level
)
{
    HAL_GPIO_WritePin(BSP_GPIO_CAN_STB_PORT, BSP_GPIO_CAN_STB_PIN,
                      (level == TRUE) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_get_dump_therm_ok
* 功能描述 :   读泄放电阻的热许可信号
*               本板没有能耗散母线回馈能量的负载，多余能量靠 Q7 + R_DUMP_A/B 烧掉；
*               这个脚告诉 MCU "现在允许烧"，比较器判据由 R_THERM_TOP/BOT 分压决定。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 允许泄放，FALSE = 不允许
*********************************************************************************************/
BOOL bsp_gpio_get_dump_therm_ok
(
    VOID
)
{
    return (HAL_GPIO_ReadPin(BSP_GPIO_THERM_OK_PORT, BSP_GPIO_THERM_OK_PIN) == GPIO_PIN_SET)
           ? TRUE : FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_read_break_n
* 功能描述 :   读 BREAK_N 网络的原始电平（只做诊断用）
*
*               这条线上挂了三样东西：
*                 U1.28  DRV8353 nFAULT（开漏，故障时拉低）
*                 U3.1/2/14 三路比较器输出（相电压低于负阈值时拉低）
*                 TIM1_BKIN（硬件刹车输入）
*               线与关系 + R_BREAK_PU 2.2k 上拉 → 任何一个报故障，整条线就是低。
*
*               因为 BKIN 已经在硬件上把 PWM 关掉了，所以本函数返回 FALSE 时，
*               功率级其实已经被切断；读它只是为了知道"刚才发生过"，以及清标志后
*               判断故障是否消失。
*
*               ⚠️ 电平为低只说明"有人拉低"，不能区分是 DRV 报故障还是比较器动作；
*                  要区分必须读 DRV8353 的 Fault Status 1/2 寄存器（见 bsp_spi）。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 高电平，无故障；FALSE = 低电平，有故障
*********************************************************************************************/
BOOL bsp_gpio_read_break_n
(
    VOID
)
{
    /* 注意：PB12 在 bsp_inverter_init() 之后是 AF 复用态，但 IDR 仍然反映引脚真实电平，
     * HAL_GPIO_ReadPin 读的就是 IDR，所以复用态下照样能读，不需要切回输入。 */
    return (HAL_GPIO_ReadPin(BSP_GPIO_BREAK_N_PORT, BSP_GPIO_BREAK_N_PIN) == GPIO_PIN_SET)
           ? TRUE : FALSE;
}

#ifdef _cplusplus
}
#endif
