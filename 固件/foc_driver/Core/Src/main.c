/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "fdcan.h"
#include "spi.h"
#include "tim.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "bsp.h"
#include "dl.h"
#include "bl.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/********************************************************************************************
 * 上板自测开关 —— 默认 0，**改 1 重烧才会动电机**
 *
 * 为什么要有这一段：本工程目前没有可用的 CAN 指令解析（整车协议表未冻结），
 * 也没有速度环。刷进去之后状态机停在 IDLE、功率级断电，什么都不会发生 ——
 * 这对"确认板子活着"是好事，但也就没法验证相序、方向、电流采样。
 * 这段临时代码提供一个最小可用的激励：延时 → 使能 → 电压斜坡 → 保持 → 软停。
 *
 * ── 打开之前必须确认的事（每一条都会烧板子）──────────────────────────────────
 *   1. 母线电压 ≥ 30 V（BL_MOTOR_BUS_MIN_V），否则联锁挡住，永远进不了 ARMED。
 *      台面上用 24 V 电源试的话，先把 bl_motor.h 里的 BL_MOTOR_BUS_MIN_V 改小。
 *   2. 电机已经接好、转子能自由转（不要带负载、不要卡住）。
 *   3. 电流采样零点对：DRV8353 的 CSA 输出在零电流时应该是 VREF/2。
 *      用万用表量 SO_U/SO_V/SO_W 对地，或者看 CAN 状态帧里的电流是不是接近 0。
 *   4. 手上的电源能限流，建议先设 2 A。
 *
 * ── 预期现象 ────────────────────────────────────────────────────────────────
 *   2 s 后功率级使能；接下来 4 s 内 q 轴电压从 0 线性加到 BL_DEMO_VQ_MV；
 *   保持 2 s；然后软停。整个过程状态机会走 IDLE → ARMED → RUN → IDLE，
 *   CAN 状态帧（ID 0x180）里的状态字会跟着变。
 *
 *   如果电机不转但状态到了 RUN：先怀疑霍尔表/方向（bl_foc_set_direction）、
 *   再怀疑相序；如果状态一直在 IDLE：看 CAN 状态帧上不来，说明被联锁挡住了，
 *   最常见的是母线欠压或温度读数无效（NTC 参数还是占位值）。
 ********************************************************************************************/
#define BL_DEMO_ENABLE                                                  (0)
#define BL_DEMO_DELAY_MS                                                (2000U)
#define BL_DEMO_RAMP_MS                                                 (4000U)
#define BL_DEMO_HOLD_MS                                                 (2000U)
#define BL_DEMO_VQ_MV                                                   (2000)      /* 2.0 V，先小后大 */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#if (BL_DEMO_ENABLE == 1)
/********************************************************************************************
* 函数名称 :   DemoTask
* 功能描述 :   上板自测激励（见文件顶部 BL_DEMO_ENABLE 的说明）
*
*              用"上电后的毫秒数"当时间轴，不保存额外状态：
*              这条时间轴是单调的，重算一次就够，出问题时也好在调试器里核对。
*
*              电压用**逐拍增量**而不是"按时间算绝对值"，因为 bl_motor 内部本来就有一层
*              斜坡（BL_MOTOR_RAMP_V_PER_S）。这里的斜率要比它慢，才不会被它盖住 ——
*              2 V / 4 s = 0.5 V/s，远小于 20 V/s，所以实际波形由这里决定。
*
* 输入参数 :   now: HAL_GetTick()
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static void DemoTask(U32 now)
{
  U32 elapsed;

  if (now < BL_DEMO_DELAY_MS)
  {
    return;
  }

  elapsed = now - BL_DEMO_DELAY_MS;

  if (elapsed == 0U)
  {
    /* 第一次进入：请求使能。真正的使能发生在 bl_motor 的下一个 1 ms 节拍里，
     * 而且要所有联锁通过 —— 所以这里返回 FAIL 是正常现象，不代表出错。 */
    (VOID)bl_motor_request_enable(TRUE);
  }

  if (elapsed < BL_DEMO_RAMP_MS)
  {
    /* 斜坡段：0 → BL_DEMO_VQ_MV，4 s 走完 */
    bl_motor_set_vq_v(((F32)BL_DEMO_VQ_MV / 1000.0F)
                    * ((F32)elapsed / (F32)BL_DEMO_RAMP_MS));
  }
  else if (elapsed < (BL_DEMO_RAMP_MS + BL_DEMO_HOLD_MS))
  {
    /* 保持段 */
    bl_motor_set_vq_v((F32)BL_DEMO_VQ_MV / 1000.0F);
  }
  else if (elapsed < (BL_DEMO_RAMP_MS + BL_DEMO_HOLD_MS + 100U))
  {
    /* 收尾：先清电压指令，再请求停止（软停由 bl_motor 自己做） */
    bl_motor_set_vq_v(0.0F);
    (VOID)bl_motor_request_enable(FALSE);
  }
  else
  {
    /* 做完就停在这里，不循环 —— 自测不该反复启停 */
  }
}
#endif
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_FDCAN1_Init();
  MX_SPI1_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */
  /* 三层初始化，顺序不能换：
   *   1. bsp_init() —— 配引脚、配外设，把板子留在"静默但已就绪"。
   *                    内部顺序见 bsp.c：先把 DRV 按进睡眠、让 CAN 收发器待机，
   *                    再起 PWM/ADC/CAN。DRV8353 探测失败时返回 FAIL，
   *                    但其余模块照常初始化完毕（坏板仍可看霍尔、走 CAN 诊断）。
   *   2. dl_init()  —— 清测量缓存、确认功率级关断。不启动任何外设。
   *   3. bl_init()  —— 注册 CAN 回调、接入总线、起电机策略层（IDLE，功率级仍关断）。
   *
   * 三层都返回成功也不代表电机会动：bl_init() 结束时状态是 IDLE，
   * 功率级是断电的。要动必须显式 bl_motor_request_enable(TRUE)。 */
  (VOID)bsp_init();
  (VOID)dl_init();
  (VOID)bl_init();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  U32 last_tick = HAL_GetTick();

  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    /* 1 ms 控制节拍。
     *
     * ⚠️ 为什么在主循环里按 tick 对齐，而不是放进 SysTick_Handler：
     *    bl_task_handler() 最终会走到 bsp_power_stage_enable()，那个函数用
     *    HAL_Delay() 轮询 DRV8353 的故障位。HAL_Delay() 依赖 SysTick 递增，
     *    如果本函数本身就在 SysTick 中断里，HAL_Delay() 会等一个永远不会推进的
     *    计数器 —— 直接死锁。这是稳压工程踩过的同一个坑。
     *
     * 用 (now - last) >= 1 而不是 == 1，是为了让主循环偶尔被卡住时能追上，
     * 而不是漏掉一整拍之后永远错位。 */
    if ((HAL_GetTick() - last_tick) >= 1U)
    {
      last_tick = HAL_GetTick();
      bl_task_handler();

#if (BL_DEMO_ENABLE == 1)
      DemoTask(HAL_GetTick());
#endif
    }

    bl_task_schedule();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV4;
  RCC_OscInitStruct.PLL.PLLN = 85;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
