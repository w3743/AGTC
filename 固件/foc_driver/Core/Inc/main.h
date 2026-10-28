/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32g4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define SO_W_ADC_Pin GPIO_PIN_0
#define SO_W_ADC_GPIO_Port GPIOA
#define SO_U_ADC_Pin GPIO_PIN_1
#define SO_U_ADC_GPIO_Port GPIOA
#define TEMP_MOS_ADC_Pin GPIO_PIN_2
#define TEMP_MOS_ADC_GPIO_Port GPIOA
#define SO_V_ADC_Pin GPIO_PIN_3
#define SO_V_ADC_GPIO_Port GPIOA
#define DRV_SPI_CS_N_Pin GPIO_PIN_4
#define DRV_SPI_CS_N_GPIO_Port GPIOA
#define BUS_ADC_Pin GPIO_PIN_4
#define BUS_ADC_GPIO_Port GPIOC
#define SAFE_ADC_Pin GPIO_PIN_0
#define SAFE_ADC_GPIO_Port GPIOB
#define TEMP_MOT_ADC_Pin GPIO_PIN_1
#define TEMP_MOT_ADC_GPIO_Port GPIOB
#define HALL_C_Pin GPIO_PIN_10
#define HALL_C_GPIO_Port GPIOB
#define DRV_ENABLE_Pin GPIO_PIN_11
#define DRV_ENABLE_GPIO_Port GPIOB
#define BREAK_N_Pin GPIO_PIN_12
#define BREAK_N_GPIO_Port GPIOB
#define HALL_A_Pin GPIO_PIN_15
#define HALL_A_GPIO_Port GPIOA
#define HALL_B_Pin GPIO_PIN_3
#define HALL_B_GPIO_Port GPIOB
#define CAN_STB_Pin GPIO_PIN_5
#define CAN_STB_GPIO_Port GPIOB
#define DUMP_THERM_OK_Pin GPIO_PIN_9
#define DUMP_THERM_OK_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
