/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file         stm32n6xx_hal_msp.c
  * @brief        This file provides code for the MSP Initialization
  *               and de-Initialization codes.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
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
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN Define */

/* USER CODE END Define */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN Macro */

/* USER CODE END Macro */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* External functions --------------------------------------------------------*/
/* USER CODE BEGIN ExternalFunctions */

/* USER CODE END ExternalFunctions */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */
/**
  * Initializes the Global MSP.
  */
void HAL_MspInit(void)
{

  /* USER CODE BEGIN MspInit 0 */

  /* USER CODE END MspInit 0 */

  /* System interrupt init*/

  HAL_PWREx_EnableVddIO2();

  HAL_PWREx_EnableVddIO3();

  HAL_PWREx_EnableVddIO4();

  HAL_PWREx_EnableVddIO5();

  /* USER CODE BEGIN MspInit 1 */

  /* USER CODE END MspInit 1 */
}

/**
  * @brief CACHEAXI MSP Initialization
  * This function configures the hardware resources used in this example
  * @param hcacheaxi: CACHEAXI handle pointer
  * @retval None
  */
void HAL_CACHEAXI_MspInit(CACHEAXI_HandleTypeDef* hcacheaxi)
{
  if(hcacheaxi->Instance==CACHEAXI)
  {
    /* USER CODE BEGIN CACHEAXI_MspInit 0 */

    /* USER CODE END CACHEAXI_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
    /* USER CODE BEGIN CACHEAXI_MspInit 1 */

    /* USER CODE END CACHEAXI_MspInit 1 */

  }

}

/**
  * @brief CACHEAXI MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param hcacheaxi: CACHEAXI handle pointer
  * @retval None
  */
void HAL_CACHEAXI_MspDeInit(CACHEAXI_HandleTypeDef* hcacheaxi)
{
  if(hcacheaxi->Instance==CACHEAXI)
  {
    /* USER CODE BEGIN CACHEAXI_MspDeInit 0 */

    /* USER CODE END CACHEAXI_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_CACHEAXI_CLK_DISABLE();
    /* USER CODE BEGIN CACHEAXI_MspDeInit 1 */

    /* USER CODE END CACHEAXI_MspDeInit 1 */
  }

}

/**
  * @brief DCMIPP MSP Initialization
  * This function configures the hardware resources used in this example
  * @param hdcmipp: DCMIPP handle pointer
  * @retval None
  */
void HAL_DCMIPP_MspInit(DCMIPP_HandleTypeDef* hdcmipp)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  RAMCFG_HandleTypeDef hramcfg  = {0};
  RIMC_MasterConfig_t RIMC_master = {0};
  if(hdcmipp->Instance==DCMIPP)
  {
    /* USER CODE BEGIN DCMIPP_MspInit 0 */
    /*** Enable peripheral clock ***/
    /* Enable DCMIPP clock and interface clocks */
    __HAL_RCC_DCMIPP_CLK_ENABLE();
    // __HAL_RCC_IC17_CLK_ENABLE();
    // __HAL_RCC_IC18_CLK_ENABLE();

    __HAL_RCC_DCMIPP_FORCE_RESET();
    __HAL_RCC_DCMIPP_RELEASE_RESET();

    /*** Configure the NVIC for DCMIPP ***/
    /* NVIC configuration for DCMIPP transfer complete interrupt */
    HAL_NVIC_SetPriority(DCMIPP_IRQn, 0x07, 0);
    HAL_NVIC_EnableIRQ(DCMIPP_IRQn);

    /*** Enable peripheral clock ***/
    /* Enable CSI clock */
    __HAL_RCC_CSI_CLK_ENABLE();

    __HAL_RCC_CSI_FORCE_RESET();
    __HAL_RCC_CSI_RELEASE_RESET();

    /*** Configure the NVIC for CSI ***/
    /* NVIC configuration for CSI transfer complete interrupt */
    HAL_NVIC_SetPriority(CSI_IRQn, 0x07, 0);
    HAL_NVIC_EnableIRQ(CSI_IRQn);

    /* USER CODE END DCMIPP_MspInit 0 */
    /* USER CODE BEGIN DCMIPP_MspInit 1 */

    /* DCMIPP Clock Config */
    /* Typical PCLK is 333 MHz so the PLL1 is configured to provide this clock */
    /* Configure DCMIPP clock to IC17 with PLL1  */
    /* PLL1_VCO Input = HSI_VALUE/PLLM = 64 Mhz / 4 = 16 */
    /* PLL1_VCO Output = PLL1_VCO Input * PLLN = 16 Mhz * 75 = 1200 */
    /* PLLLCDCLK = PLL1_VCO Output/(PLLP1 * PLLP2) = 1200/4 = 300Mhz */
    /* DCMIPP clock frequency = PLLLCDCLK = 300 Mhz */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_DCMIPP;
    PeriphClkInitStruct.DcmippClockSelection = RCC_DCMIPPCLKSOURCE_IC17;
    PeriphClkInitStruct.ICSelection[RCC_IC17].ClockSelection = RCC_ICCLKSOURCE_PLL1;
    PeriphClkInitStruct.ICSelection[RCC_IC17].ClockDivider = 4;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* CSI PHY clock from PLL1 via IC18 (for 1.6 Gbps) */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_CSI;
    PeriphClkInitStruct.ICSelection[RCC_IC18].ClockSelection = RCC_ICCLKSOURCE_PLL1;
    PeriphClkInitStruct.ICSelection[RCC_IC18].ClockDivider = 60;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* IMX335 Sensor HW Reset */
    /* Enable GPIO clocks */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();

    GPIO_InitTypeDef gpio_init_structure = {0};

    /* Initialize camera NRST pin */
    gpio_init_structure.Pin       = GPIO_PIN_8;
    gpio_init_structure.Pull      = GPIO_NOPULL;
    gpio_init_structure.Mode      = GPIO_MODE_OUTPUT_PP;
    gpio_init_structure.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOC, &gpio_init_structure);

    /* Initialize camera EN pin */
    gpio_init_structure.Pin       = GPIO_PIN_2;
    gpio_init_structure.Pull      = GPIO_NOPULL;
    gpio_init_structure.Mode      = GPIO_MODE_OUTPUT_PP;
    gpio_init_structure.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOD, &gpio_init_structure);

    /* Camera sensor Power-On sequence */
    /* Assert the camera Enable and NRST pins */
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, GPIO_PIN_RESET);
    HAL_Delay(200);   /* NRST and Enable signals asserted during 200ms */

    /* De-assert the camera STANDBY pin (active high) */
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, GPIO_PIN_SET);
    HAL_Delay(3);     /* NRST de-asserted during 3ms */

    /* SRAM3 and SRAM4 memories clock enable */
    LL_MEM_EnableClock(LL_MEM_AXISRAM3);
    LL_MEM_EnableClock(LL_MEM_AXISRAM4);

    /* Power On AXSRAM3 and AXISRAM4 */
    hramcfg.Instance = RAMCFG_SRAM3_AXI;
    HAL_RAMCFG_EnableAXISRAM(&hramcfg);

    hramcfg.Instance = RAMCFG_SRAM4_AXI;
    HAL_RAMCFG_EnableAXISRAM(&hramcfg);

    /* RIF security attributes: allow non-secure access */
    __HAL_RCC_RIFSC_CLK_ENABLE();

    RIMC_master.MasterCID = RIF_CID_1;
    RIMC_master.SecPriv = RIF_ATTRIBUTE_NSEC | RIF_ATTRIBUTE_PRIV;

HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_DCMIPP, &RIMC_master);

/* Slave 同樣標成 Secure + Privileged */
HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_DCMIPP,
                                      RIF_ATTRIBUTE_NSEC | RIF_ATTRIBUTE_PRIV);
    /* USER CODE END DCMIPP_MspInit 1 */

  }

}

/**
  * @brief DCMIPP MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param hdcmipp: DCMIPP handle pointer
  * @retval None
  */
void HAL_DCMIPP_MspDeInit(DCMIPP_HandleTypeDef* hdcmipp)
{
  if(hdcmipp->Instance==DCMIPP)
  {
    /* USER CODE BEGIN DCMIPP_MspDeInit 0 */
    __HAL_RCC_DCMIPP_FORCE_RESET();
    __HAL_RCC_DCMIPP_RELEASE_RESET();

    /* Disable NVIC  for DCMIPP transfer complete interrupt */
    HAL_NVIC_DisableIRQ(DCMIPP_IRQn);

    /* Disable DCMIPP clock */
    __HAL_RCC_DCMIPP_CLK_DISABLE();

    __HAL_RCC_CSI_FORCE_RESET();
    __HAL_RCC_CSI_RELEASE_RESET();

    /* Disable NVIC  for CSI transfer complete interrupt */
    HAL_NVIC_DisableIRQ(CSI_IRQn);

    /* Disable CSI clock */
    __HAL_RCC_CSI_CLK_DISABLE();
    /* USER CODE END DCMIPP_MspDeInit 0 */

    /* USER CODE BEGIN DCMIPP_MspDeInit 1 */

    /* USER CODE END DCMIPP_MspDeInit 1 */
  }

}

/**
  * @brief I2C MSP Initialization
  * This function configures the hardware resources used in this example
  * @param hi2c: I2C handle pointer
  * @retval None
  */
void HAL_I2C_MspInit(I2C_HandleTypeDef* hi2c)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(hi2c->Instance==I2C1)
  {
    /* USER CODE BEGIN I2C1_MspInit 0 */

    /* USER CODE END I2C1_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2C1;
    PeriphClkInitStruct.I2c1ClockSelection = RCC_I2C1CLKSOURCE_PCLK1;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    /**I2C1 GPIO Configuration
    PC1     ------> I2C1_SDA
    PH9     ------> I2C1_SCL
    */
    GPIO_InitStruct.Pin = I2C1_SDA_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(I2C1_SDA_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = I2C1_SCL_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(I2C1_SCL_GPIO_Port, &GPIO_InitStruct);

    /* Peripheral clock enable */
    __HAL_RCC_I2C1_CLK_ENABLE();
    /* USER CODE BEGIN I2C1_MspInit 1 */

    /* USER CODE END I2C1_MspInit 1 */

  }

}

/**
  * @brief I2C MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param hi2c: I2C handle pointer
  * @retval None
  */
void HAL_I2C_MspDeInit(I2C_HandleTypeDef* hi2c)
{
  if(hi2c->Instance==I2C1)
  {
    /* USER CODE BEGIN I2C1_MspDeInit 0 */

    /* USER CODE END I2C1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_I2C1_CLK_DISABLE();

    /**I2C1 GPIO Configuration
    PC1     ------> I2C1_SDA
    PH9     ------> I2C1_SCL
    */
    HAL_GPIO_DeInit(I2C1_SDA_GPIO_Port, I2C1_SDA_Pin);

    HAL_GPIO_DeInit(I2C1_SCL_GPIO_Port, I2C1_SCL_Pin);

    /* USER CODE BEGIN I2C1_MspDeInit 1 */

    /* USER CODE END I2C1_MspDeInit 1 */
  }

}

/**
  * @brief LTDC MSP Initialization
  * This function configures the hardware resources used in this example
  * @param hltdc: LTDC handle pointer
  * @retval None
  */
void HAL_LTDC_MspInit(LTDC_HandleTypeDef* hltdc)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  RIMC_MasterConfig_t RIMC_master = {0};
  if(hltdc->Instance==LTDC)
  {
    /* USER CODE BEGIN LTDC_MspInit 0 */

    /* USER CODE END LTDC_MspInit 0 */

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_LTDC;
  PeriphClkInitStruct.LtdcClockSelection = RCC_LTDCCLKSOURCE_IC16;
  PeriphClkInitStruct.ICSelection[RCC_IC16].ClockSelection = RCC_ICCLKSOURCE_PLL1;
  PeriphClkInitStruct.ICSelection[RCC_IC16].ClockDivider = 48;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* Peripheral clock enable and reset */
    __HAL_RCC_LTDC_CLK_ENABLE();
    __HAL_RCC_LTDC_FORCE_RESET();
    __HAL_RCC_LTDC_RELEASE_RESET();

    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOQ_CLK_ENABLE();
    /**LTDC GPIO Configuration
    PH3     ------> LTDC_B4
    PH6     ------> LTDC_B5
    PD15     ------> LTDC_R2
    PB14     ------> LTDC_HSYNC
    PB13     ------> LTDC_CLK
    PB2     ------> LTDC_B2
    PB15     ------> LTDC_G4
    PE11     ------> LTDC_VSYNC
    PD8     ------> LTDC_R7
    PH4     ------> LTDC_R4
    PD9     ------> LTDC_R1
    PG6     ------> LTDC_B3
    PA1     ------> LTDC_G2
    PB11     ------> LTDC_G6
    PA15(JTDI)     ------> LTDC_R5
    PG15     ------> LTDC_B0
    PG1     ------> LTDC_G1
    PB12     ------> LTDC_G5
    PA7     ------> LTDC_B1
    PG0     ------> LTDC_R0
    PA2     ------> LTDC_B7
    PG12     ------> LTDC_G0
    PB4(NJTRST)     ------> LTDC_R3
    PG8     ------> LTDC_G7
    PA8     ------> LTDC_B6
    PG13     ------> LTDC_DE
    PA0     ------> LTDC_G3
    PG11     ------> LTDC_R6
    */
    GPIO_InitStruct.Pin = LCD_B4_Pin|LCD_B5_Pin|LCD_R4_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_R2_Pin|LCD_R7_Pin|LCD_R1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_HSYNC_Pin|GPIO_PIN_13|LCD_B2_Pin|LCD_G4_Pin
                          |LCD_G6_Pin|LCD_G5_Pin|LCD_R3_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_VSYNC_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(LCD_VSYNC_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_B3_Pin|LCD_B0_Pin|LCD_G1_Pin|LCD_R0_Pin
                          |LCD_G0_Pin|LCd_G7_Pin|LCD_DE_Pin|LCD_R6_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_G2_Pin|LCD_R5_Pin|LCD_B1_Pin|LCD_B7_Pin
                          |GPIO_PIN_8|LCD_G3_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* Backlight and LCD ON/OFF control pins */
    GPIO_InitStruct.Pin = GPIO_PIN_3|GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOQ, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOQ, GPIO_PIN_3, GPIO_PIN_SET); /* LCD ON */
    HAL_GPIO_WritePin(GPIOQ, GPIO_PIN_6, GPIO_PIN_SET); /* Backlight */

    /* USER CODE BEGIN LTDC_MspInit 1 */

    /* USER CODE END LTDC_MspInit 1 */

    RIMC_master.MasterCID = RIF_CID_1;
    RIMC_master.SecPriv = RIF_ATTRIBUTE_NSEC | RIF_ATTRIBUTE_PRIV;

HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC1, &RIMC_master);

/* Slave 同樣標成 Secure + Privileged */
HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL1,
                                      RIF_ATTRIBUTE_NSEC | RIF_ATTRIBUTE_PRIV);

  }

}

/**
  * @brief LTDC MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param hltdc: LTDC handle pointer
  * @retval None
  */
void HAL_LTDC_MspDeInit(LTDC_HandleTypeDef* hltdc)
{
  if(hltdc->Instance==LTDC)
  {
    /* USER CODE BEGIN LTDC_MspDeInit 0 */

    /* USER CODE END LTDC_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_LTDC_CLK_DISABLE();

    /**LTDC GPIO Configuration
    PH3     ------> LTDC_B4
    PH6     ------> LTDC_B5
    PD15     ------> LTDC_R2
    PB14     ------> LTDC_HSYNC
    PB13     ------> LTDC_CLK
    PB2     ------> LTDC_B2
    PB15     ------> LTDC_G4
    PE11     ------> LTDC_VSYNC
    PD8     ------> LTDC_R7
    PH4     ------> LTDC_R4
    PD9     ------> LTDC_R1
    PG6     ------> LTDC_B3
    PA1     ------> LTDC_G2
    PB11     ------> LTDC_G6
    PA15(JTDI)     ------> LTDC_R5
    PG15     ------> LTDC_B0
    PG1     ------> LTDC_G1
    PB12     ------> LTDC_G5
    PA7     ------> LTDC_B1
    PG0     ------> LTDC_R0
    PA2     ------> LTDC_B7
    PG12     ------> LTDC_G0
    PB4(NJTRST)     ------> LTDC_R3
    PG8     ------> LTDC_G7
    PA8     ------> LTDC_B6
    PG13     ------> LTDC_DE
    PA0     ------> LTDC_G3
    PG11     ------> LTDC_R6
    */
    HAL_GPIO_DeInit(GPIOH, LCD_B4_Pin|LCD_B5_Pin|LCD_R4_Pin);

    HAL_GPIO_DeInit(GPIOD, LCD_R2_Pin|LCD_R7_Pin|LCD_R1_Pin);

    HAL_GPIO_DeInit(GPIOB, LCD_HSYNC_Pin|GPIO_PIN_13|LCD_B2_Pin|LCD_G4_Pin
                          |LCD_G6_Pin|LCD_G5_Pin|LCD_R3_Pin);

    HAL_GPIO_DeInit(LCD_VSYNC_GPIO_Port, LCD_VSYNC_Pin);

    HAL_GPIO_DeInit(GPIOG, LCD_B3_Pin|LCD_B0_Pin|LCD_G1_Pin|LCD_R0_Pin
                          |LCD_G0_Pin|LCd_G7_Pin|LCD_DE_Pin|LCD_R6_Pin);

    HAL_GPIO_DeInit(GPIOA, LCD_G2_Pin|LCD_R5_Pin|LCD_B1_Pin|LCD_B7_Pin
                          |GPIO_PIN_8|LCD_G3_Pin);

    /* USER CODE BEGIN LTDC_MspDeInit 1 */

    /* USER CODE END LTDC_MspDeInit 1 */
  }

}

/**
  * @brief RAMCFG MSP Initialization
  * This function configures the hardware resources used in this example
  * @param hramcfg: RAMCFG handle pointer
  * @retval None
  */
void HAL_RAMCFG_MspInit(RAMCFG_HandleTypeDef* hramcfg)
{
    /* USER CODE BEGIN RAMCFG_MspInit 0 */

    /* USER CODE END RAMCFG_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_RAMCFG_CLK_ENABLE();
    /* USER CODE BEGIN RAMCFG_MspInit 1 */

    /* USER CODE END RAMCFG_MspInit 1 */

}

/**
  * @brief RAMCFG MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param hramcfg: RAMCFG handle pointer
  * @retval None
  */
void HAL_RAMCFG_MspDeInit(RAMCFG_HandleTypeDef* hramcfg)
{
    /* USER CODE BEGIN RAMCFG_MspDeInit 0 */

    /* USER CODE END RAMCFG_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_RAMCFG_CLK_DISABLE();
    /* USER CODE BEGIN RAMCFG_MspDeInit 1 */

    /* USER CODE END RAMCFG_MspDeInit 1 */

}

/**
  * @brief UART MSP Initialization
  * This function configures the hardware resources used in this example
  * @param huart: UART handle pointer
  * @retval None
  */
void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(huart->Instance==USART1)
  {
    /* USER CODE BEGIN USART1_MspInit 0 */

    /* USER CODE END USART1_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART1;
    PeriphClkInitStruct.Usart1ClockSelection = RCC_USART1CLKSOURCE_CLKP;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* Peripheral clock enable */
    __HAL_RCC_USART1_CLK_ENABLE();

    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**USART1 GPIO Configuration
    PE5     ------> USART1_TX
    PE6     ------> USART1_RX
    PA11     ------> USART1_CTS
    */
    GPIO_InitStruct.Pin = VCP_TX_Pin|VCP_RX_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_11;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* USER CODE BEGIN USART1_MspInit 1 */

    /* USER CODE END USART1_MspInit 1 */

  }

}

/**
  * @brief UART MSP De-Initialization
  * This function freeze the hardware resources used in this example
  * @param huart: UART handle pointer
  * @retval None
  */
void HAL_UART_MspDeInit(UART_HandleTypeDef* huart)
{
  if(huart->Instance==USART1)
  {
    /* USER CODE BEGIN USART1_MspDeInit 0 */

    /* USER CODE END USART1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART1_CLK_DISABLE();

    /**USART1 GPIO Configuration
    PE5     ------> USART1_TX
    PE6     ------> USART1_RX
    PA11     ------> USART1_CTS
    */
    HAL_GPIO_DeInit(GPIOE, VCP_TX_Pin|VCP_RX_Pin);

    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_11);

    /* USER CODE BEGIN USART1_MspDeInit 1 */

    /* USER CODE END USART1_MspDeInit 1 */
  }

}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */
