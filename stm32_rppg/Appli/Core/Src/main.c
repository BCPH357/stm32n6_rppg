/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "rk050hr18.h"
#include "stm32n6570_discovery_camera.h"
#include "stm32n6570_discovery_bus.h"
#include "camera_display.h"
#include "isp_api.h"          // ISP middleware
#include "isp_core.h"         // ISP core
#include "imx335_E27_isp_param_conf.h"  // ISP parameters
#include "imx335.h"           // IMX335 sensor driver
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#define ENABLE_CAMERA_PIPELINE 1


/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CACHEAXI_HandleTypeDef hcacheaxi;

DCMIPP_HandleTypeDef hdcmipp;

I2C_HandleTypeDef hi2c1;

LTDC_HandleTypeDef hltdc;

RAMCFG_HandleTypeDef hramcfg_SRAM3;
RAMCFG_HandleTypeDef hramcfg_SRAM4;
RAMCFG_HandleTypeDef hramcfg_SRAM5;
RAMCFG_HandleTypeDef hramcfg_SRAM6;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
// ISP handle from BSP driver
extern ISP_HandleTypeDef hcamera_isp;  // Defined in stm32n6570_discovery_camera.c

// IMX335 sensor object (defined here, not in BSP)
IMX335_Object_t IMX335Obj;

// ISP cached values (for Get functions)
static int32_t isp_gain = 0;
static int32_t isp_exposure = 0;

// Forward declarations
static int32_t IMX335_Probe(uint32_t Resolution, uint32_t PixelFormat);
static ISP_StatusTypeDef GetSensorInfoHelper(uint32_t Instance, ISP_SensorInfoTypeDef *SensorInfo);
static ISP_StatusTypeDef SetSensorGainHelper(uint32_t Instance, int32_t Gain);
static ISP_StatusTypeDef GetSensorGainHelper(uint32_t Instance, int32_t *Gain);
static ISP_StatusTypeDef SetSensorExposureHelper(uint32_t Instance, int32_t Exposure);
static ISP_StatusTypeDef GetSensorExposureHelper(uint32_t Instance, int32_t *Exposure);
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_CACHEAXI_Init(void);
static void MX_RAMCFG_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_DCMIPP_App_Init(void);
static void MX_I2C1_App_Init(uint32_t timing);
static void MX_LTDC_Init(void);
static void MX_DCMIPP_Init(void);
static void SystemIsolation_Config(void);
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* Enable I-Cache (keep D-Cache off to avoid framebuffer artifacts) */
  SCB_EnableICache();

  /* HAL init */
  HAL_Init();
  SystemClock_Config();

  /* Peripherals */
  MX_GPIO_Init();
  MX_CACHEAXI_Init();
  MX_RAMCFG_Init();
  MX_USART1_UART_Init();

  printf("\r\n========================================\r\n");
  printf("  STM32N6 rPPG Camera Display Demo\r\n");
  printf("========================================\r\n");
  printf("[Init] Basic peripherals initialized\r\n");

  /* Initialize I2C1 first (may be needed for LCD controller configuration) */
  MX_I2C1_App_Init(0x30C0EDFF);  // Use same fixed timing as old version
  printf("[Init] I2C1 initialized\r\n");

  /* Initialize LTDC (display will show framebuffer later) */
  MX_LTDC_Init();
  printf("[Init] LTDC initialized\r\n");

  /* Initialize X-CUBE-AI (enables AXISRAM clocks including AXISRAM3 for framebuffer!) */
  MX_X_CUBE_AI_Init();
  printf("[Init] X-CUBE-AI initialized (AXISRAM clocks enabled)\r\n");

  /* Configure system isolation (RIF for NPU, DCMIPP, LTDC, and GPIO attributes) */
  /* ⚠️ DISABLED: SystemIsolation_Config() sets RIF_ATTRIBUTE_SEC which blocks CSI PHY!
   * Official DCMIPP_ContinuousMode example does NOT have this function.
   * When IMX335 starts streaming, SEC attribute causes system crash.
   * See: gentle-crunching-finch.md plan for details.
   */
  // SystemIsolation_Config();
  printf("[Init] SystemIsolation_Config SKIPPED (causes CSI PHY block with SEC attribute)\r\n");

  /* Initialize camera display (sets up LTDC layer and enables display) */
  CameraDisplay_Init();
  printf("[Init] Camera display initialized (LTDC layer enabled)\r\n");

  /* --------------------------------------------------------------
   * CAMERA PIPELINE INITIALIZATION (OFFICIAL ORDER)
   * -------------------------------------------------------------- */
#if ENABLE_CAMERA_PIPELINE
  printf("[Init] Starting camera pipeline initialization...\r\n");

  /* =============================================================
   * STEP 1: Initialize DCMIPP (FULL OFFICIAL SEQUENCE REQUIRED!)
   * ============================================================= */
  MX_DCMIPP_Init();     // <== 必須完整移植官方的版本
  printf("[Init] DCMIPP HAL initialized\r\n");

  /* =============================================================
   * STEP 2: Hardware reset for IMX335
   * NOTE: Official DCMIPP_ContinuousMode does NOT call BSP_CAMERA_HwReset()!
   *       It directly calls IMX335_Probe(). Removing this call.
   * ============================================================= */
  /* DISABLED - Official doesn't have this
  if (BSP_CAMERA_HwReset(0) != 0)
  {
    printf("[ERROR] Camera hardware reset failed\r\n");
    Error_Handler();
  }
  printf("[Init] Camera hardware reset completed\r\n");
  */
  printf("[Init] Skipping BSP_CAMERA_HwReset (official doesn't call it)\r\n");

  /* =============================================================
   * STEP 3: Probe & Initialize IMX335 Sensor
   * (MODE_SELECT = 0x00 is written inside IMX335_Init - same as official)
   * ============================================================= */
  if (IMX335_Probe(IMX335_R2592_1944, IMX335_RAW_RGGB10) != IMX335_OK)
  {
    printf("[ERROR] IMX335 sensor probe failed\r\n");
    Error_Handler();
  }
  printf("[Init] IMX335 sensor initialized (streaming started in IMX335_Init)\r\n");

  /* =============================================================
   * STEP 4: Initialize ISP (helpers + IQ table)
   * ============================================================= */
  ISP_AppliHelpersTypeDef appliHelpers;
  appliHelpers.GetSensorInfo     = GetSensorInfoHelper;
  appliHelpers.SetSensorGain     = SetSensorGainHelper;
  appliHelpers.GetSensorGain     = GetSensorGainHelper;
  appliHelpers.SetSensorExposure = SetSensorExposureHelper;
  appliHelpers.GetSensorExposure = GetSensorExposureHelper;
  printf("[ISP DEBUG] Helper ptrs: %p %p %p %p %p\r\n",
         appliHelpers.GetSensorInfo,
         appliHelpers.SetSensorGain,
         appliHelpers.GetSensorGain,
         appliHelpers.SetSensorExposure,
         appliHelpers.GetSensorExposure);

  printf("[DEBUG] Entering ISP_Init...\r\n");
  ISP_StatusTypeDef istat =
      ISP_Init(&hcamera_isp,
               &hdcmipp,
               0,
               &appliHelpers,
               ISP_IQParamCacheInit[0]);   // from imx335_E27_isp_param_conf.h

  if (istat != ISP_OK)
  {
    printf("[ERROR] ISP init failed, status=%d\r\n", (int)istat);
    Error_Handler();
  }
  printf("[Init] ISP initialized\r\n");

  /* =============================================================
   * STEP 5: Start DCMIPP CSI PIPE
   * (MUST happen AFTER ISP_Init)
   * ============================================================= */
  if (HAL_DCMIPP_CSI_PIPE_Start(&hdcmipp,
                                DCMIPP_PIPE1,
                                DCMIPP_VIRTUAL_CHANNEL0,
                                BUFFER_ADDRESS,
                                DCMIPP_MODE_CONTINUOUS) != HAL_OK)
  {
    printf("[ERROR] DCMIPP CSI pipe start failed\r\n");
    Error_Handler();
  }

  /* Ensure frame/vsync/overrun interrupts are enabled */
  __HAL_DCMIPP_ENABLE_IT(&hdcmipp, DCMIPP_IT_PIPE1_FRAME | DCMIPP_IT_PIPE1_VSYNC | DCMIPP_IT_PIPE1_OVR);

  printf("[Init] DCMIPP CSI pipe started\r\n");

  /* =============================================================
   * STEP 6: Start ISP Processing
   * ============================================================= */
  if (ISP_Start(&hcamera_isp) != ISP_OK)
  {
    printf("[ERROR] ISP start failed\r\n");
    Error_Handler();
  }
  printf("[Init] ISP started\r\n");

  /* NOTE: MODE_SELECT is now written inside IMX335_Init() (same as official)
   * Streaming was already started in STEP 3 before ISP/DCMIPP_Start.
   * This matches official DCMIPP_ContinuousMode behavior.
   */

  /* Quick dump of DCMIPP interrupt enable/status right after start */
  printf("[DCMIPP DEBUG] CMCR=0x%08lX CMIER=0x%08lX CMSR2=0x%08lX\r\n",
         (unsigned long)hdcmipp.Instance->CMCR,
         (unsigned long)hdcmipp.Instance->CMIER,
         (unsigned long)hdcmipp.Instance->CMSR2);

#else
  /* Manual mode for debugging without ISP */
  CameraDisplay_Init();
  MX_DCMIPP_App_Init();
  printf("[Init] DCMIPP initialized (manual mode, no ISP)\r\n");
#endif

  printf("[Main] Entering main loop\r\n");

  /* =============================================================
   * MAIN LOOP
   * ============================================================= */
  while (1)
  {
#if ENABLE_CAMERA_PIPELINE
    /* ISP background processing (AE, AWB, etc.) */
    if (ISP_BackgroundProcess(&hcamera_isp) != ISP_OK)
    {
      /* Non-fatal, continue */
    }
#endif

#if ENABLE_CAMERA_PIPELINE
    /* Periodic DCMIPP status dump every ~1s */
    static uint32_t last_dump = 0;
    uint32_t now = HAL_GetTick();
    if ((now - last_dump) >= 1000U)
    {
      last_dump = now;
      uint32_t cmcr  = hdcmipp.Instance->CMCR;
      uint32_t cmier = hdcmipp.Instance->CMIER;
      uint32_t cmsr2 = hdcmipp.Instance->CMSR2;
      uint32_t pstate = hdcmipp.PipeState[DCMIPP_PIPE1];
      printf("[DCMIPP DEBUG] Tick=%lu CMCR=0x%08lX CMIER=0x%08lX CMSR2=0x%08lX Pipe1State=%lu\r\n",
             (unsigned long)now,
             (unsigned long)cmcr,
             (unsigned long)cmier,
             (unsigned long)cmsr2,
             (unsigned long)pstate);
    }
#endif

    /* LTDC continuously displays framebuffer at BUFFER_ADDRESS */
  }
}


/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};

  /** Configure the System Power Supply */
  if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY ) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable HSI
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if(HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Get current CPU/System buses clocks configuration and
 if necessary switch to intermediate HSI clock to ensure target clock can be set
  */
  HAL_RCC_GetClockConfig(&RCC_ClkInitStruct);
  if((RCC_ClkInitStruct.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
     (RCC_ClkInitStruct.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11))
  {
    RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_SYSCLK);
    RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL1.PLLM = 4;
  RCC_OscInitStruct.PLL1.PLLN = 75;
  RCC_OscInitStruct.PLL1.PLLFractional = 0;
  RCC_OscInitStruct.PLL1.PLLP1 = 1;
  RCC_OscInitStruct.PLL1.PLLP2 = 1;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_HCLK
                              |RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1
                              |RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_PCLK5
                              |RCC_CLOCKTYPE_PCLK4;
  RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;
  RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC1Selection.ClockDivider = 2;
  RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC2Selection.ClockDivider = 3;
  RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC6Selection.ClockDivider = 3;
  RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC11Selection.ClockDivider = 3;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CACHEAXI Initialization Function
  * @param None
  * @retval None
  */
static void MX_CACHEAXI_Init(void)
{

  /* USER CODE BEGIN CACHEAXI_Init 0 */

  /* USER CODE END CACHEAXI_Init 0 */

  /* USER CODE BEGIN CACHEAXI_Init 1 */

  /* USER CODE END CACHEAXI_Init 1 */
  hcacheaxi.Instance = CACHEAXI;
  if (HAL_CACHEAXI_Init(&hcacheaxi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CACHEAXI_Init 2 */

  /* USER CODE END CACHEAXI_Init 2 */

}

/**
  * @brief DCMIPP Initialization Function
  * @param None
  * @retval None
  */
static void MX_DCMIPP_App_Init(void)
{

  /* USER CODE BEGIN DCMIPP_Init 0 */
  /* USER CODE END DCMIPP_Init 0 */

  DCMIPP_PipeConfTypeDef pPipeConf = {0};
  DCMIPP_CSI_PIPE_ConfTypeDef pCSIPipeConf = {0};
  DCMIPP_CSI_ConfTypeDef csiconf = {0};
  DCMIPP_DownsizeTypeDef DonwsizeConf ={0};

  /* USER CODE BEGIN DCMIPP_Init 1 */

  /* USER CODE END DCMIPP_Init 1 */
  hdcmipp.Instance = DCMIPP;
  if (HAL_DCMIPP_Init(&hdcmipp) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the CSI */
  csiconf.DataLaneMapping = DCMIPP_CSI_PHYSICAL_DATA_LANES;
  csiconf.NumberOfLanes   = DCMIPP_CSI_TWO_DATA_LANES;
  csiconf.PHYBitrate      = DCMIPP_CSI_PHY_BT_1600;
  if(HAL_DCMIPP_CSI_SetConfig(&hdcmipp, &csiconf) != HAL_OK)
  {
    Error_Handler();
  }
  /* Configure the Virtual Channel 0 */
  if(HAL_DCMIPP_CSI_SetVCConfig(&hdcmipp, DCMIPP_VIRTUAL_CHANNEL0, DCMIPP_CSI_DT_BPP10) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the serial Pipe */
  pCSIPipeConf.DataTypeMode = DCMIPP_DTMODE_DTIDA;
  pCSIPipeConf.DataTypeIDA  = DCMIPP_DT_RAW10;
  pCSIPipeConf.DataTypeIDB  = DCMIPP_DT_RAW10;

  if (HAL_DCMIPP_CSI_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE1, &pCSIPipeConf) != HAL_OK)
  {
    Error_Handler();
  }

  pPipeConf.FrameRate  = DCMIPP_FRAME_RATE_ALL;
  pPipeConf.PixelPackerFormat = DCMIPP_PIXEL_PACKER_FORMAT_RGB565_1;

    /* Set Pitch for Main and Ancillary Pipes */
  pPipeConf.PixelPipePitch  = 1600 ; /* Number of bytes */
  /* Configure Pipe */
  if (HAL_DCMIPP_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE1, &pPipeConf) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the downsize */
  DonwsizeConf.HRatio      = 25656;
  DonwsizeConf.VRatio      = 33161;
  DonwsizeConf.HSize       = 800;
  DonwsizeConf.VSize       = 480;
  DonwsizeConf.HDivFactor  = 316;
  DonwsizeConf.VDivFactor  = 253;

  if(HAL_DCMIPP_PIPE_SetDownsizeConfig(&hdcmipp, DCMIPP_PIPE1, &DonwsizeConf) != HAL_OK)
  {
    Error_Handler();
  }
  if(HAL_DCMIPP_PIPE_EnableDownsize(&hdcmipp, DCMIPP_PIPE1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCMIPP_Init 2 */

  /* USER CODE END DCMIPP_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param timing Pre-calculated I2C timing value from BSP driver
  * @retval None
  */
static void MX_I2C1_App_Init(uint32_t timing)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  /* Use timing provided by BSP driver (dynamically calculated) */
  hi2c1.Init.Timing = timing;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/* BSP bus layer expects this signature; use provided timing */
HAL_StatusTypeDef MX_I2C1_Init(I2C_HandleTypeDef *phi2c, uint32_t timing)
{
  /* BSP driver provides dynamically calculated timing - use it! */
  MX_I2C1_App_Init(timing);
  if (phi2c != NULL)
  {
    *phi2c = hi2c1;
  }
  return HAL_OK;
}

/**
  * @brief LTDC Initialization Function
  * @param None
  * @retval None
  */
static void MX_LTDC_Init(void)
{

  /* USER CODE BEGIN LTDC_Init 0 */

  /* USER CODE END LTDC_Init 0 */

  LTDC_LayerCfgTypeDef pLayerCfg = {0};
  LTDC_LayerCfgTypeDef pLayerCfg1 = {0};

  /* USER CODE BEGIN LTDC_Init 1 */

  /* USER CODE END LTDC_Init 1 */
  hltdc.Instance = LTDC;
  hltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;
  hltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
  hltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
  hltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;
  hltdc.Init.HorizontalSync     = RK050HR18_HSYNC - 1;
  hltdc.Init.AccumulatedHBP     = RK050HR18_HSYNC + RK050HR18_HBP - 1;
  hltdc.Init.AccumulatedActiveW = RK050HR18_HSYNC + LCD_WIDTH + RK050HR18_HBP -1;
  hltdc.Init.TotalWidth         = RK050HR18_HSYNC + LCD_WIDTH + RK050HR18_HBP + RK050HR18_HFP - 1;
  hltdc.Init.VerticalSync       = RK050HR18_VSYNC - 1;
  hltdc.Init.AccumulatedVBP     = RK050HR18_VSYNC + RK050HR18_VBP - 1;
  hltdc.Init.AccumulatedActiveH = RK050HR18_VSYNC + LCD_HEIGHT + RK050HR18_VBP -1 ;
  hltdc.Init.TotalHeigh         = RK050HR18_VSYNC + LCD_HEIGHT + RK050HR18_VBP + RK050HR18_VFP - 1;
  hltdc.Init.Backcolor.Blue = 0;
  hltdc.Init.Backcolor.Green = 0xFF; /* green background like FSBL for visibility */
  hltdc.Init.Backcolor.Red = 0;
  if (HAL_LTDC_Init(&hltdc) != HAL_OK)
  {
    Error_Handler();
  }
  pLayerCfg.WindowX0       = 0;
  pLayerCfg.WindowX1       = LCD_WIDTH;
  pLayerCfg.WindowY0       = 0;
  pLayerCfg.WindowY1       = LCD_HEIGHT;
  pLayerCfg.PixelFormat    = LTDC_PIXEL_FORMAT_RGB565;
  pLayerCfg.FBStartAdress  = BUFFER_ADDRESS;
  pLayerCfg.Alpha = LTDC_LxCACR_CONSTA;
  pLayerCfg.Alpha0 = 0;
  pLayerCfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
  pLayerCfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
  pLayerCfg.ImageWidth = LCD_WIDTH;
  pLayerCfg.ImageHeight = LCD_HEIGHT;
  pLayerCfg.Backcolor.Blue = 0;
  pLayerCfg.Backcolor.Green = 0;
  pLayerCfg.Backcolor.Red = 0;
  if (HAL_LTDC_ConfigLayer(&hltdc, &pLayerCfg, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* Do not enable second layer to avoid unintended overlay */
  /* USER CODE BEGIN LTDC_Init 2 */

  /* USER CODE END LTDC_Init 2 */

}

/**
  * @brief RAMCFG Initialization Function
  * @param None
  * @retval None
  */
static void MX_RAMCFG_Init(void)
{

  /* USER CODE BEGIN RAMCFG_Init 0 */

  /* USER CODE END RAMCFG_Init 0 */

  /* USER CODE BEGIN RAMCFG_Init 1 */

  /* USER CODE END RAMCFG_Init 1 */

  /** Initialize RAMCFG SRAM3
  */
  hramcfg_SRAM3.Instance = RAMCFG_SRAM3_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM4
  */
  hramcfg_SRAM4.Instance = RAMCFG_SRAM4_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM5
  */
  hramcfg_SRAM5.Instance = RAMCFG_SRAM5_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM6
  */
  hramcfg_SRAM6.Instance = RAMCFG_SRAM6_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RAMCFG_Init 2 */

  /* USER CODE END RAMCFG_Init 2 */

}

/**
  * @brief RIF Initialization Function
  * @param None
  * @retval None
  */
 static void SystemIsolation_Config(void)
{

  /* USER CODE BEGIN RIF_Init 0 */
  __HAL_RCC_RIFSC_CLK_ENABLE();
	RIMC_MasterConfig_t RIMC_master = {0};
	RIMC_master.MasterCID = RIF_CID_1;
	RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;  // Changed back to SEC like old version

	HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &RIMC_master);
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU, RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
  /* USER CODE END RIF_Init 0 */

  /* set all required IPs as secure privileged */
  __HAL_RCC_RIFSC_CLK_ENABLE();

  /*RIMC configuration - use RIMC_master from USER CODE above */
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_DCMIPP, &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC1, &RIMC_master);

  /* RIF-Aware IPs Config */

  /* set up GPIO configuration */
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOF,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);

  /* USER CODE BEGIN RIF_Init 1 */

  /* USER CODE END RIF_Init 1 */
  /* USER CODE BEGIN RIF_Init 2 */

  /* USER CODE END RIF_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(EN_MODULE_GPIO_Port, EN_MODULE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  /* NRST_CAM is active low: GPIO_PIN_RESET releases reset, GPIO_PIN_SET asserts reset */
  HAL_GPIO_WritePin(NRST_CAM_GPIO_Port, NRST_CAM_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_10, GPIO_PIN_RESET);

  /*Configure GPIO pin : EN_MODULE_Pin */
  GPIO_InitStruct.Pin = EN_MODULE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(EN_MODULE_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : NRST_CAM_Pin */
  GPIO_InitStruct.Pin = NRST_CAM_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(NRST_CAM_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PG10 */
  GPIO_InitStruct.Pin = GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
__weak PUTCHAR_PROTOTYPE
{
  HAL_UART_Transmit(&huart1, (uint8_t *)ch, 1, 0xFFFF);
  return ch;
}

__weak int _write(int fd, char * ptr, int len){
  HAL_UART_Transmit(&huart1, (uint8_t *) ptr, len, HAL_MAX_DELAY);
  return len;
}

/**
  * @brief Configure MPU region for framebuffer as non-cacheable
  * @retval None
  */
static void MPU_Config_Framebuffer(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};
  MPU_Attributes_InitTypeDef MPU_Attr = {0};

  HAL_MPU_Disable();

  /* Attributes index 1: normal memory, non-cacheable, inner/outer */
  MPU_Attr.Number = MPU_ATTRIBUTES_NUMBER1;
  MPU_Attr.Attributes = INNER_OUTER(MPU_NOT_CACHEABLE);
  HAL_MPU_ConfigMemoryAttributes(&MPU_Attr);

  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER1;
  MPU_InitStruct.AttributesIndex = MPU_ATTRIBUTES_NUMBER1;
  MPU_InitStruct.BaseAddress = BUFFER_ADDRESS;
  MPU_InitStruct.LimitAddress = BUFFER_ADDRESS + 0xFFFFF; /* 1MB region */
  MPU_InitStruct.AccessPermission = MPU_REGION_ALL_RW;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.DisablePrivExec = MPU_PRIV_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_INNER_SHAREABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/* Camera helper functions removed; BSP driver handles sensor/ISP */
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
/* USER CODE BEGIN 4 */
/**
  * @brief DCMIPP Initialization Function
  * @param None
  * @retval None
  */
static void MX_DCMIPP_Init(void)
{
  /* USER CODE BEGIN DCMIPP_Init 0 */
	 printf(">>> Enter FSBL MX_DCMIPP_Init()\n");
  /* USER CODE END DCMIPP_Init 0 */
  DCMIPP_PipeConfTypeDef pPipeConf = {0};
  DCMIPP_CSI_PIPE_ConfTypeDef pCSIPipeConf = {0};
  DCMIPP_CSI_ConfTypeDef csiconf = {0};
  DCMIPP_DownsizeTypeDef DonwsizeConf ={0};

  /* Set DCMIPP instance */
  hdcmipp.Instance = DCMIPP;
  if (HAL_DCMIPP_Init(&hdcmipp) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the CSI */
  csiconf.DataLaneMapping = DCMIPP_CSI_PHYSICAL_DATA_LANES;
  csiconf.NumberOfLanes   = DCMIPP_CSI_TWO_DATA_LANES;
  csiconf.PHYBitrate      = DCMIPP_CSI_PHY_BT_1600;
  if(HAL_DCMIPP_CSI_SetConfig(&hdcmipp, &csiconf) != HAL_OK)
  {
    Error_Handler();
  }
  /* Configure the Virtual Channel 0 */
  /* Set Virtual Channel config */
  if(HAL_DCMIPP_CSI_SetVCConfig(&hdcmipp, DCMIPP_VIRTUAL_CHANNEL0, DCMIPP_CSI_DT_BPP10) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the serial Pipe */
  pCSIPipeConf.DataTypeMode = DCMIPP_DTMODE_DTIDA;
  pCSIPipeConf.DataTypeIDA  = DCMIPP_DT_RAW10;
  pCSIPipeConf.DataTypeIDB  = DCMIPP_DT_RAW10; /* Don't Care */


  if (HAL_DCMIPP_CSI_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE1, &pCSIPipeConf) != HAL_OK)
  {
    Error_Handler();
  }

  pPipeConf.FrameRate  = DCMIPP_FRAME_RATE_ALL;
  pPipeConf.PixelPackerFormat = DCMIPP_PIXEL_PACKER_FORMAT_RGB565_1;

  /* Set Pitch for Main and Ancillary Pipes */
  pPipeConf.PixelPipePitch  = 1600 ; /* Number of bytes */

  /* Configure Pipe */
  if (HAL_DCMIPP_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE1, &pPipeConf) != HAL_OK)
  {
    Error_Handler();
  }

  /* Configure the downsize */
  DonwsizeConf.HRatio      = 25656;
  DonwsizeConf.VRatio      = 33161;
  DonwsizeConf.HSize       = 800;
  DonwsizeConf.VSize       = 480;
  DonwsizeConf.HDivFactor  = 316;
  DonwsizeConf.VDivFactor  = 253;

  if(HAL_DCMIPP_PIPE_SetDownsizeConfig(&hdcmipp, DCMIPP_PIPE1, &DonwsizeConf) != HAL_OK)
  {
    Error_Handler();
  }
  if(HAL_DCMIPP_PIPE_EnableDownsize(&hdcmipp, DCMIPP_PIPE1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCMIPP_Init 2 */
  /* USER CODE END DCMIPP_Init 2 */
}

/**
  * @brief  Probe and initialize IMX335 sensor
  * @param  Resolution Sensor resolution
  * @param  PixelFormat Pixel format
  * @retval IMX335 status
  */
static int32_t IMX335_Probe(uint32_t Resolution, uint32_t PixelFormat)
{
  IMX335_IO_t IOCtx;
  uint32_t id;

  printf("[IMX335] Configuring I/O context...\r\n");

  /* Configure the camera driver I/O functions */
  IOCtx.Address     = CAMERA_IMX335_ADDRESS;
  IOCtx.Init        = BSP_I2C1_Init;
  IOCtx.DeInit      = BSP_I2C1_DeInit;
  IOCtx.ReadReg     = BSP_I2C1_ReadReg16;
  IOCtx.WriteReg    = BSP_I2C1_WriteReg16;
  IOCtx.GetTick     = BSP_GetTick;

  printf("[IMX335] Registering bus I/O...\r\n");

  /* Register bus I/O */
  if (IMX335_RegisterBusIO(&IMX335Obj, &IOCtx) != IMX335_OK)
  {
    printf("[ERROR] IMX335 RegisterBusIO failed\r\n");
    return IMX335_ERROR;
  }

  printf("[IMX335] Reading sensor ID...\r\n");

  /* Read sensor ID */
  if (IMX335_ReadID(&IMX335Obj, &id) != IMX335_OK)
  {
    printf("[ERROR] IMX335 ReadID failed\r\n");
    return IMX335_ERROR;
  }

  printf("[IMX335] Sensor ID read: 0x%lX (expected 0x%X)\r\n", id, IMX335_CHIP_ID);

  /* Verify sensor ID */
  if (id != (uint32_t)IMX335_CHIP_ID)
  {
    printf("[ERROR] IMX335 chip ID mismatch\r\n");
    return IMX335_ERROR;
  }

  printf("[IMX335] Initializing sensor (resolution: %lu, format: %lu)...\r\n", Resolution, PixelFormat);

  /* Initialize sensor */
  if (IMX335_Init(&IMX335Obj, Resolution, PixelFormat) != IMX335_OK)
  {
    printf("[ERROR] IMX335 Init failed\r\n");
    return IMX335_ERROR;
  }

  printf("[IMX335] Setting frequency to 24MHz...\r\n");

  /* Set input clock frequency */
  if (IMX335_SetFrequency(&IMX335Obj, IMX335_INCK_24MHZ) != IMX335_OK)
  {
    printf("[ERROR] IMX335 SetFrequency failed\r\n");
    return IMX335_ERROR;
  }

/* NOTE: MODE_SELECT is now written inside IMX335_Init() (restored to match official)
   * The streaming is started in IMX335_Init, before ISP_Init/DCMIPP_Start.
   * This is the same behavior as official DCMIPP_ContinuousMode.
   */

  printf("[IMX335] Probe completed successfully\r\n");

  return IMX335_OK;
}

/**
  * @brief  ISP helper: Get sensor info
  */
static ISP_StatusTypeDef GetSensorInfoHelper(uint32_t Instance, ISP_SensorInfoTypeDef *SensorInfo)
{
  (void)Instance;
  return (ISP_StatusTypeDef) IMX335_GetSensorInfo(&IMX335Obj, (IMX335_SensorInfo_t *)SensorInfo);
}

/**
  * @brief  ISP helper: Set sensor gain
  */
static ISP_StatusTypeDef SetSensorGainHelper(uint32_t Instance, int32_t Gain)
{
  (void)Instance;
  isp_gain = Gain;  // Cache the value
  return (ISP_StatusTypeDef) IMX335_SetGain(&IMX335Obj, Gain);
}

/**
  * @brief  ISP helper: Get sensor gain
  */
static ISP_StatusTypeDef GetSensorGainHelper(uint32_t Instance, int32_t *Gain)
{
  (void)Instance;
  *Gain = isp_gain;  // Return cached value
  return ISP_OK;
}

/**
  * @brief  ISP helper: Set sensor exposure
  */
static ISP_StatusTypeDef SetSensorExposureHelper(uint32_t Instance, int32_t Exposure)
{
  (void)Instance;
  isp_exposure = Exposure;  // Cache the value
  return (ISP_StatusTypeDef) IMX335_SetExposure(&IMX335Obj, Exposure);
}

/**
  * @brief  ISP helper: Get sensor exposure
  */
static ISP_StatusTypeDef GetSensorExposureHelper(uint32_t Instance, int32_t *Exposure)
{
  (void)Instance;
  *Exposure = isp_exposure;  // Return cached value
  return ISP_OK;
}



/* USER CODE END 4 */

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
