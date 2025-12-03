/**
  ******************************************************************************
  * @file    camera_display.c
  * @brief   Camera capture and LCD display implementation
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "camera_display.h"
#include "main.h"
#include <stdio.h>

/* External variables --------------------------------------------------------*/
extern DCMIPP_HandleTypeDef hdcmipp;
extern LTDC_HandleTypeDef hltdc;

/* Private variables ---------------------------------------------------------*/
static volatile uint32_t frame_count = 0;

/**
  * @brief  Initialize camera and display
  * @retval None
  */
void CameraDisplay_Init(void)
{
    printf("[Camera Display] Initializing...\r\n");
    printf("[Camera Display] Buffer address: 0x%08lX\r\n", (unsigned long)BUFFER_ADDRESS);

    /* Fill buffer with test pattern (green color in RGB565) - FULL SCREEN */
    volatile uint16_t *buffer = (volatile uint16_t *)BUFFER_ADDRESS;
    uint32_t pixel_count = LCD_WIDTH * LCD_HEIGHT;  /* 800x480 full screen */

    printf("[Camera Display] Filling buffer with test pattern (800x480, green)...\r\n");
    for (uint32_t i = 0; i < pixel_count; i++)
    {
        buffer[i] = 0x07E0;  /* Green in RGB565 format */
    }

    /* Verify buffer content */
    printf("[Camera Display] Verifying buffer: [0]=0x%04X, [100]=0x%04X, [1000]=0x%04X\r\n",
           buffer[0], buffer[100], buffer[1000]);

    /* Clean D-Cache to ensure data is written to memory */
    SCB_CleanDCache_by_Addr((uint32_t *)BUFFER_ADDRESS, LCD_WIDTH * LCD_HEIGHT * 2);

    printf("[Camera Display] Test pattern ready\r\n");

    /* Debug: Check clocks */
    uint32_t pclk5_freq = HAL_RCC_GetPCLK5Freq();
    uint32_t hclk_freq = HAL_RCC_GetHCLKFreq();
    uint32_t sysclk_freq = HAL_RCC_GetSysClockFreq();
    printf("[Clock Debug] SYSCLK: %lu Hz (expected: 400 MHz)\r\n", sysclk_freq);
    printf("[Clock Debug] HCLK: %lu Hz (expected: 200 MHz)\r\n", hclk_freq);
    printf("[Clock Debug] PCLK5: %lu Hz\r\n", pclk5_freq);
    printf("[Clock Debug] Expected LTDC pixel clock: 25 MHz (from IC16: PLL1/48)\r\n");

    /* Debug: Check LTDC status */
    printf("[LTDC Debug] Instance: 0x%08lX\r\n", (unsigned long)hltdc.Instance);
    printf("[LTDC Debug] State: %d\r\n", hltdc.State);
    if (hltdc.Instance != NULL && hltdc.Instance == LTDC)
    {
        /* Ensure pitch matches 800 pixels in RGB565 */
        uint32_t line_bytes = LCD_WIDTH * 2U;
        /* CFBLL = line_bytes + 3; CFBP = pitch_bytes + 3 */
        uint32_t cfblr = ((line_bytes + 3U) << 16) | (line_bytes + 3U);
        LTDC_Layer1->CFBLR = cfblr;
        LTDC->SRCR = LTDC_SRCR_IMR;

        uint32_t ltdc_gcr = LTDC->GCR;
        printf("[LTDC Debug] GCR: 0x%08lX (LTDCEN=%lu)\r\n", ltdc_gcr, (ltdc_gcr & 0x1));
        printf("[LTDC Debug] SSCR: 0x%08lX\r\n", LTDC->SSCR);
        printf("[LTDC Debug] BPCR: 0x%08lX\r\n", LTDC->BPCR);
        printf("[LTDC Debug] AWCR: 0x%08lX\r\n", LTDC->AWCR);
        printf("[LTDC Debug] TWCR: 0x%08lX\r\n", LTDC->TWCR);
        printf("[LTDC Debug] Layer1 CR: 0x%08lX (LEN=%lu)\r\n",
               LTDC_Layer1->CR, (LTDC_Layer1->CR & 0x1));
        printf("[LTDC Debug] Layer1 WHPCR: 0x%08lX\r\n", LTDC_Layer1->WHPCR);
        printf("[LTDC Debug] Layer1 WVPCR: 0x%08lX\r\n", LTDC_Layer1->WVPCR);
        printf("[LTDC Debug] Layer1 CFBAR: 0x%08lX\r\n", LTDC_Layer1->CFBAR);
        printf("[LTDC Debug] Layer1 CFBLR: 0x%08lX\r\n", LTDC_Layer1->CFBLR);
        printf("[LTDC Debug] Layer1 CFBLNR: 0x%08lX\r\n", LTDC_Layer1->CFBLNR);
        printf("[LTDC Debug] SRCR: 0x%08lX\r\n", LTDC->SRCR);
        printf("[LTDC Debug] CDSR: 0x%08lX\r\n", LTDC->CDSR);

        /* Try another reload to be sure */
        printf("[Camera Display] Forcing shadow register reload...\r\n");
        LTDC->SRCR = LTDC_SRCR_IMR;
        HAL_Delay(10);
        printf("[LTDC Debug] SRCR after reload: 0x%08lX\r\n", LTDC->SRCR);
        printf("[LTDC Debug] CDSR after reload: 0x%08lX\r\n", LTDC->CDSR);
    }

    /* Enable Layer 0 explicitly */
    LTDC_Layer1->CR |= LTDC_LxCR_LEN;
    LTDC->SRCR = LTDC_SRCR_IMR;

    printf("[Camera Display] Initialization complete\r\n");
}

/**
  * @brief  Start camera capture and display
  * @retval None
  */
void CameraDisplay_Start(void)
{
    printf("[Camera Display] Starting capture...\r\n");

    /* Start DCMIPP CSI Pipe 1 capture to buffer */
    if (HAL_DCMIPP_CSI_PIPE_Start(&hdcmipp, DCMIPP_PIPE1,
                                   DCMIPP_VIRTUAL_CHANNEL0,
                                   BUFFER_ADDRESS,
                                   DCMIPP_MODE_CONTINUOUS) != HAL_OK)
    {
        printf("[ERROR] Failed to start DCMIPP\r\n");
        Error_Handler();
    }

    printf("[Camera Display] Capture started\r\n");
    printf("[Camera Display] LCD should display camera image now\r\n");
}

/**
  * @brief  Stop camera capture and display
  * @retval None
  */
void CameraDisplay_Stop(void)
{
    printf("[Camera Display] Stopping...\r\n");

    /* Stop DCMIPP capture */
    HAL_DCMIPP_PIPE_Stop(&hdcmipp, DCMIPP_PIPE1);

    printf("[Camera Display] Stopped\r\n");
}

/**
  * @brief  DCMIPP frame event callback (VSYNC or frame complete)
  * @param  hdcmipp DCMIPP handle
  * @param  Pipe Pipe number
  * @retval None
  */
__weak void HAL_DCMIPP_PIPE_FrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp, uint32_t Pipe)
{
    /* Ignore unused parameter */
    (void)hdcmipp;
    (void)Pipe;

    frame_count++;

    /* Print frame count every 30 frames (~1 sec at 30 fps) */
    if (frame_count % 30 == 0)
    {
        printf("[Camera] Frame %lu captured\r\n", frame_count);
    }
}
