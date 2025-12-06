/**
  ******************************************************************************
  * @file    camera_display.c
  * @brief   Camera capture and LCD display implementation
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "camera_display.h"
#include "main.h"
#include "isp_api.h"
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
/* HAL callback is provided by BSP and routes to BSP_CAMERA_FrameEventCallback().
 * Override the BSP weak hook here so we get logs when frames arrive. */
void BSP_CAMERA_FrameEventCallback(uint32_t Instance)
{
    (void)Instance;
    frame_count++;

    /* Print frame count every 30 frames (~1 sec at 30 fps) */
    if ((frame_count % 30U) == 0U)
    {
        printf("[Camera] Frame %lu captured\r\n", frame_count);

        /* Peek first few pixels to verify buffer is being written */
        volatile uint16_t *fb = (volatile uint16_t *)BUFFER_ADDRESS;
        printf("[Camera] Buffer @0x%08lX, FB[0..3]=0x%04X 0x%04X 0x%04X 0x%04X\r\n",
               (unsigned long)BUFFER_ADDRESS, fb[0], fb[1], fb[2], fb[3]);

        /* Dump DCMIPP Pipe1 pixel packer destination registers */
        printf("[DCMIPP DST] P1PPM0AR1=0x%08lX P1PPM0AR2=0x%08lX P1PPM0PR=0x%08lX\r\n",
               (unsigned long)DCMIPP->P1PPM0AR1,
               (unsigned long)DCMIPP->P1PPM0AR2,
               (unsigned long)DCMIPP->P1PPM0PR);

        /* Dump critical PIPE1 control registers */
        printf("[DCMIPP PIPE1] P1FSCR=0x%08lX (PIPEN=%lu) P1FCTCR=0x%08lX (CPTREQ=%lu CPTMODE=%lu)\r\n",
               (unsigned long)DCMIPP->P1FSCR,
               (unsigned long)((DCMIPP->P1FSCR >> 31) & 0x1),  /* PIPEN bit 31 */
               (unsigned long)DCMIPP->P1FCTCR,
               (unsigned long)((DCMIPP->P1FCTCR >> 3) & 0x1),  /* CPTREQ bit 3 */
               (unsigned long)((DCMIPP->P1FCTCR >> 1) & 0x1)); /* CPTMODE bit 1 */

        /* Dump CMSR1 status - capture active flags */
        printf("[DCMIPP STATUS] CMSR1=0x%08lX (P1CPTACT=%lu)\r\n",
               (unsigned long)DCMIPP->CMSR1,
               (unsigned long)((DCMIPP->CMSR1 >> 9) & 0x1));  /* P1CPTACT bit 9 */

        /* Dump Pixel Packer config */
        printf("[DCMIPP P1PPCR] P1PPCR=0x%08lX (FORMAT=%lu)\r\n",
               (unsigned long)DCMIPP->P1PPCR,
               (unsigned long)(DCMIPP->P1PPCR & 0xF));  /* FORMAT bits 0-3 */

        /* Dump ISP/Demosaic enable status */
        printf("[DCMIPP ISP] P1DMCR=0x%08lX (ENABLE=%lu) P1DECR=0x%08lX (ENABLE=%lu)\r\n",
               (unsigned long)DCMIPP->P1DMCR,
               (unsigned long)(DCMIPP->P1DMCR & 0x1),  /* ENABLE bit 0 */
               (unsigned long)DCMIPP->P1DECR,
               (unsigned long)(DCMIPP->P1DECR & 0x1)); /* ENABLE bit 0 */

        /* ===== Downsize registers ===== */
        printf("[DCMIPP DOWNSIZE] P1DSCR=0x%08lX (ENABLE=%lu, HDIV=%lu, VDIV=%lu)\r\n",
               (unsigned long)DCMIPP->P1DSCR,
               (unsigned long)((DCMIPP->P1DSCR >> 31) & 0x1),
               (unsigned long)(DCMIPP->P1DSCR & 0x3FF),
               (unsigned long)((DCMIPP->P1DSCR >> 16) & 0x3FF));
        printf("[DCMIPP DOWNSIZE] P1CRSZR=0x%08lX (ENABLE=%lu, HSIZE=%lu, VSIZE=%lu)\r\n",
               (unsigned long)DCMIPP->P1CRSZR,
               (unsigned long)((DCMIPP->P1CRSZR >> 31) & 0x1),
               (unsigned long)(DCMIPP->P1CRSZR & 0xFFF),
               (unsigned long)((DCMIPP->P1CRSZR >> 16) & 0xFFF));

        /* ===== Check buffer at different offsets ===== */
        printf("[Buffer Check] Checking 4 corners:\r\n");
        printf("  [0,0]=0x%04X [400,0]=0x%04X [0,240]=0x%04X [799,479]=0x%04X\r\n",
               fb[0],                         /* top-left */
               fb[400],                       /* top-middle */
               fb[240*800],                   /* middle-left */
               fb[479*800 + 799]);            /* bottom-right */

        /* Check if CMSR1 P1CPTACT is really at bit 23 */
        printf("[CMSR1 DEBUG] Raw=0x%08lX bit23=%lu bit9=%lu\r\n",
               (unsigned long)DCMIPP->CMSR1,
               (unsigned long)((DCMIPP->CMSR1 >> 23) & 0x1),
               (unsigned long)((DCMIPP->CMSR1 >> 9) & 0x1));
    }
}

/* Note: HAL_DCMIPP_PIPE_VsyncEventCallback is implemented in BSP driver
   (stm32n6570_discovery_camera.c) and handles ISP statistics gathering */
