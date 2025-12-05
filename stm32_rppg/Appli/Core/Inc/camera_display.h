/**
  ******************************************************************************
  * @file    camera_display.h
  * @brief   Camera capture and LCD display module
  ******************************************************************************
  */

#ifndef __CAMERA_DISPLAY_H
#define __CAMERA_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32n6xx_hal.h"

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
#define CAMERA_WIDTH    800   /* Down-sized output width from ISP/DCMIPP */
#define CAMERA_HEIGHT   480   /* Down-sized output height from ISP/DCMIPP */
#define LCD_WIDTH       800
#define LCD_HEIGHT      480
#define BUFFER_ADDRESS  0x34200000  /* AXISRAM3 (1 MB) */

/* Exported macro ------------------------------------------------------------*/
/* Exported functions prototypes ---------------------------------------------*/
void CameraDisplay_Init(void);
void CameraDisplay_Start(void);
void CameraDisplay_Stop(void);

#ifdef __cplusplus
}
#endif

#endif /* __CAMERA_DISPLAY_H */
