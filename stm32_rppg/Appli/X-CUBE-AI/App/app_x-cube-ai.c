
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
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

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Neural ART Embedded Client API
  *          Re-target of the printf function is out-of-scope.
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"

/* USER CODE BEGIN includes */
/* USER CODE END includes */

/* Entry points --------------------------------------------------------------*/

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(network_rppg)
uint8_t *buffer_in;
uint8_t *buffer_out;

void set_clk_sleep_mode(void)
{
  /* Leave clocks enabled in Low Power modes */
  // Low-power clock enable misc
#if defined (CPU_IN_SECURE_STATE)
  __HAL_RCC_DBG_CLK_SLEEP_ENABLE();
#endif
  __HAL_RCC_XSPIPHYCOMP_CLK_SLEEP_ENABLE();

  // Low-power clock enable for memories
  __HAL_RCC_AXISRAM1_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM2_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM3_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM4_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM5_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM6_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_FLEXRAM_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXIRAM_MEM_CLK_SLEEP_ENABLE();
  // LP clock AHB1: None
  // LP clock AHB2: None
  // LP clock AHB3
#if defined (CPU_IN_SECURE_STATE)
  __HAL_RCC_RIFSC_CLK_SLEEP_ENABLE();
  __HAL_RCC_RISAF_CLK_SLEEP_ENABLE();
  __HAL_RCC_IAC_CLK_SLEEP_ENABLE();
#endif
  // LP clock AHB4: None
  // LP clocks AHB5
  __HAL_RCC_XSPI1_CLK_SLEEP_ENABLE();
  __HAL_RCC_XSPI2_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXI_CLK_SLEEP_ENABLE();
  __HAL_RCC_NPU_CLK_SLEEP_ENABLE();
  // LP clocks APB1: None
  // LP clocks APB2
  __HAL_RCC_USART1_CLK_SLEEP_ENABLE();
  // LP clocks APB4: None
  // LP clocks APB5: None
}

void MX_X_CUBE_AI_Init(void)
{
    __HAL_RCC_AXISRAM2_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM3_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM4_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM5_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM6_MEM_CLK_ENABLE();
    RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    BSP_XSPI_NOR_Init_t xspiInit;
    xspiInit.InterfaceMode = MX66UW1G45G_OPI_MODE;
    xspiInit.TransferRate = MX66UW1G45G_DTR_TRANSFER;
    BSP_XSPI_NOR_Init(0,&xspiInit);
    BSP_XSPI_NOR_EnableMemoryMappedMode(0);
    MODIFY_REG(XSPI2->CR, XSPI_CR_NOPREF, HAL_XSPI_AUTOMATIC_PREFETCH_DISABLE);
    BSP_XSPI_RAM_Init(0);
    BSP_XSPI_RAM_EnableMemoryMappedMode(0);
    MODIFY_REG(XSPI1->CR, XSPI_CR_NOPREF, HAL_XSPI_AUTOMATIC_PREFETCH_DISABLE);
    set_clk_sleep_mode();
    __HAL_RCC_NPU_CLK_ENABLE();
    __HAL_RCC_NPU_FORCE_RESET();
    __HAL_RCC_NPU_RELEASE_RESET();
    npu_cache_init();
    /* USER CODE BEGIN 5 */
    /* USER CODE END 5 */
}
uint32_t buff_in_len, buff_out_len;
void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
	LL_ATON_RT_RetValues_t ll_aton_rt_ret = LL_ATON_RT_DONE;
	const LL_Buffer_InfoTypeDef * ibuffersInfos = NN_Interface_network_rppg.input_buffers_info();
	const LL_Buffer_InfoTypeDef * obuffersInfos = NN_Interface_network_rppg.output_buffers_info();
	buffer_in = (uint8_t *)LL_Buffer_addr_start(&ibuffersInfos[0]);
	buffer_out = (uint8_t *)LL_Buffer_addr_start(&obuffersInfos[0]);
	// Printing buffer start and end addresses.
	printf("Input buffer: offset start = %lu, \n \r offset end = %lu \n \r",ibuffersInfos->offset_start,ibuffersInfos->offset_end);
	printf("Output buffer: offset start = %lu, \n \r offset end = %lu \n \r",obuffersInfos->offset_start,obuffersInfos->offset_end);
	// Getting buffer size and printing it.
	buff_in_len = ibuffersInfos->offset_end - ibuffersInfos->offset_start;
	buff_out_len = obuffersInfos->offset_end - obuffersInfos->offset_start;
	printf("Buffer input size = %lu \n\r Buffer output size = %lu \n\r", buff_in_len, buff_out_len);
	uint8_t val = 10;
	LL_ATON_RT_RuntimeInit();
	// Run 10 inferences
	for (int inferenceNb = 0; inferenceNb < 10; ++inferenceNb) {
		/* ------------- */
		/* - Inference - */
		/* ------------- */
		/* Pre-process and fill the input buffer */
		// Fill input buffer with constant data.
		for(uint32_t i = 0; i < buff_in_len; i++){
			buffer_in[i] = val;
		}
		// Clean and invalidate MCU DCache and invalidate NPU cache.
		mcu_cache_clean_invalidate_range(buffer_in, buffer_in + buff_in_len);
		npu_cache_invalidate();
		// Check that input buffer was properly assigned with "val".
		printf("Buffer[1] = %d \n \r", buffer_in[1]);
		printf("Buffer[1000] = %d \n \r", buffer_in[1000]);
		printf("Buffer[10000] = %d \n \r", buffer_in[10000]);
		//_pre_process(buffer_in);
		/* Perform the inference */
		LL_ATON_RT_Init_Network(&NN_Instance_network_rppg); // Initialize network instance
		do {
			// Execute first/next epoch block
			ll_aton_rt_ret = LL_ATON_RT_RunEpochBlock(&NN_Instance_network_rppg);
			// Wait for event if required
			if (ll_aton_rt_ret == LL_ATON_RT_WFE) {
				LL_ATON_OSAL_WFE();
			}
		} while (ll_aton_rt_ret != LL_ATON_RT_DONE);
		// Post-process the output buffer
		// Invalidate CPU cache if needed
		// Convert int8 to float. Buffer is int8, but model's output is float.
		uint8_t aux[4];
		float_t *conver;
		for(uint32_t i = 0; i < buff_out_len; i += 4){
			aux[0] = buffer_out[i];
			aux[1] = buffer_out[i+1];
			aux[2] = buffer_out[i+2];
			aux[3] = buffer_out[i+3];
			conver = (float_t *)aux;
			printf("Out %lu = %f \n \r", i, *conver);
		}
		//_post_process(buffer_out);
		LL_ATON_RT_DeInit_Network(&NN_Instance_network_rppg);
		/* -------------------- */
		/* - End of Inference - */
		/* -------------------- */
	}
	LL_ATON_RT_RuntimeDeInit();
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
