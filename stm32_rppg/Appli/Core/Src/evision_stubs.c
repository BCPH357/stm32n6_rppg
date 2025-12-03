/* Minimal stubs for eVision AE/AWB APIs to satisfy linking.
 * Replace with real libraries if using full ISP algorithms.
 */

#include <stdint.h>

void *evision_api_st_ae_new(void) { return (void *)1; }
int32_t evision_api_st_ae_init(void *ctx) { (void)ctx; return 0; }
int32_t evision_api_st_ae_process(void *ctx) { (void)ctx; return 0; }
int32_t evision_api_st_ae_delete(void *ctx) { (void)ctx; return 0; }

void *evision_api_awb_new(void) { return (void *)1; }
int32_t evision_api_awb_init_profiles(void *ctx) { (void)ctx; return 0; }
int32_t evision_api_awb_set_profile(void *ctx, uint32_t profile_id) { (void)ctx; (void)profile_id; return 0; }
int32_t evision_api_awb_run_average(void *ctx) { (void)ctx; return 0; }
int32_t evision_api_awb_delete(void *ctx) { (void)ctx; return 0; }
