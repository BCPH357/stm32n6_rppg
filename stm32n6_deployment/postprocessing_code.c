/**
 * rPPG Multi-ROI 輸出後處理代碼範例
 *
 * 功能:
 * 1. 讀取 AI 輸出（float32 心率值）
 * 2. 範圍檢查（30-180 BPM）
 * 3. 平滑濾波（移動平均/中位數）
 * 4. 顯示或傳輸結果
 *
 * 用法: 整合到 app_x-cube-ai.c 的 MX_X_CUBE_AI_Process()
 */

#include "main.h"
#include "app_x-cube-ai.h"
#include <stdio.h>
#include <math.h>

/* 配置參數 */
#define HR_MIN        30.0f   // 最小心率 (BPM)
#define HR_MAX        180.0f  // 最大心率 (BPM)
#define SMOOTH_WINDOW 10      // 平滑窗口大小
#define OUTLIER_THRESHOLD 20.0f  // 異常值閾值

/* 平滑緩衝區 */
static float hr_history[SMOOTH_WINDOW] = {0};
static uint32_t hr_history_idx = 0;
static uint32_t hr_history_count = 0;

/**
 * 讀取 AI 輸出
 *
 * 注意: 輸出數據類型取決於 X-CUBE-AI 配置
 * - float32: 直接讀取
 * - int8: 需要反量化（scale + zero_point）
 */
static float read_ai_output(const void* output_buffer, bool is_int8)
{
    if (is_int8) {
        // INT8 輸出（需要反量化）
        const int8_t* int8_output = (const int8_t*)output_buffer;

        // 量化參數（從 X-CUBE-AI 分析報告獲取）
        // 這些值需要根據實際模型調整
        const float scale = 1.1765f;  // 示例值
        const int8_t zero_point = 0;  // 示例值

        float hr = (float)(int8_output[0] - zero_point) * scale;
        return hr;
    } else {
        // float32 輸出（推薦）
        const float* float_output = (const float*)output_buffer;
        return float_output[0];
    }
}

/**
 * 範圍檢查
 */
static bool is_hr_valid(float hr)
{
    if (isnan(hr) || isinf(hr)) {
        printf("Warning: HR is NaN or Inf\n");
        return false;
    }

    if (hr < HR_MIN || hr > HR_MAX) {
        printf("Warning: HR out of range: %.2f BPM\n", hr);
        return false;
    }

    return true;
}

/**
 * 異常值檢測
 *
 * 檢查新值與歷史平均的差異
 */
static bool is_outlier(float new_hr)
{
    if (hr_history_count == 0) {
        return false;  // 第一個值，無法判斷
    }

    // 計算歷史平均
    float sum = 0.0f;
    uint32_t count = (hr_history_count < SMOOTH_WINDOW) ?
                     hr_history_count : SMOOTH_WINDOW;

    for (uint32_t i = 0; i < count; i++) {
        sum += hr_history[i];
    }
    float mean = sum / count;

    // 檢查偏差
    float diff = fabsf(new_hr - mean);
    if (diff > OUTLIER_THRESHOLD) {
        printf("Warning: Possible outlier (diff=%.2f BPM)\n", diff);
        return true;
    }

    return false;
}

/**
 * 添加到歷史緩衝區
 */
static void add_to_history(float hr)
{
    hr_history[hr_history_idx] = hr;
    hr_history_idx = (hr_history_idx + 1) % SMOOTH_WINDOW;

    if (hr_history_count < SMOOTH_WINDOW) {
        hr_history_count++;
    }
}

/**
 * 移動平均濾波
 */
static float moving_average_filter(void)
{
    if (hr_history_count == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    uint32_t count = (hr_history_count < SMOOTH_WINDOW) ?
                     hr_history_count : SMOOTH_WINDOW;

    for (uint32_t i = 0; i < count; i++) {
        sum += hr_history[i];
    }

    return sum / count;
}

/**
 * 中位數濾波（更魯棒）
 */
static float median_filter(void)
{
    if (hr_history_count == 0) {
        return 0.0f;
    }

    uint32_t count = (hr_history_count < SMOOTH_WINDOW) ?
                     hr_history_count : SMOOTH_WINDOW;

    // 複製到臨時數組（避免修改原始數據）
    float temp[SMOOTH_WINDOW];
    memcpy(temp, hr_history, count * sizeof(float));

    // 簡單排序（冒泡排序，適合小數組）
    for (uint32_t i = 0; i < count - 1; i++) {
        for (uint32_t j = 0; j < count - i - 1; j++) {
            if (temp[j] > temp[j + 1]) {
                float swap = temp[j];
                temp[j] = temp[j + 1];
                temp[j + 1] = swap;
            }
        }
    }

    // 返回中位數
    if (count % 2 == 0) {
        return (temp[count / 2 - 1] + temp[count / 2]) / 2.0f;
    } else {
        return temp[count / 2];
    }
}

/**
 * 主後處理函數
 *
 * 輸入: AI 輸出緩衝區
 * 輸出: 濾波後的心率值
 */
float postprocess_ai_output(
    const void* output_buffer,
    bool is_int8,
    bool use_median_filter
)
{
    // 1. 讀取原始輸出
    float raw_hr = read_ai_output(output_buffer, is_int8);

    printf("Raw HR: %.2f BPM", raw_hr);

    // 2. 範圍檢查
    if (!is_hr_valid(raw_hr)) {
        printf(" [INVALID]\n");
        // 返回上一次有效值（或默認值）
        return (hr_history_count > 0) ?
               hr_history[(hr_history_idx - 1 + SMOOTH_WINDOW) % SMOOTH_WINDOW] :
               75.0f;  // 默認心率
    }

    // 3. 異常值檢測
    if (is_outlier(raw_hr)) {
        printf(" [OUTLIER]");
        // 不添加到歷史，返回濾波值
        float smoothed = use_median_filter ?
                        median_filter() :
                        moving_average_filter();
        printf(" → Smoothed: %.2f BPM\n", smoothed);
        return smoothed;
    }

    // 4. 添加到歷史
    add_to_history(raw_hr);

    // 5. 應用濾波
    float smoothed_hr = use_median_filter ?
                       median_filter() :
                       moving_average_filter();

    printf(" → Smoothed: %.2f BPM\n", smoothed_hr);

    return smoothed_hr;
}

/**
 * 顯示結果（LCD/UART）
 */
void display_heart_rate(float hr)
{
    // 方法 A: UART 輸出（調試用）
    printf("\n===================================\n");
    printf("  Heart Rate: %.1f BPM\n", hr);
    printf("===================================\n\n");

    // 方法 B: LCD 顯示（如果有）
    // lcd_clear();
    // lcd_set_cursor(0, 0);
    // lcd_printf("Heart Rate:");
    // lcd_set_cursor(1, 0);
    // lcd_printf("%.1f BPM", hr);

    // 方法 C: LED 指示（簡單可視化）
    // 心率分級:
    //   < 60: 藍燈（低）
    //   60-100: 綠燈（正常）
    //   > 100: 紅燈（高）
    if (hr < 60.0f) {
        // HAL_GPIO_WritePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin, GPIO_PIN_SET);
    } else if (hr <= 100.0f) {
        // HAL_GPIO_WritePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin, GPIO_PIN_SET);
    } else {
        // HAL_GPIO_WritePin(LED_RED_GPIO_Port, LED_RED_Pin, GPIO_PIN_SET);
    }
}

/**
 * 完整的後處理流程（整合到推論函數）
 *
 * 用法範例:
 *   void MX_X_CUBE_AI_Process(void)
 *   {
 *       // 運行推論
 *       ai_i32 nbatch = ai_network_run(network, ai_input, ai_output);
 *
 *       if (nbatch == 1) {
 *           // 後處理
 *           float hr = process_and_display_result(ai_output[0].data);
 *
 *           // 可選: 記錄到存儲
 *           log_heart_rate(hr, HAL_GetTick());
 *       }
 *   }
 */
float process_and_display_result(const void* output_buffer)
{
    // 配置（根據 X-CUBE-AI 設置）
    const bool is_int8 = false;         // 輸出類型（通常推薦 float32）
    const bool use_median = true;       // 使用中位數濾波（更魯棒）

    // 後處理
    float hr = postprocess_ai_output(output_buffer, is_int8, use_median);

    // 顯示
    display_heart_rate(hr);

    return hr;
}

/**
 * 重置濾波器（當檢測到新用戶時）
 */
void reset_hr_filter(void)
{
    hr_history_idx = 0;
    hr_history_count = 0;
    memset(hr_history, 0, sizeof(hr_history));
    printf("HR filter reset\n");
}
