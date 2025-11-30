/**
 * @file test_temporal_fusion.c
 * @brief Temporal Fusion 單元測試
 *
 * 編譯：
 *   gcc -o test_temporal_fusion test_temporal_fusion.c temporal_fusion.c temporal_fusion_weights.c -lm
 *
 * 執行：
 *   ./test_temporal_fusion
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "temporal_fusion.h"

/* ============================================================================
 * 測試用例
 * ========================================================================== */

/**
 * @brief 測試用例 1：全零輸入
 */
void test_case_1_all_zeros(void) {
    printf("\n========================================\n");
    printf("測試用例 1: 全零輸入\n");
    printf("========================================\n");

    float features[24][16] = {0};  // 全零初始化

    float hr = temporal_fusion_infer(features);

    printf("輸入: 全零 (24×16)\n");
    printf("輸出: %.2f BPM\n", hr);
    printf("預期: 應該在 [30, 180] 範圍內\n");

    if (hr >= 30.0f && hr <= 180.0f) {
        printf("[PASS] HR 在合理範圍內\n");
    } else {
        printf("[FAIL] HR 超出範圍！\n");
    }
}

/**
 * @brief 測試用例 2：隨機輸入
 */
void test_case_2_random_input(void) {
    printf("\n========================================\n");
    printf("測試用例 2: 隨機輸入\n");
    printf("========================================\n");

    float features[24][16];

    // 生成隨機輸入（模擬 Spatial CNN 輸出）
    srand(42);  // 固定種子，確保可重現
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 16; j++) {
            // 隨機值範圍 [-1.0, 1.0]
            features[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    float hr = temporal_fusion_infer(features);

    printf("輸入: 隨機值 (24×16, 範圍 [-1, 1])\n");
    printf("輸出: %.2f BPM\n", hr);

    if (hr >= 30.0f && hr <= 180.0f) {
        printf("[PASS] HR 在合理範圍內\n");
    } else {
        printf("[FAIL] HR 超出範圍！\n");
    }
}

/**
 * @brief 測試用例 3：模擬真實特徵（正常心率）
 */
void test_case_3_normal_hr(void) {
    printf("\n========================================\n");
    printf("測試用例 3: 模擬正常心率特徵\n");
    printf("========================================\n");

    float features[24][16];

    // 模擬平滑的時序特徵（代表穩定的心率信號）
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 16; j++) {
            // 正弦波 + 小噪聲（模擬心率信號）
            float phase = (float)i / 24.0f * 2.0f * M_PI;
            float signal = sinf(phase * 3.0f);  // 3 個週期
            float noise = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            features[i][j] = signal + noise;
        }
    }

    float hr = temporal_fusion_infer(features);

    printf("輸入: 正弦波特徵 (模擬穩定心跳)\n");
    printf("輸出: %.2f BPM\n", hr);
    printf("預期: 應該接近正常心率 (~60-100 BPM)\n");

    if (hr >= 50.0f && hr <= 120.0f) {
        printf("[PASS] HR 在正常範圍內\n");
    } else {
        printf("[WARNING] HR 可能偏高或偏低\n");
    }
}

/**
 * @brief 測試用例 4：顯示輸入數據（前 3 個特徵向量）
 */
void test_case_4_show_input(void) {
    printf("\n========================================\n");
    printf("測試用例 4: 顯示輸入格式\n");
    printf("========================================\n");

    float features[24][16];

    // 填充遞增值
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 16; j++) {
            features[i][j] = (float)(i * 16 + j) * 0.01f;
        }
    }

    printf("輸入特徵矩陣 (24×16):\n");
    printf("顯示前 3 個特徵向量（每個 16 維）:\n\n");

    for (int i = 0; i < 3; i++) {
        printf("特徵向量 [%2d]: [", i);
        for (int j = 0; j < 16; j++) {
            printf("%.2f", features[i][j]);
            if (j < 15) printf(", ");
        }
        printf("]\n");
    }

    printf("...\n");

    float hr = temporal_fusion_infer(features);

    printf("\n輸出: %.2f BPM\n", hr);
}

/**
 * @brief 性能測試：運行 1000 次推論
 */
void test_case_5_performance(void) {
    printf("\n========================================\n");
    printf("測試用例 5: 性能測試\n");
    printf("========================================\n");

    float features[24][16];

    // 準備隨機輸入
    srand(12345);
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 16; j++) {
            features[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    // 運行 1000 次推論（測量時間需要添加計時代碼）
    int num_runs = 1000;
    float total_hr = 0.0f;

    printf("運行 %d 次推論...\n", num_runs);

    for (int n = 0; n < num_runs; n++) {
        float hr = temporal_fusion_infer(features);
        total_hr += hr;
    }

    float avg_hr = total_hr / num_runs;

    printf("完成 %d 次推論\n", num_runs);
    printf("平均輸出: %.2f BPM\n", avg_hr);
    printf("注意: 在 STM32 上需要使用計時器測量實際性能\n");
}

/* ============================================================================
 * 主函數
 * ========================================================================== */

int main(void) {
    printf("======================================================================\n");
    printf("Temporal Fusion 模型單元測試（純 C 實現）\n");
    printf("======================================================================\n");
    printf("\n模型配置:\n");
    printf("  輸入: [24][16] float32\n");
    printf("  輸出: 1 float32 (心率 BPM)\n");
    printf("  參數量: 10,353\n");
    printf("  內存占用: ~42.5 KB (FP32 權重 + 緩衝區)\n");

    // 運行所有測試用例
    test_case_1_all_zeros();
    test_case_2_random_input();
    test_case_3_normal_hr();
    test_case_4_show_input();
    test_case_5_performance();

    printf("\n======================================================================\n");
    printf("測試完成\n");
    printf("======================================================================\n");
    printf("\n下一步:\n");
    printf("  1. 使用 export_temporal_fusion_weights.py 導出實際權重\n");
    printf("  2. 替換 temporal_fusion_weights.c 中的佔位符權重\n");
    printf("  3. 重新編譯並驗證輸出與 PyTorch 一致\n");
    printf("  4. 整合到 STM32N6 項目中\n");
    printf("======================================================================\n");

    return 0;
}
