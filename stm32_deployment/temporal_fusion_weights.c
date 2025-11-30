/**
 * @file temporal_fusion_weights.c
 * @brief Temporal Fusion 權重定義（佔位符）
 *
 * 注意：這些是範例權重，實際權重需要從 PyTorch 模型導出
 *
 * 使用 export_temporal_fusion_weights.py 腳本生成實際權重
 */

#include "temporal_fusion.h"

/* ============================================================================
 * Conv1D Layer 1 權重: (48 → 32, kernel=3)
 * ========================================================================== */

// g_conv1_weight[32][48][3] - 4,608 個參數
const float g_conv1_weight[TF_CONV1_OUT_CH][TF_CONV1_IN_CH][TF_CONV_KERNEL] = {
    // 範例：全部初始化為 0.01f（佔位符）
    // 實際權重由 Python 腳本生成
    [0 ... (TF_CONV1_OUT_CH - 1)] = {
        [0 ... (TF_CONV1_IN_CH - 1)] = {0.01f, 0.01f, 0.01f}
    }
};

// g_conv1_bias[32]
const float g_conv1_bias[TF_CONV1_OUT_CH] = {
    [0 ... (TF_CONV1_OUT_CH - 1)] = 0.0f
};

/* ============================================================================
 * Conv1D Layer 2 權重: (32 → 16, kernel=3)
 * ========================================================================== */

// g_conv2_weight[16][32][3] - 1,536 個參數
const float g_conv2_weight[TF_CONV2_OUT_CH][TF_CONV2_IN_CH][TF_CONV_KERNEL] = {
    [0 ... (TF_CONV2_OUT_CH - 1)] = {
        [0 ... (TF_CONV2_IN_CH - 1)] = {0.01f, 0.01f, 0.01f}
    }
};

// g_conv2_bias[16]
const float g_conv2_bias[TF_CONV2_OUT_CH] = {
    [0 ... (TF_CONV2_OUT_CH - 1)] = 0.0f
};

/* ============================================================================
 * FC Layer 1 權重: (128 → 32)
 * ========================================================================== */

// g_fc1_weight[32][128] - 4,096 個參數
const float g_fc1_weight[TF_FC1_OUT][TF_FC1_IN] = {
    [0 ... (TF_FC1_OUT - 1)] = {
        [0 ... (TF_FC1_IN - 1)] = 0.01f
    }
};

// g_fc1_bias[32]
const float g_fc1_bias[TF_FC1_OUT] = {
    [0 ... (TF_FC1_OUT - 1)] = 0.0f
};

/* ============================================================================
 * FC Layer 2 權重: (32 → 1)
 * ========================================================================== */

// g_fc2_weight[1][32] - 32 個參數
const float g_fc2_weight[TF_FC2_OUT][TF_FC2_IN] = {
    {[0 ... (TF_FC2_IN - 1)] = 0.01f}
};

// g_fc2_bias[1]
const float g_fc2_bias[TF_FC2_OUT] = {0.0f};

/* ============================================================================
 * 參數統計
 * ========================================================================== */

/*
 * 總參數量：10,353
 *
 * 分層統計：
 *   Conv1D Layer 1:
 *     - weight: 32 × 48 × 3 = 4,608
 *     - bias:   32
 *     - 小計: 4,640
 *
 *   Conv1D Layer 2:
 *     - weight: 16 × 32 × 3 = 1,536
 *     - bias:   16
 *     - 小計: 1,552
 *
 *   FC Layer 1:
 *     - weight: 32 × 128 = 4,096
 *     - bias:   32
 *     - 小計: 4,128
 *
 *   FC Layer 2:
 *     - weight: 1 × 32 = 32
 *     - bias:   1
 *     - 小計: 33
 *
 * 內存占用（FP32）：
 *   - 權重: 10,353 × 4 bytes = 41,412 bytes (~40.4 KB)
 *   - 中間緩衝區: ~2 KB
 *   - 總計: ~42.5 KB
 */
