/**
 * @file temporal_fusion.h
 * @brief Temporal Fusion 模型 - CPU 推論（純 C 實現）
 *
 * 架構：
 *   Input: [24][16] float32 (8 幀 × 3 ROI × 16 特徵)
 *   Output: 1 float32 (心率 BPM)
 *
 * 層結構：
 *   1. Reshape: (24, 16) → (8, 48) → (48, 8)
 *   2. Conv1D(48 → 32, kernel=3, padding=1) + ReLU
 *   3. Conv1D(32 → 16, kernel=3, padding=1) + ReLU
 *   4. Flatten → (128,)
 *   5. FC(128 → 32) + ReLU
 *   6. FC(32 → 1) + Sigmoid
 *   7. Scale: sigmoid * 150 + 30 → [30, 180] BPM
 *
 * 參數總量：10,353
 */

#ifndef TEMPORAL_FUSION_H
#define TEMPORAL_FUSION_H

#include <stdint.h>
#include <math.h>

/* ============================================================================
 * 模型配置
 * ========================================================================== */

#define TF_WINDOW_SIZE      8      // 時間窗口大小（幀數）
#define TF_NUM_ROIS         3      // ROI 數量
#define TF_FEATURE_DIM      16     // 每個 ROI 的特徵維度

#define TF_INPUT_SIZE       (TF_WINDOW_SIZE * TF_NUM_ROIS * TF_FEATURE_DIM)  // 24 × 16
#define TF_CONV1_IN_CH      48     // Conv1D 輸入通道
#define TF_CONV1_OUT_CH     32     // Conv1D 輸出通道
#define TF_CONV2_IN_CH      32     // Conv1D 輸入通道
#define TF_CONV2_OUT_CH     16     // Conv1D 輸出通道
#define TF_CONV_KERNEL      3      // Conv1D kernel size
#define TF_SEQ_LEN          8      // 序列長度

#define TF_FC1_IN           128    // FC1 輸入維度 (16 × 8)
#define TF_FC1_OUT          32     // FC1 輸出維度
#define TF_FC2_IN           32     // FC2 輸入維度
#define TF_FC2_OUT          1      // FC2 輸出維度

/* ============================================================================
 * 權重聲明（extern，實際定義在 temporal_fusion_weights.c）
 * ========================================================================== */

// Conv1D Layer 1: (48, 32, 3) - [out_ch][in_ch][kernel]
extern const float g_conv1_weight[TF_CONV1_OUT_CH][TF_CONV1_IN_CH][TF_CONV_KERNEL];
extern const float g_conv1_bias[TF_CONV1_OUT_CH];

// Conv1D Layer 2: (32, 16, 3) - [out_ch][in_ch][kernel]
extern const float g_conv2_weight[TF_CONV2_OUT_CH][TF_CONV2_IN_CH][TF_CONV_KERNEL];
extern const float g_conv2_bias[TF_CONV2_OUT_CH];

// FC Layer 1: (128, 32) - [out][in]
extern const float g_fc1_weight[TF_FC1_OUT][TF_FC1_IN];
extern const float g_fc1_bias[TF_FC1_OUT];

// FC Layer 2: (32, 1) - [out][in]
extern const float g_fc2_weight[TF_FC2_OUT][TF_FC2_IN];
extern const float g_fc2_bias[TF_FC2_OUT];

/* ============================================================================
 * 基礎操作函數
 * ========================================================================== */

/**
 * @brief ReLU 激活函數
 * @param x 輸入值
 * @return max(0, x)
 */
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief Sigmoid 激活函數
 * @param x 輸入值
 * @return 1 / (1 + exp(-x))
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief 1D 卷積操作（單個輸出通道）
 *
 * @param input       輸入張量 [in_channels][seq_len]
 * @param weights     卷積核權重 [in_channels][kernel_size]
 * @param bias        偏置
 * @param in_channels 輸入通道數
 * @param seq_len     序列長度
 * @param kernel_size 卷積核大小
 * @param output      輸出張量 [seq_len]
 */
void conv1d_single_channel(
    const float input[][TF_SEQ_LEN],
    const float weights[][TF_CONV_KERNEL],
    float bias,
    int in_channels,
    int seq_len,
    int kernel_size,
    float *output
);

/**
 * @brief 1D 卷積層（多個輸出通道）+ ReLU
 *
 * @param input       輸入張量 [in_channels][seq_len]
 * @param weights     卷積核權重 [out_channels][in_channels][kernel_size]
 * @param bias        偏置 [out_channels]
 * @param in_channels 輸入通道數
 * @param out_channels 輸出通道數
 * @param seq_len     序列長度
 * @param kernel_size 卷積核大小
 * @param output      輸出張量 [out_channels][seq_len]
 */
void conv1d_relu(
    const float input[][TF_SEQ_LEN],
    const float weights[][TF_CONV1_IN_CH][TF_CONV_KERNEL],
    const float *bias,
    int in_channels,
    int out_channels,
    int seq_len,
    int kernel_size,
    float output[][TF_SEQ_LEN]
);

/**
 * @brief 全連接層 + ReLU
 *
 * @param input    輸入向量 [in_dim]
 * @param weights  權重矩陣 [out_dim][in_dim]
 * @param bias     偏置向量 [out_dim]
 * @param in_dim   輸入維度
 * @param out_dim  輸出維度
 * @param output   輸出向量 [out_dim]
 * @param use_relu 是否使用 ReLU 激活
 */
void fc_layer(
    const float *input,
    const float weights[][TF_FC1_IN],
    const float *bias,
    int in_dim,
    int out_dim,
    float *output,
    int use_relu
);

/* ============================================================================
 * Temporal Fusion 主推論函數
 * ========================================================================== */

/**
 * @brief Temporal Fusion 完整推論
 *
 * @param features 輸入特徵 [24][16] - 來自 Spatial CNN 的 24 個特徵向量
 * @return 預測的心率 (BPM)，範圍 [30, 180]
 *
 * 流程：
 *   1. Reshape: (24, 16) → (8, 3, 16) → (8, 48) → (48, 8)
 *   2. Conv1D(48 → 32) + ReLU
 *   3. Conv1D(32 → 16) + ReLU
 *   4. Flatten → (128)
 *   5. FC(128 → 32) + ReLU
 *   6. FC(32 → 1) + Sigmoid
 *   7. Scale: sigmoid * 150 + 30
 */
float temporal_fusion_infer(const float features[24][16]);

#endif /* TEMPORAL_FUSION_H */
