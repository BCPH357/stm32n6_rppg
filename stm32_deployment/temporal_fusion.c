/**
 * @file temporal_fusion.c
 * @brief Temporal Fusion 模型實現（純 C 語言）
 */

#include "temporal_fusion.h"
#include <string.h>

/* ============================================================================
 * Conv1D 實現
 * ========================================================================== */

/**
 * @brief 1D 卷積操作（單個輸出通道，padding=1）
 */
void conv1d_single_channel(
    const float input[][TF_SEQ_LEN],
    const float weights[][TF_CONV_KERNEL],
    float bias,
    int in_channels,
    int seq_len,
    int kernel_size,
    float *output
) {
    int padding = kernel_size / 2;  // padding=1 for kernel=3

    // 遍歷每個輸出時間步
    for (int t = 0; t < seq_len; t++) {
        float sum = bias;

        // 遍歷輸入通道
        for (int ic = 0; ic < in_channels; ic++) {
            // 遍歷卷積核
            for (int k = 0; k < kernel_size; k++) {
                int t_in = t - padding + k;  // 輸入時間索引

                // Padding：邊界使用零填充
                if (t_in >= 0 && t_in < seq_len) {
                    sum += input[ic][t_in] * weights[ic][k];
                }
            }
        }

        output[t] = sum;
    }
}

/**
 * @brief 1D 卷積層（多個輸出通道）+ ReLU（通用版本）
 */
void conv1d_relu_generic(
    const float input[][TF_SEQ_LEN],
    const void *weights_ptr,  // 使用 void* 避免類型問題
    const float *bias,
    int in_channels,
    int out_channels,
    int seq_len,
    int kernel_size,
    float output[][TF_SEQ_LEN]
) {
    // 將 void* 轉換為適當的指針
    const float (*weights)[in_channels][kernel_size] = weights_ptr;

    // 對每個輸出通道執行卷積
    for (int oc = 0; oc < out_channels; oc++) {
        conv1d_single_channel(
            input,
            weights[oc],
            bias[oc],
            in_channels,
            seq_len,
            kernel_size,
            output[oc]
        );

        // ReLU 激活
        for (int t = 0; t < seq_len; t++) {
            output[oc][t] = relu(output[oc][t]);
        }
    }
}

// 保留舊的函數名作為包裝器（向後兼容）
void conv1d_relu(
    const float input[][TF_SEQ_LEN],
    const float weights[][TF_CONV1_IN_CH][TF_CONV_KERNEL],
    const float *bias,
    int in_channels,
    int out_channels,
    int seq_len,
    int kernel_size,
    float output[][TF_SEQ_LEN]
) {
    conv1d_relu_generic(input, weights, bias, in_channels, out_channels, seq_len, kernel_size, output);
}

/* ============================================================================
 * 全連接層實現
 * ========================================================================== */

/**
 * @brief 全連接層（支持可選的 ReLU）
 */
void fc_layer(
    const float *input,
    const float weights[][TF_FC1_IN],
    const float *bias,
    int in_dim,
    int out_dim,
    float *output,
    int use_relu
) {
    for (int o = 0; o < out_dim; o++) {
        float sum = bias[o];

        // 矩陣乘法：weights[o] · input
        for (int i = 0; i < in_dim; i++) {
            sum += weights[o][i] * input[i];
        }

        // 可選的 ReLU
        output[o] = use_relu ? relu(sum) : sum;
    }
}

/* ============================================================================
 * Temporal Fusion 主推論函數
 * ========================================================================== */

/**
 * @brief Temporal Fusion 完整推論
 */
float temporal_fusion_infer(const float features[24][16]) {
    // -------------------------------------------------------------------------
    // Step 1: Reshape (24, 16) → (8, 3, 16) → (8, 48) → (48, 8)
    // -------------------------------------------------------------------------
    // PyTorch: x.view(B, 8, 3, 16).view(B, 8, 48).transpose(1, 2)
    // 這意味著：features[t*3 + roi][f] → temp[t][roi*16 + f] → reshaped[roi*16 + f][t]

    static float reshaped[TF_CONV1_IN_CH][TF_SEQ_LEN];  // (48, 8)

    for (int t = 0; t < TF_WINDOW_SIZE; t++) {        // t = 0..7
        for (int roi = 0; roi < TF_NUM_ROIS; roi++) { // roi = 0..2
            for (int f = 0; f < TF_FEATURE_DIM; f++) {  // f = 0..15
                // 源索引：features[t*3 + roi][f]
                int src_t_roi = t * TF_NUM_ROIS + roi;  // 0..23

                // 目標索引：reshaped[roi*16 + f][t]
                int dst_channel = roi * TF_FEATURE_DIM + f;  // 0..47

                reshaped[dst_channel][t] = features[src_t_roi][f];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: Conv1D(48 → 32, kernel=3) + ReLU
    // -------------------------------------------------------------------------
    static float conv1_out[TF_CONV1_OUT_CH][TF_SEQ_LEN];  // (32, 8)

    conv1d_relu(
        reshaped,
        g_conv1_weight,
        g_conv1_bias,
        TF_CONV1_IN_CH,
        TF_CONV1_OUT_CH,
        TF_SEQ_LEN,
        TF_CONV_KERNEL,
        conv1_out
    );

    // -------------------------------------------------------------------------
    // Step 3: Conv1D(32 → 16, kernel=3) + ReLU
    // -------------------------------------------------------------------------
    static float conv2_out[TF_CONV2_OUT_CH][TF_SEQ_LEN];  // (16, 8)

    conv1d_relu_generic(
        conv1_out,
        g_conv2_weight,  // 使用通用版本，無需類型轉換
        g_conv2_bias,
        TF_CONV2_IN_CH,
        TF_CONV2_OUT_CH,
        TF_SEQ_LEN,
        TF_CONV_KERNEL,
        conv2_out
    );

    // -------------------------------------------------------------------------
    // Step 4: Flatten (16, 8) → (128)
    // -------------------------------------------------------------------------
    static float flattened[TF_FC1_IN];  // 128

    int idx = 0;
    for (int ch = 0; ch < TF_CONV2_OUT_CH; ch++) {
        for (int t = 0; t < TF_SEQ_LEN; t++) {
            flattened[idx++] = conv2_out[ch][t];
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: FC(128 → 32) + ReLU
    // -------------------------------------------------------------------------
    static float fc1_out[TF_FC1_OUT];  // 32

    fc_layer(
        flattened,
        g_fc1_weight,
        g_fc1_bias,
        TF_FC1_IN,
        TF_FC1_OUT,
        fc1_out,
        1  // use_relu = true
    );

    // -------------------------------------------------------------------------
    // Step 6: FC(32 → 1) + Sigmoid
    // -------------------------------------------------------------------------
    float fc2_out;

    // FC(32 → 1): g_fc2_weight 是 [1][32]，可以直接轉換
    const float (*fc2_weight_ptr)[TF_FC2_IN] = (const float (*)[TF_FC2_IN])g_fc2_weight;

    fc_layer(
        fc1_out,
        fc2_weight_ptr,
        g_fc2_bias,
        TF_FC2_IN,
        TF_FC2_OUT,
        &fc2_out,
        0  // use_relu = false
    );

    float sigmoid_out = sigmoid(fc2_out);

    // -------------------------------------------------------------------------
    // Step 7: Scale to [30, 180] BPM
    // -------------------------------------------------------------------------
    float hr_bpm = sigmoid_out * 150.0f + 30.0f;

    return hr_bpm;
}
