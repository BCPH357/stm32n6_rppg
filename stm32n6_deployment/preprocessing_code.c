/**
 * rPPG Multi-ROI 輸入預處理代碼範例
 *
 * 功能:
 * 1. 從攝像頭獲取 RGB 幀
 * 2. 臉部檢測（簡化版）
 * 3. 提取 3 個 ROI（前額、左右臉頰）
 * 4. Resize 到 36×36
 * 5. 轉換為 INT8
 * 6. 組織為時間窗口 (8, 3, 36, 36, 3)
 *
 * 用法: 整合到 app_x-cube-ai.c 的 MX_X_CUBE_AI_Process()
 */

#include "main.h"
#include "app_x-cube-ai.h"
#include <string.h>
#include <stdio.h>

/* 配置參數 */
#define INPUT_WIDTH  640
#define INPUT_HEIGHT 480
#define ROI_SIZE     36
#define TIME_WINDOW  8
#define NUM_ROIS     3

/* ROI 位置定義（相對臉部 bbox）*/
typedef struct {
    float x_start;
    float x_end;
    float y_start;
    float y_end;
} ROI_Config;

const ROI_Config roi_configs[NUM_ROIS] = {
    {0.20f, 0.80f, 0.05f, 0.25f},  // Forehead
    {0.05f, 0.30f, 0.35f, 0.65f},  // Left Cheek
    {0.70f, 0.95f, 0.35f, 0.65f}   // Right Cheek
};

/* 全局緩衝區 */
static uint8_t frame_buffer[INPUT_HEIGHT][INPUT_WIDTH][3];  // RGB 幀
static int8_t roi_buffer[TIME_WINDOW][NUM_ROIS][ROI_SIZE][ROI_SIZE][3];  // ROI 時間窗口
static uint32_t frame_count = 0;

/* 臉部檢測結果（簡化版）*/
typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;
} FaceRect;

/**
 * 簡化的臉部檢測
 *
 * 注意: 這是極簡版本，實際應用需要：
 * - 移植 Haar Cascade 或 MediaPipe
 * - 或使用專用臉部檢測 MCU/NPU 模型
 */
static bool detect_face_simple(FaceRect* face)
{
    // 假設臉部在畫面中央（用於初步測試）
    face->x = INPUT_WIDTH / 4;
    face->y = INPUT_HEIGHT / 4;
    face->width = INPUT_WIDTH / 2;
    face->height = INPUT_HEIGHT / 2;

    return true;  // 總是返回檢測成功（測試用）
}

/**
 * 雙線性插值 Resize
 *
 * 將 src (src_w × src_h) resize 到 dst (ROI_SIZE × ROI_SIZE)
 */
static void resize_bilinear(
    const uint8_t* src,
    uint16_t src_w,
    uint16_t src_h,
    uint8_t* dst,
    uint16_t dst_w,
    uint16_t dst_h,
    uint8_t channels
)
{
    float x_ratio = (float)src_w / dst_w;
    float y_ratio = (float)src_h / dst_h;

    for (uint16_t y = 0; y < dst_h; y++) {
        for (uint16_t x = 0; x < dst_w; x++) {
            // 計算源坐標
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            uint16_t x0 = (uint16_t)src_x;
            uint16_t y0 = (uint16_t)src_y;
            uint16_t x1 = (x0 + 1 < src_w) ? x0 + 1 : x0;
            uint16_t y1 = (y0 + 1 < src_h) ? y0 + 1 : y0;

            float dx = src_x - x0;
            float dy = src_y - y0;

            // 對每個通道進行插值
            for (uint8_t c = 0; c < channels; c++) {
                uint16_t idx00 = (y0 * src_w + x0) * channels + c;
                uint16_t idx01 = (y0 * src_w + x1) * channels + c;
                uint16_t idx10 = (y1 * src_w + x0) * channels + c;
                uint16_t idx11 = (y1 * src_w + x1) * channels + c;

                float val = src[idx00] * (1 - dx) * (1 - dy) +
                           src[idx01] * dx * (1 - dy) +
                           src[idx10] * (1 - dx) * dy +
                           src[idx11] * dx * dy;

                uint16_t dst_idx = (y * dst_w + x) * channels + c;
                dst[dst_idx] = (uint8_t)(val + 0.5f);  // 四捨五入
            }
        }
    }
}

/**
 * 提取 ROI 並轉換為 INT8
 */
static void extract_roi(
    const uint8_t* frame,
    const FaceRect* face,
    uint8_t roi_idx,
    int8_t* output  // (ROI_SIZE, ROI_SIZE, 3)
)
{
    const ROI_Config* cfg = &roi_configs[roi_idx];

    // 計算 ROI 在原始幀中的坐標
    uint16_t roi_x = face->x + (uint16_t)(cfg->x_start * face->width);
    uint16_t roi_y = face->y + (uint16_t)(cfg->y_start * face->height);
    uint16_t roi_w = (uint16_t)((cfg->x_end - cfg->x_start) * face->width);
    uint16_t roi_h = (uint16_t)((cfg->y_end - cfg->y_start) * face->height);

    // 確保不超出邊界
    if (roi_x + roi_w > INPUT_WIDTH) roi_w = INPUT_WIDTH - roi_x;
    if (roi_y + roi_h > INPUT_HEIGHT) roi_h = INPUT_HEIGHT - roi_y;

    // 提取並 resize ROI 區域
    uint8_t roi_rgb[ROI_SIZE * ROI_SIZE * 3];
    const uint8_t* roi_src = frame + (roi_y * INPUT_WIDTH + roi_x) * 3;

    resize_bilinear(roi_src, roi_w, roi_h, roi_rgb, ROI_SIZE, ROI_SIZE, 3);

    // 轉換為 INT8: [0, 255] → [-128, 127]
    for (uint32_t i = 0; i < ROI_SIZE * ROI_SIZE * 3; i++) {
        output[i] = (int8_t)(roi_rgb[i] - 128);
    }
}

/**
 * 主預處理函數
 *
 * 輸入: RGB 幀 (640×480×3)
 * 輸出: 更新時間窗口緩衝區，當滿 8 幀時返回 true
 */
bool preprocess_frame_for_inference(const uint8_t* rgb_frame)
{
    // 1. 臉部檢測
    FaceRect face;
    if (!detect_face_simple(&face)) {
        printf("Face not detected\n");
        return false;
    }

    // 2. 當前幀索引（循環緩衝區）
    uint32_t frame_idx = frame_count % TIME_WINDOW;

    // 3. 提取 3 個 ROI
    for (uint8_t roi = 0; roi < NUM_ROIS; roi++) {
        int8_t* roi_dst = &roi_buffer[frame_idx][roi][0][0][0];
        extract_roi(rgb_frame, &face, roi, roi_dst);
    }

    frame_count++;

    // 4. 檢查是否已收集足夠幀數
    if (frame_count >= TIME_WINDOW) {
        return true;  // 可以進行推論
    }

    return false;  // 需要更多幀
}

/**
 * 將時間窗口緩衝區複製到 AI 輸入緩衝區
 *
 * 格式: (8, 3, 36, 36, 3) → 連續內存布局
 */
void copy_to_ai_input_buffer(int8_t* ai_input)
{
    // 計算起始幀索引（考慮循環緩衝區）
    uint32_t start_idx = (frame_count >= TIME_WINDOW) ?
                        (frame_count % TIME_WINDOW) : 0;

    // 複製 8 幀數據（按時間順序）
    for (uint8_t t = 0; t < TIME_WINDOW; t++) {
        uint32_t buf_idx = (start_idx + t) % TIME_WINDOW;

        for (uint8_t roi = 0; roi < NUM_ROIS; roi++) {
            uint32_t dst_offset = (t * NUM_ROIS + roi) * ROI_SIZE * ROI_SIZE * 3;
            memcpy(
                ai_input + dst_offset,
                &roi_buffer[buf_idx][roi][0][0][0],
                ROI_SIZE * ROI_SIZE * 3
            );
        }
    }
}

/**
 * 完整的預處理流程（整合到主循環）
 *
 * 用法範例:
 *   while (1) {
 *       // 獲取攝像頭幀
 *       camera_get_frame(frame_buffer);
 *
 *       // 預處理
 *       if (preprocess_and_prepare_inference(frame_buffer)) {
 *           // 運行推論
 *           run_inference();
 *       }
 *   }
 */
bool preprocess_and_prepare_inference(const uint8_t* rgb_frame)
{
    // 處理當前幀
    if (!preprocess_frame_for_inference(rgb_frame)) {
        return false;  // 需要更多幀
    }

    // 複製到 AI 輸入緩衝區
    extern int8_t* g_ai_input_buffer;  // 定義在 app_x-cube-ai.c
    copy_to_ai_input_buffer(g_ai_input_buffer);

    printf("Inference ready (frame %lu)\n", frame_count);
    return true;
}
