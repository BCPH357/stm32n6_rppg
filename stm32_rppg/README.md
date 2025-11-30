# 🚀 STM32N6 rPPG 部署

本目錄包含 STM32N6 上部署 rPPG 心率檢測系統所需的所有代碼和文檔。

## 🎯 部署架構

```
Camera (640×480 RGB) → ROI 提取 (3 × 36×36)
    ↓
Spatial CNN (NPU, INT8) × 24 次推論
    ↓ 產生 24 × 16 特徵
Temporal Fusion (CPU, C 語言)
    ↓
Heart Rate [30, 180] BPM
```

## 📁 目錄結構

```
stm32_rppg/
├── temporal_fusion/              # Temporal Fusion C 實現
│   ├── temporal_fusion.h         # 標頭檔
│   ├── temporal_fusion.c         # 實現（已驗證 PERFECT 等價）
│   ├── temporal_fusion_weights_exported.c  # 權重陣列（~200 KB）
│   └── test_temporal_fusion.c    # 單元測試
├── preprocessing/                # 前處理代碼範例
│   └── preprocessing_code.c      # ROI 提取、INT8 轉換
├── postprocessing/               # 後處理代碼範例
│   └── postprocessing_code.c     # 濾波、顯示
└── docs/                         # 部署文檔
    ├── deployment_guide.md       # 完整部署指南
    ├── cubemx_config.md          # STM32CubeMX 配置
    └── troubleshooting.md        # 故障排除
```

## 🚀 快速開始

### 1. 準備模型檔案

確保已完成量化階段，並準備好：
- `../models/spatial_cnn_int8.tflite` - INT8 量化模型
- `temporal_fusion/temporal_fusion_weights_exported.c` - C 權重

### 2. STM32CubeMX 配置

1. 打開 STM32CubeMX
2. 選擇 STM32N6 系列 MCU
3. 啟用 X-CUBE-AI 中間件
4. 導入 `spatial_cnn_int8.tflite`
5. 配置選項：
   - **Optimization**: Time (O2) 或 Default (O1) - **避免 O3**
   - **Runtime**: Neural-ART (NPU)
   - **Memory Pools**: Auto
6. 生成代碼

**詳細步驟**: 參見 `docs/cubemx_config.md`

### 3. 整合代碼

#### A. 添加 Temporal Fusion

將以下檔案複製到 STM32 項目：
```
Core/Inc/
  └── temporal_fusion.h
Core/Src/
  ├── temporal_fusion.c
  └── temporal_fusion_weights_exported.c
```

#### B. 實現應用層邏輯

參考 `preprocessing/preprocessing_code.c` 和 `postprocessing/postprocessing_code.c`：

```c
// 主循環
while (1) {
    // 1. 捕獲 8 幀影像
    // 2. 提取 3 個 ROI（前額、左右臉頰）
    // 3. Spatial CNN 推論 × 24 次
    // 4. Temporal Fusion 推論
    // 5. 顯示心率結果
}
```

### 4. 編譯與測試

1. 在 STM32CubeIDE 中編譯項目
2. Flash 到 STM32N6 開發板
3. 驗證推論結果

**故障排除**: 參見 `docs/troubleshooting.md`

## ⚙️ 關鍵配置

### X-CUBE-AI 配置（重要！）

基於 Zero-DCE 失敗經驗的教訓：

| 配置項 | 推薦值 | 原因 |
|--------|--------|------|
| **Optimization** | O1 或 O2 | ❌ 避免 O3（導致激進內存重用和緩衝區重疊） |
| **Runtime** | Neural-ART | NPU 加速 |
| **Memory Pools** | Auto | ✅ 信任工具自動分配，不手動修改 |
| **Input Data Type** | int8 | 量化模型要求 |
| **Output Data Type** | float32 | 特徵向量 |

### 內存需求

- **Spatial CNN**: ~100 KB（NPU 推論）
- **Temporal Fusion**: ~42 KB（權重） + ~10 KB（激活）
- **總計**: < 200 KB SRAM

## 📊 性能指標

**預期性能**（STM32N6 @ 600 MHz）：
- Spatial CNN 推論: ~20 ms/次（NPU）
- Temporal Fusion 推論: ~5 ms（CPU）
- 總延遲: ~500 ms（包含 8 幀捕獲）
- 幀率: ~2 Hz（心率更新頻率）

## 📝 文檔參考

| 文檔 | 說明 |
|------|------|
| `docs/deployment_guide.md` | 完整部署流程 |
| `docs/cubemx_config.md` | STM32CubeMX 詳細配置 |
| `docs/troubleshooting.md` | 常見問題與解決方案 |

## ❗ 常見問題

### Q1: NPU 推論失敗返回 ERROR？
**A**: 檢查優化級別是否為 O3，改為 O1 或 O2。參見 `docs/troubleshooting.md`。

### Q2: 內存不足？
**A**: 確認使用 Auto Memory Pools，並檢查 SRAM 配置。

### Q3: C 權重檔案太大？
**A**: `temporal_fusion_weights_exported.c` 約 200 KB，這是正常的。確保 Flash 容量足夠。

## 🔗 相關資源

- [X-CUBE-AI 官方文檔](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [STM32N6 產品頁](https://www.st.com/stm32n6)
- [Neural-ART Runtime](https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI)

---

**MCU**: STM32N6 系列
**IDE**: STM32CubeIDE
**工具**: STM32CubeMX + X-CUBE-AI 10.x
