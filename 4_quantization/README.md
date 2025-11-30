# ⚡ 階段 4：模型量化

將拆分後的模型量化為適合 STM32N6 部署的格式。

## 🎯 目標

1. **Spatial CNN**: 導出為 INT8 量化的 TFLite 模型（NPU 推論）
2. **Temporal Fusion**: 導出權重為 C 語言陣列（CPU 推論）

## 📁 目錄結構

```
4_quantization/
├── spatial_cnn/           # Spatial CNN TFLite 量化
│   ├── export_tflite_split_v2.py
│   └── validate_tflite.py
└── temporal_fusion/       # Temporal Fusion C 權重導出
    ├── export_temporal_fusion_weights.py
    ├── validate_c_vs_pytorch.py
    └── debug_c_implementation.py
```

## 🚀 執行流程

### Part A: Spatial CNN 量化（TFLite INT8）

#### 環境準備

```bash
# 創建 TFLite 導出環境（僅首次）
conda create -n tflite_export python=3.10
conda activate tflite_export
pip install -r ../requirements_tflite_export.txt
```

#### 執行導出

```bash
cd spatial_cnn/

# 導出 TFLite INT8 模型
python export_tflite_split_v2.py

# 驗證量化精度
python validate_tflite.py
```

**輸出**：
- `../../models/spatial_cnn_int8.tflite` - INT8 量化模型（~20 KB）

**預期結果**：
- MAE 增加: < 1.5 BPM
- 模型大小: 80 KB → 20 KB（4x 壓縮）

---

### Part B: Temporal Fusion 權重導出（C 語言）

#### 執行導出

```bash
cd temporal_fusion/

# 激活 rPPG 訓練環境
conda activate rppg_training

# 導出權重為 C 陣列
python export_temporal_fusion_weights.py
```

**輸出**：
- `../../stm32_rppg/temporal_fusion/temporal_fusion_weights_exported.c` - C 權重文件（~200 KB）

#### 驗證 C 實現

```bash
# 在服務器上編譯並驗證
python validate_c_vs_pytorch.py
```

**預期結果**：
```
[差異統計]
  最大差異: 0.00001526 BPM
  平均差異: 0.00000496 BPM
  質量: PERFECT (< 1e-5)
```

## 📊 量化方法說明

### Spatial CNN: Post-Training Quantization (PTQ)

- **方法**: Full INT8 Quantization
- **校準數據**: 分層採樣 100 樣本（確保各 HR 範圍均有代表）
- **格式**: QDQ (Quantize-Dequantize) + Per-channel 量化

### Temporal Fusion: 權重導出

- **格式**: C 語言浮點數陣列（FP32）
- **結構**:
  - Conv1D 權重: `[out_ch][in_ch][kernel]`
  - FC 權重: `[out_dim][in_dim]`
  - Bias: `[out_dim]`

## 📝 下一步

完成量化後：
1. 前往 `5_validation/` 進行最終精度驗證
2. 將模型部署到 `stm32_rppg/` STM32N6 項目

---

**環境要求**:
- TFLite 導出: TensorFlow 2.13.1, PyTorch 2.0+
- C 權重導出: PyTorch 2.0+

**參見**:
- `../requirements_tflite_export.txt`
- `../requirements_rppg_training.txt`
