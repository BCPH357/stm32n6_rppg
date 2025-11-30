# 🔄 階段 3：模型轉換

將訓練好的模型拆分為 Spatial CNN 和 Temporal Fusion，並轉換為適合 STM32N6 部署的格式。

## 🎯 目標

1. 拆分模型：UltraLightRPPG → SpatialCNN + TemporalFusion
2. 遷移權重：無需重新訓練，直接拷貝權重
3. 轉換為 4D 格式：符合 X-CUBE-AI 輸入限制

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `model_split.py` | 定義 SpatialCNN 和 TemporalFusion 類別 |
| `migrate_weights.py` | 從 best_model.pth 遷移權重到拆分模型 |
| `convert_to_4d_for_stm32.py` | 將 6D 模型轉換為 4D ONNX（可選） |

## 🚀 執行流程

### Step 1: 模型拆分與權重遷移

```bash
# 在服務器上執行
cd /mnt/data_8T/ChenPinHao/rppg/3_model_conversion/

conda activate rppg_training

python migrate_weights.py
```

**輸出**：
- `checkpoints/spatial_cnn.pth` - Spatial CNN 權重
- `checkpoints/temporal_fusion.pth` - Temporal Fusion 權重
- `checkpoints/combined_model.pth` - 組合模型（驗證用）

### Step 2: 驗證等價性

腳本會自動驗證拆分後的模型是否與原始模型等價（差異 < 1e-5）。

**預期輸出**：
```
✅ 驗證通過！拆分模型與原始模型等價
最大差異: 0.00000123 BPM
```

## 📦 模型拆分說明

### Spatial CNN (9,840 params)
- **輸入**: (B, 3, 36, 36) - 單個 ROI 的 RGB 影像
- **輸出**: (B, 16) - 空間特徵向量
- **部署**: STM32N6 NPU（INT8 量化）

### Temporal Fusion (10,353 params)
- **輸入**: (B, 24, 16) - 24 個特徵向量（8 時間步 × 3 ROI）
- **輸出**: (B, 1) - 心率 [30, 180] BPM
- **部署**: STM32N6 CPU（純 C 實現）

## 📝 下一步

完成模型拆分後：
1. Spatial CNN → 前往 `4_quantization/spatial_cnn/` 進行 TFLite 量化
2. Temporal Fusion → 前往 `4_quantization/temporal_fusion/` 導出 C 權重

---

**環境要求**: Python 3.8+, PyTorch 2.0+
**參見**: `../requirements_rppg_training.txt`
