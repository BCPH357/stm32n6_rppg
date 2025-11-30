# ✅ 階段 5：模型驗證

驗證量化後模型的精度和功能正確性。

## 🎯 目標

1. 評估 INT8 量化對 Spatial CNN 的影響
2. 驗證 ROI 提取邏輯的正確性
3. 確認端到端流程可用

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `evaluate_quantized_model.py` | 評估量化模型精度（與 PyTorch 對比） |
| `test_roi_extraction.py` | 測試 ROI 提取邏輯（可視化） |

## 🚀 執行驗證

### 1. 評估量化精度

```bash
cd /mnt/data_8T/ChenPinHao/rppg/5_validation/

conda activate rppg_training

python evaluate_quantized_model.py
```

**預期輸出**：
```
[精度評估]
  PyTorch FP32 MAE: 4.65 BPM
  TFLite INT8 MAE: 5.12 BPM
  差異: +0.47 BPM
  質量評級: EXCELLENT
```

### 2. 測試 ROI 提取

```bash
python test_roi_extraction.py --input test_video.avi --visualize
```

會顯示帶有 ROI 標記的影像：
- 紅框: Forehead
- 藍框: Left Cheek
- 橙框: Right Cheek

## 📊 驗證標準

### 量化精度

| 質量等級 | MAE 增加 | 說明 |
|----------|----------|------|
| **PERFECT** | < 0.5 BPM | 幾乎無損失 |
| **EXCELLENT** | < 1.5 BPM | 優秀 |
| **GOOD** | < 3.0 BPM | 可接受 |
| **POOR** | ≥ 3.0 BPM | 需要重新量化 |

### ROI 提取準確性

- ✅ 臉部檢測率 > 95%
- ✅ ROI 位置穩定（抖動 < 5 像素）
- ✅ 無明顯偏移或遮擋

## 📝 下一步

驗證通過後，前往 `stm32_rppg/` 準備 STM32N6 部署。

---

**環境要求**: Python 3.8+, PyTorch 2.0+, OpenCV 4.8+
**參見**: `../requirements_rppg_training.txt`
