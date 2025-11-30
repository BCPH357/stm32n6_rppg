# rPPG Multi-ROI 模型 INT8 量化指南

本目錄包含將 PyTorch Multi-ROI rPPG 模型量化為 INT8 的完整工具和流程。

---

## 📋 目錄

- [概述](#概述)
- [環境設置](#環境設置)
- [快速開始](#快速開始)
- [詳細流程](#詳細流程)
- [預期結果](#預期結果)
- [故障排除](#故障排除)
- [下一步](#下一步)

---

## 概述

### 為什麼需要 INT8 量化？

**STM32N6 NPU 僅支持 INT8 運算**。使用量化模型可以：
- ✅ **啟用 NPU 加速**：5-10x 推論速度提升
- ✅ **減少模型大小**：4x 壓縮（80 KB → 20 KB）
- ✅ **降低內存使用**：激活值佔用減半
- ✅ **降低功耗**：~30-50% 功耗降低

### 量化方法

使用 **Post-Training Quantization (PTQ)**：
- **優點**：無需重新訓練，30 分鐘內完成
- **預期精度損失**：< 2 BPM（MAE 從 4.65 → 6-7 BPM）
- **適用場景**：小模型（<100K 參數）

---

## 環境設置

### 必要條件

1. **Conda 環境**：`zerodce_tf`
   ```bash
   conda activate zerodce_tf
   ```

2. **已安裝套件**（已在環境中）：
   - `onnx` 1.19.1
   - `onnxruntime` 1.23.2
   - `torch` (PyTorch)
   - `numpy`

3. **訓練好的模型**：
   - 路徑：`../server_training/checkpoints/best_model.pth`
   - 必須存在且訓練完成

4. **預處理數據**：
   - 路徑：`../server_training/data/ubfc_processed.pt`
   - 用於校準和驗證

---

## 快速開始

### 一鍵執行（Windows）

```bash
cd D:\MIAT\rppg\quantization
test_quantization.bat
```

這將自動執行 4 個步驟：
1. 準備校準數據集（200 samples）
2. 導出 FP32 ONNX 模型
3. 執行 INT8 量化
4. 驗證量化精度

**預計時間**：5-10 分鐘

---

## 詳細流程

### Step 1: 準備校準數據集

**目的**：為量化過程提供代表性數據樣本

```bash
python quantize_utils.py
```

**參數**：
- `--data`: 訓練數據路徑（默認：`../server_training/data/ubfc_processed.pt`）
- `--output`: 校準數據輸出（默認：`calibration_data.pt`）
- `--num_samples`: 校準樣本數（默認：200）

**輸出**：
- `calibration_data.pt`（~50-100 MB）
- 包含 200 個分層採樣的樣本（涵蓋 40-160 BPM）

**說明**：
- 使用分層採樣確保各心率範圍都有代表
- 不使用測試集，只用訓練集的子集
- 200 samples 對於 20K 參數模型已足夠

---

### Step 2: 導出 FP32 ONNX 模型

**目的**：將 PyTorch 模型轉換為 ONNX 格式（量化前）

```bash
python export_onnx.py
```

**參數**：
- `--checkpoint`: PyTorch 模型路徑（默認：`../server_training/checkpoints/best_model.pth`）
- `--output`: ONNX 輸出路徑（默認：`models/rppg_fp32.onnx`）
- `--opset`: ONNX opset 版本（默認：13，推薦）

**輸出**：
- `models/rppg_fp32.onnx`（~80 KB）
- ONNX opset 13 格式（支持 BatchNorm folding）

**說明**：
- Opset 13 是 X-CUBE-AI 推薦版本
- `do_constant_folding=True` 優化常量運算
- 動態 batch 維度（適配不同批次大小）

---

### Step 3: 執行 INT8 量化

**目的**：使用 ONNX Runtime 進行 INT8 量化（QDQ 格式）

```bash
python quantize_onnx.py
```

**參數**：
- `--input`: FP32 ONNX 模型（默認：`models/rppg_fp32.onnx`）
- `--output`: INT8 ONNX 輸出（默認：`models/rppg_int8_qdq.onnx`）
- `--calibration`: 校準數據路徑（默認：`calibration_data.pt`）
- `--per_channel`: 使用 per-channel 量化（默認：True）

**輸出**：
- `models/rppg_int8_qdq.onnx`（~20 KB）
- QDQ（Quantize-DeQuantize）格式
- 4x 壓縮

**說明**：
- **QDQ 格式**：X-CUBE-AI 完全支持
- **Per-channel 量化**：準確度更高（相比 per-tensor）
- **量化配置**：
  - 權重：INT8 signed symmetric
  - 激活：INT8 signed asymmetric
- `optimize_model=False`：避免 X-CUBE-AI 解析問題

---

### Step 4: 驗證量化精度

**目的**：對比 FP32 vs INT8 模型精度，確保可接受

```bash
python verify_quantization.py
```

**參數**：
- `--fp32`: FP32 ONNX 模型（默認：`models/rppg_fp32.onnx`）
- `--int8`: INT8 ONNX 模型（默認：`models/rppg_int8_qdq.onnx`）
- `--data`: 測試數據路徑（默認：`../server_training/data/ubfc_processed.pt`）
- `--num_samples`: 測試樣本數（默認：500）

**輸出報告**：
```
📊 FP32 Model Performance:
   MAE:  4.65 BPM
   RMSE: 6.63 BPM

📊 INT8 Model Performance:
   MAE:  6.12 BPM
   RMSE: 8.01 BPM

📈 Quantization Impact:
   MAE increase:  +1.47 BPM (+31.61%)
   RMSE increase: +1.38 BPM (+20.81%)

🔍 Output Difference (FP32 vs INT8):
   Mean:  1.23 BPM
   Max:   5.67 BPM
   Std:   1.05 BPM

✅ Quantization ACCEPTABLE
   MAE increase (1.47 BPM) < threshold (2.0 BPM)
```

**可接受標準**：
- ✅ MAE 增加 < 2 BPM
- ✅ 輸出差異平均 < 2 BPM
- ⚠️  超過閾值則考慮增加校準樣本或使用 QAT

---

## 預期結果

### 文件輸出

完成後應有以下文件：

```
quantization/
├── calibration_data.pt           # 校準數據（~50-100 MB）
└── models/
    ├── rppg_fp32.onnx            # FP32 ONNX（~80 KB）
    └── rppg_int8_qdq.onnx        # INT8 ONNX（~20 KB）✨ 用於 X-CUBE-AI
```

### 性能預測

| 指標 | FP32 | INT8 (PTQ) |
|------|------|-----------|
| **MAE** | 4.65 BPM | **5.5-7.0 BPM** |
| **RMSE** | 6.63 BPM | **7.5-9.0 BPM** |
| **模型大小** | 80 KB | **20 KB** (4x 壓縮) |
| **推論速度** | ~50 ms (CPU) | **~5-10 ms (NPU)** |
| **內存使用** | ~500 KB | **~200 KB** |

---

## 故障排除

### 問題 1: 找不到訓練數據

**錯誤訊息**：
```
❌ Error: Data file not found at ../server_training/data/ubfc_processed.pt
```

**解決方案**：
1. 確認訓練數據已預處理完成
2. 檢查路徑是否正確
3. 如需重新預處理：
   ```bash
   cd ../server_training
   python preprocess_data.py --dataset ubfc
   ```

---

### 問題 2: 找不到訓練模型

**錯誤訊息**：
```
❌ Error: Checkpoint not found at ../server_training/checkpoints/best_model.pth
```

**解決方案**：
1. 確認模型訓練已完成
2. 檢查檢查點路徑
3. 如需重新訓練：
   ```bash
   cd ../server_training
   bash run_training.sh
   ```

---

### 問題 3: 量化精度不足

**症狀**：
```
⚠️ Quantization DEGRADATION SIGNIFICANT
   MAE increase (3.2 BPM) >= threshold (2.0 BPM)
```

**解決方案**：

**方案 A：增加校準樣本**
```bash
python quantize_utils.py --num_samples 500
python quantize_onnx.py
python verify_quantization.py
```

**方案 B：使用 QAT（Quantization-Aware Training）**
- 需要重新訓練模型
- 預期精度損失 < 1 BPM
- 時間成本：數小時

**方案 C：調整量化參數**
- 嘗試關閉 per-channel：`python quantize_onnx.py --per_channel=False`
- 使用不同校準數據分布

---

### 問題 4: ONNX Runtime 錯誤

**錯誤訊息**：
```
Error during quantization: ...
```

**解決方案**：
1. 檢查 ONNX Runtime 版本：
   ```bash
   pip show onnxruntime
   ```
   應為 1.23.x

2. 重新安裝套件：
   ```bash
   pip install --upgrade onnxruntime onnx
   ```

3. 如仍失敗，檢查 ONNX 模型結構：
   ```bash
   python -c "import onnx; onnx.checker.check_model('models/rppg_fp32.onnx')"
   ```

---

## 下一步

### X-CUBE-AI 轉換

量化完成後，使用 **INT8 ONNX 模型**進行 X-CUBE-AI 轉換：

**文件**：`models/rppg_int8_qdq.onnx` ✨

**步驟**：
1. 參考 `../stm32n6_deployment/deployment_guide.md`
2. 在 STM32CubeMX 中配置 X-CUBE-AI
3. 選擇優化級別：**O1 或 O2**（避免 O3！）
4. 生成代碼並部署到 STM32N6

---

## 參考資源

### 官方文檔
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [X-CUBE-AI User Manual](https://www.st.com/en/embedded-software/x-cube-ai.html)

### 相關文件
- `../stm32n6_deployment/deployment_guide.md` - STM32N6 完整部署指南
- `../stm32n6_deployment/troubleshooting.md` - 故障排除（基於 Zero-DCE 經驗）
- `../CLAUDE.md` - 項目開發記錄

---

**版本**: 1.0
**創建日期**: 2025-01-20
**維護者**: Claude Code AI
