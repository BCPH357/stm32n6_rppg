# 服務器端 INT8 量化指南

本指南說明如何在服務器 (`miat@140.115.53.67`) 上執行 rPPG 模型的 INT8 量化。

---

## 📋 前置條件

### 1. 訓練完成

確保以下文件存在於服務器上：

```bash
/mnt/data_8T/ChenPinHao/server_training/
├── data/
│   └── ubfc_processed.pt          # 訓練數據（必須）
└── checkpoints/
    └── best_model.pth              # 訓練好的模型（必須）
```

### 2. 檢查訓練狀態

```bash
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/

# 檢查數據文件
ls -lh data/ubfc_processed.pt

# 檢查模型文件
ls -lh checkpoints/best_model.pth

# 查看訓練日誌（確認 MAE < 10 BPM）
tail -50 logs/training_*.log
```

---

## 🚀 快速開始

### Step 1: 安裝 ONNX 依賴

```bash
cd /mnt/data_8T/ChenPinHao/server_training/quantization
conda activate rppg_training
pip install -r requirements_server.txt
```

**預期輸出**：
```
Successfully installed onnx-1.19.1 onnxruntime-1.23.2
```

### Step 2: 執行量化流程

```bash
bash run_quantization.sh
```

**流程說明**：
1. **準備校準數據**（~2 分鐘）- 從訓練數據中提取 200 個樣本
2. **導出 FP32 ONNX**（~30 秒）- 將 PyTorch 模型轉換為 ONNX
3. **INT8 量化**（~3-5 分鐘）- 使用 ONNX Runtime 進行量化
4. **驗證精度**（~2 分鐘）- 對比 FP32 vs INT8 性能

**總時間**：約 10-15 分鐘

### Step 3: 下載量化模型

```bash
# 在本地執行（Windows PowerShell）
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/quantization/models/rppg_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
```

---

## 📊 預期結果

### 成功案例

```
==================================================================
[Step 4/4] Verifying Quantization Accuracy
==================================================================

📊 FP32 Model Performance:
   MAE:  4.65 BPM
   RMSE: 6.63 BPM

📊 INT8 Model Performance:
   MAE:  6.12 BPM
   RMSE: 8.01 BPM

📈 Quantization Impact:
   MAE increase:  +1.47 BPM (+31.61%)
   RMSE increase: +1.38 BPM (+20.81%)

✅ Quantization ACCEPTABLE
   MAE increase (1.47 BPM) < threshold (2.0 BPM)

==================================================================
Quantization Workflow Completed!
==================================================================

✅ Status: SUCCESS - Quantization acceptable

Next steps:
1. Download INT8 model: models/rppg_int8_qdq.onnx
2. Use X-CUBE-AI to convert for STM32N6
3. Refer to: ../stm32n6_deployment/deployment_guide.md
```

### 輸出文件

```
quantization/
├── calibration_data.pt           # 校準數據（~50-100 MB）
└── models/
    ├── rppg_fp32.onnx            # FP32 ONNX（~80 KB）
    └── rppg_int8_qdq.onnx        # INT8 ONNX（~20 KB）✨ 用於部署
```

---

## ⚠️ 故障排除

### 問題 1: 找不到訓練數據

**錯誤**：
```
❌ Error: Data file not found at ../data/ubfc_processed.pt
```

**解決**：
```bash
# 檢查數據路徑
ls -l /mnt/data_8T/ChenPinHao/server_training/data/ubfc_processed.pt

# 如果不存在，重新預處理
cd /mnt/data_8T/ChenPinHao/server_training/
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
```

---

### 問題 2: 找不到訓練模型

**錯誤**：
```
❌ Error: Checkpoint not found at ../checkpoints/best_model.pth
```

**解決**：
```bash
# 檢查模型路徑
ls -l /mnt/data_8T/ChenPinHao/server_training/checkpoints/

# 如果不存在或訓練未完成，重新訓練
cd /mnt/data_8T/ChenPinHao/server_training/
bash start_training_background.sh
```

---

### 問題 3: ONNX 套件未安裝

**錯誤**：
```
ModuleNotFoundError: No module named 'onnx'
```

**解決**：
```bash
conda activate rppg_training
pip install onnx onnxruntime
```

---

### 問題 4: 量化精度不足

**錯誤**：
```
⚠️ Quantization DEGRADATION SIGNIFICANT
   MAE increase (3.2 BPM) >= threshold (2.0 BPM)
```

**解決方案 A - 增加校準樣本**：
```bash
python quantize_utils.py --data ../data/ubfc_processed.pt \
                         --output calibration_data.pt \
                         --num_samples 500

python quantize_onnx.py
python verify_quantization.py
```

**解決方案 B - 檢查訓練模型**：
```bash
# 確認訓練模型本身的性能
python -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pth')
print(f'Validation MAE: {checkpoint.get(\"val_mae\", \"N/A\")} BPM')
print(f'Epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
"
```

如果訓練 MAE > 10 BPM，量化後必然不佳，需要重新訓練。

---

## 📁 目錄結構

完成後，服務器上的目錄結構如下：

```
/mnt/data_8T/ChenPinHao/server_training/
├── quantization/                  # 量化腳本（新增）
│   ├── run_quantization.sh        # 執行腳本
│   ├── quantize_utils.py          # 校準數據準備
│   ├── export_onnx.py             # ONNX 導出
│   ├── quantize_onnx.py           # INT8 量化
│   ├── verify_quantization.py     # 精度驗證
│   ├── requirements_server.txt    # 依賴清單
│   ├── calibration_data.pt        # 校準數據（生成）
│   └── models/                    # 模型輸出（生成）
│       ├── rppg_fp32.onnx
│       └── rppg_int8_qdq.onnx     # 最終產物
├── data/
│   └── ubfc_processed.pt          # 訓練數據（已存在）
├── checkpoints/
│   └── best_model.pth             # 訓練模型（已存在）
├── preprocess_data.py
├── train.py
├── model.py
└── ...
```

---

## 🔧 手動執行（Debug 用）

如果自動腳本失敗，可以手動逐步執行：

```bash
cd /mnt/data_8T/ChenPinHao/server_training/quantization
conda activate rppg_training

# Step 1: 校準數據
python quantize_utils.py --data ../data/ubfc_processed.pt

# Step 2: 導出 FP32 ONNX
python export_onnx.py --checkpoint ../checkpoints/best_model.pth

# Step 3: INT8 量化
python quantize_onnx.py

# Step 4: 驗證精度
python verify_quantization.py --data ../data/ubfc_processed.pt
```

---

## 📖 參考資源

- **完整量化文檔**: `README.md`（本目錄）
- **STM32N6 部署指南**: `../stm32n6_deployment/deployment_guide.md`
- **項目記錄**: `../CLAUDE.md`

---

**版本**: 1.0
**創建日期**: 2025-01-20
**維護者**: Claude Code AI
