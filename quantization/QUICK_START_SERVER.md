# 服務器端 INT8 量化 - 快速開始指南

所有文件已上傳到服務器！現在可以直接在服務器上執行量化流程。

---

## ✅ 已完成

- [x] 所有 Python 腳本已上傳
- [x] 執行腳本 `run_quantization.sh` 已上傳並設置權限
- [x] 文檔已上傳
- [x] 模型訓練已完成（`best_model.pth` 存在）
- [x] 訓練數據已準備（`ubfc_processed.pt` 存在）

---

## 🚀 執行步驟（服務器端）

### Step 1: SSH 連接到服務器

```bash
ssh miat@140.115.53.67
```

### Step 2: 進入量化目錄

```bash
cd /mnt/data_8T/ChenPinHao/server_training/quantization
```

### Step 3: 安裝 ONNX 依賴

```bash
conda activate rppg_training
pip install -r requirements_server.txt
```

預期輸出：
```
Collecting onnx>=1.19.0
Collecting onnxruntime>=1.23.0
...
Successfully installed onnx-1.19.1 onnxruntime-1.23.2
```

### Step 4: 執行量化流程

```bash
bash run_quantization.sh
```

**流程說明**：
1. 準備校準數據（200 samples，~2 分鐘）
2. 導出 FP32 ONNX（~30 秒）
3. INT8 量化（~3-5 分鐘）
4. 驗證精度（500 samples，~2 分鐘）

**總時間**：約 10-15 分鐘

### Step 5: 檢查結果

量化完成後，檢查生成的文件：

```bash
ls -lh models/
```

預期輸出：
```
rppg_fp32.onnx       # FP32 模型（~80 KB）
rppg_int8_qdq.onnx   # INT8 量化模型（~20 KB）← 用於部署
```

### Step 6: 下載量化模型（本地執行）

在本地 Windows PowerShell 執行：

```powershell
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/quantization/models/rppg_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
```

---

## 📊 預期結果

如果量化成功，應該看到類似輸出：

```
==================================================================
Quantization Workflow Completed!
==================================================================

✅ Status: SUCCESS - Quantization acceptable

Next steps:
1. Download INT8 model: models/rppg_int8_qdq.onnx
2. Use X-CUBE-AI to convert for STM32N6
3. Refer to: ../stm32n6_deployment/deployment_guide.md

==================================================================
```

**關鍵指標**：
- MAE 增加 < 2.0 BPM（可接受）
- 模型大小：FP32 ~80 KB → INT8 ~20 KB（4x 壓縮）
- 精度損失：MAE +1.0~1.5 BPM（預期範圍）

---

## ⚠️ 如果遇到問題

### 問題 1: 找不到訓練數據

```bash
# 檢查數據是否存在
ls -l /mnt/data_8T/ChenPinHao/server_training/data/ubfc_processed.pt
```

如果不存在，需要重新預處理：
```bash
cd /mnt/data_8T/ChenPinHao/server_training/
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
```

### 問題 2: ONNX 安裝失敗

```bash
# 手動安裝
conda activate rppg_training
pip install onnx==1.19.1 onnxruntime==1.23.2
```

### 問題 3: 量化精度不足

如果驗證顯示 MAE 增加 >= 2.0 BPM：

```bash
# 增加校準樣本數量到 500
python quantize_utils.py --data ../data/ubfc_processed.pt --num_samples 500
python quantize_onnx.py
python verify_quantization.py
```

### 問題 4: 權限錯誤

```bash
# 設置執行權限
chmod +x run_quantization.sh
```

---

## 📁 服務器目錄結構

```
/mnt/data_8T/ChenPinHao/server_training/
├── quantization/                  ← 新增的量化目錄
│   ├── quantize_utils.py          ← 校準數據準備
│   ├── export_onnx.py             ← ONNX 導出
│   ├── quantize_onnx.py           ← INT8 量化
│   ├── verify_quantization.py     ← 精度驗證
│   ├── run_quantization.sh        ← 執行腳本
│   ├── README_SERVER.md           ← 詳細文檔
│   ├── requirements_server.txt    ← 依賴清單
│   └── models/                    ← 生成的模型（執行後）
│       ├── rppg_fp32.onnx
│       └── rppg_int8_qdq.onnx
├── data/
│   └── ubfc_processed.pt          ← 訓練數據（已存在）
├── checkpoints/
│   └── best_model.pth             ← 訓練模型（已存在）
├── model.py
├── train.py
└── ...
```

---

## 🎯 下一步（量化完成後）

1. **下載 INT8 模型**：`models/rppg_int8_qdq.onnx`
2. **使用 X-CUBE-AI 轉換**：參考 `D:\MIAT\rppg\stm32n6_deployment\deployment_guide.md`
3. **部署到 STM32N6**：使用 O1 或 O2 優化（避免 O3！）
4. **驗證推論**：在硬件上測試心率檢測

---

## 📖 更多資源

- **詳細量化文檔**: `README_SERVER.md`（服務器端）或 `README.md`（本地）
- **STM32N6 部署指南**: `D:\MIAT\rppg\stm32n6_deployment\deployment_guide.md`
- **故障排除**: `D:\MIAT\rppg\stm32n6_deployment\troubleshooting.md`
- **項目記錄**: `D:\MIAT\rppg\CLAUDE.md`

---

**準備好了嗎？** 現在就可以在服務器上執行 `bash run_quantization.sh` 開始量化！

---

**版本**: 1.0
**創建日期**: 2025-01-20
**最後更新**: 2025-01-20
