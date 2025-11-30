# 🎓 階段 2：模型訓練

使用預處理後的數據訓練 Multi-ROI rPPG 心率檢測模型。

## 🎯 目標

訓練輕量級模型（~20K 參數），達到 MAE < 5 BPM 的心率預測精度。

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `model.py` | UltraLightRPPG 模型定義（Shared CNN + Temporal Fusion） |
| `train.py` | 訓練主程式 |
| `config.yaml` | 訓練配置（學習率、Batch size等） |

## 🚀 快速開始

### 服務器上執行

```bash
# 1. 確保已完成階段 1 的數據預處理
cd /mnt/data_8T/ChenPinHao/rppg/2_training/

# 2. 激活環境
conda activate rppg_training

# 3. 開始訓練（前台）
python train.py --config config.yaml

# 4. 或後台運行（推薦）
nohup python train.py --config config.yaml > logs/training.log 2>&1 &

# 5. 監控訓練
tail -f logs/training.log
```

## 📦 輸入/輸出

### 輸入
- `../1_preprocessing/data/ubfc_processed.pt` - 預處理數據

### 輸出
- `checkpoints/best_model.pth` - 最佳模型權重
- `checkpoints/latest_model.pth` - 最新模型權重
- `logs/training.log` - 訓練日誌

## 🏗️ 模型架構

```python
UltraLightRPPG (19,761 params)
├── Shared CNN (9,840 params)
│   ├── Conv2D(3→16) + BN + ReLU + MaxPool
│   ├── Conv2D(16→32) + BN + ReLU + MaxPool
│   └── Conv2D(32→16) + BN + ReLU + GAP
└── Temporal Fusion (9,921 params)
    ├── Reshape: (24,16) → (48,8)
    ├── Conv1D(48→32) + ReLU
    ├── Conv1D(32→16) + ReLU
    ├── FC(128→32) + ReLU
    └── FC(32→1) + Sigmoid → HR [30,180]
```

## 📊 預期結果

**目標指標**（驗證集）：
- MAE: < 5.0 BPM
- RMSE: < 8.0 BPM
- MAPE: < 8%
- R²: > 0.85

## 下一步

訓練完成後，前往 `3_model_conversion/` 進行模型拆分和轉換。
