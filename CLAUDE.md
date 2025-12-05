# rPPG 心率檢測系統 - LLM 開發快速上手指南

**目的**: 讓所有 LLM 一看就能立即進入開發狀態
**架構**: Pattern A（Spatial CNN on NPU + Temporal Fusion on CPU）
**當前狀態**: ✅ 模型拆分完成 | ✅ C 實現驗證 (PERFECT) | ⏳ STM32N6 部署準備中

---

## 🎯 一分鐘快速理解

### 這是什麼項目？

**遠端光電容積描記法 (rPPG)** - 用攝像頭非接觸式檢測心率

**核心流程**:
```
Camera → Face Detection → 3 ROIs (前額、左右臉頰)
  → Spatial CNN (NPU, INT8) × 24 次
  → Temporal Fusion (CPU, FP32)
  → Heart Rate (BPM)
```

**參數量**: 僅 20K (9,840 + 10,353)
**精度**: MAE 4.65 BPM (訓練), ~5.0 BPM (量化後)
**部署目標**: STM32N6 (NPU + CPU)

---

## 📂 項目結構（重構後）

```
rppg/
├── 1_preprocessing/          # 數據前處理（UBFC-rPPG 數據集）
├── 2_training/               # 模型訓練（UltraLightRPPG）
├── 3_model_conversion/       # 模型拆分（Spatial + Temporal）
├── 4_quantization/           # 量化
│   ├── spatial_cnn/          # TFLite INT8 量化（NPU）
│   └── temporal_fusion/      # C 權重導出（CPU）
├── 5_validation/             # 精度驗證
├── stm32_rppg/               # STM32N6 部署項目
│   ├── temporal_fusion/      # C 實現（~300 行）
│   ├── preprocessing/        # ROI 提取代碼範例
│   ├── postprocessing/       # 濾波、顯示代碼範例
│   └── docs/                 # 部署文檔
├── webapp/                   # Web 即時心率監測
└── models/                   # 共享模型文件
```

**關鍵原則**: 1→2→3→4→5 順序清晰，每個階段獨立可驗證

---

## 🏗️ Pattern A 架構（當前方案）

### 為什麼採用 Pattern A？

#### ❌ 之前方案（統一模型）的問題：

```
Input (B, 72, 36, 36)
  → Shared CNN → ROI Fusion → Temporal Conv1D → Output
                 ↑ 整個模型需 INT8 量化
```

**問題**:
1. **精度損失嚴重**: Temporal 時序特徵對量化敏感，預期退化 +2-5 BPM
2. **TFLite 轉換困難**: Temporal Conv1D 可能不兼容
3. **調試困難**: 單一故障點，無法分別驗證

#### ✅ Pattern A 方案（拆分模型）：

```
┌────────────────────────────────────────┐
│ Camera (640×480 RGB)                   │
└─────────────┬──────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ ROI Extraction (3 × 36×36 patches)    │
└─────────────┬──────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ NPU: Spatial CNN (INT8 TFLite)        │
│ - 推論 24 次 (8 frames × 3 ROIs)      │
│ - 輸出: (24, 16) FP32 特徵矩陣         │
└─────────────┬──────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ CPU: Temporal Fusion (Pure C, FP32)   │
│ - Conv1D + FC Layers                  │
│ - 輸出: Heart Rate (30-180 BPM)       │
└────────────────────────────────────────┘
```

**優勢**:
- ✅ **精度**: Spatial INT8 + Temporal FP32，預期退化 < 1.0 BPM
- ✅ **轉換穩定**: Spatial 簡單 CNN → TFLite 穩定；Temporal 直接 C 導出
- ✅ **獨立驗證**: 可分別測試 NPU 和 CPU 模組
- ✅ **靈活**: 可獨立優化兩個模組

---

## 🔑 核心概念

### 1. Multi-ROI 策略

**3 個 ROI 區域**（相對臉部檢測框）：

| ROI | 位置 | 原因 |
|-----|------|------|
| **Forehead** | x: [0.20w, 0.80w]<br>y: [0.05h, 0.25h] | 血管密集，BVP 信號強 |
| **Left Cheek** | x: [0.05w, 0.30w]<br>y: [0.35h, 0.65h] | 補充信息，防遮擋 |
| **Right Cheek** | x: [0.70w, 0.95w]<br>y: [0.35h, 0.65h] | 補充信息，防遮擋 |

**為什麼 3 個 ROI？**
- 單 ROI 易受遮擋、光照影響
- 多 ROI 融合提高魯棒性
- Shared CNN 降低參數（不是 3 倍增加）

### 2. Spatial CNN (9,840 params)

**功能**: 提取單個 ROI 的空間特徵

**架構**:
```python
Input: (B, 3, 36, 36)  # 單個 ROI (RGB)
  ↓ Conv2D (16 filters, 3×3) + ReLU + MaxPool
  ↓ Conv2D (32 filters, 3×3) + ReLU + MaxPool
  ↓ Conv2D (64 filters, 3×3) + ReLU + MaxPool
  ↓ AdaptiveAvgPool → (B, 64, 4, 4)
  ↓ Flatten + FC → (B, 16)
Output: (B, 16)  # 空間特徵向量
```

**部署**: NPU (INT8 TFLite)
**文件**: `models/spatial_cnn_int8.tflite` (~20 KB)

### 3. Temporal Fusion (10,353 params)

**功能**: 融合時序特徵並預測心率

**架構**:
```python
Input: (B, 24, 16)  # 24 = 8 時間步 × 3 ROIs
  ↓ Conv1D (32 filters, kernel=3) + ReLU
  ↓ Conv1D (16 filters, kernel=3) + ReLU
  ↓ Flatten
  ↓ FC (64) → FC (32) → FC (16) → FC (1)
Output: (B, 1)  # Heart Rate (BPM)
```

**部署**: CPU (純 C 實現, FP32)
**文件**: `stm32_rppg/temporal_fusion/temporal_fusion_weights_exported.c` (~200 KB)

---

## 🚀 快速開始（針對不同任務）

### 任務 1: 我需要訓練新模型

```bash
# 服務器端（miat@140.115.53.67）
cd /mnt/data_8T/ChenPinHao/server_training/

# 1. 數據預處理（首次或數據更新時）
conda activate rppg_training
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data

# 2. 驗證數據
python validate_data.py --mode preprocessed
# 預期: Min 40-50, Max 120-150, Mean 70-90, Std 8-15

# 3. 訓練（後台）
nohup python train.py --config config.yaml > logs/training.log 2>&1 &

# 4. 監控
tail -f logs/training.log
# 目標: MAE < 5 BPM, RMSE < 8 BPM
```

**參考**: `2_training/README.md`

### 任務 2: 我需要拆分並量化模型

```bash
# 服務器端
cd /mnt/data_8T/ChenPinHao/server_training/

# Step 1: 模型拆分（從 best_model.pth）
python migrate_weights.py
# 輸出: spatial_cnn.pth, temporal_fusion.pth
# 驗證: 差異 < 1e-5

# Step 2: Spatial CNN 量化（TFLite INT8）
python export_tflite_split_v2.py
# 輸出: models/spatial_cnn_int8.tflite

# Step 3: Temporal Fusion 權重導出（C 陣列）
python export_temporal_fusion_weights.py
# 輸出: stm32_rppg/temporal_fusion/temporal_fusion_weights_exported.c

# Step 4: 驗證 C 實現
python validate_c_vs_pytorch.py
# 預期: 差異 < 1e-5 (PERFECT)
```

**參考**:
- `3_model_conversion/README.md`
- `4_quantization/spatial_cnn/README.md`
- `4_quantization/temporal_fusion/README.md`

### 任務 3: 我需要部署到 STM32N6

```bash
# 準備文件（從服務器下載）
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/spatial_cnn_int8.tflite D:\MIAT\rppg\models\
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/stm32_rppg/temporal_fusion/temporal_fusion_weights_exported.c D:\MIAT\rppg\stm32_rppg\temporal_fusion\
```

**STM32CubeMX 配置**:
1. 導入模型: `spatial_cnn_int8.tflite`
2. 驗證形狀: Input `(1, 3, 36, 36)` int8, Output `(1, 16)` float32
3. 配置:
   - Optimization: **Time (O2)** 或 **Default (O1)** ❗ **避免 O3**
   - Runtime: Neural-ART (NPU)
   - Memory Pools: Auto
4. Generate Code

**集成 Temporal Fusion**:
```c
// 複製到 STM32 項目
Core/Inc/temporal_fusion.h
Core/Src/temporal_fusion.c
Core/Src/temporal_fusion_weights_exported.c
```

**主循環邏輯**:
```c
while (1) {
    // 1. 捕獲 8 幀影像（攝像頭）
    // 2. 臉部檢測（Haar Cascade 或簡化）
    // 3. 提取 3 個 ROI（每個 36×36×3）
    // 4. Spatial CNN 推論 × 24 次（NPU）
    //    → 累積到 features[24][16] (FP32)
    // 5. Temporal Fusion 推論（CPU）
    //    → output: heart_rate (BPM)
    // 6. 後處理（濾波、顯示）
}
```

**參考**:
- `stm32_rppg/README.md` - 快速開始
- `stm32_rppg/docs/deployment_guide.md` - 完整流程
- `stm32_rppg/docs/cubemx_config.md` - CubeMX 詳細配置
- `stm32_rppg/preprocessing/preprocessing_code.c` - ROI 提取範例
- `stm32_rppg/postprocessing/postprocessing_code.c` - 後處理範例

### 任務 4: 我需要運行 Web 應用

```bash
cd D:\MIAT\rppg\webapp

# 安裝依賴（首次）
pip install -r requirements.txt

# 啟動服務器
python app.py

# 訪問
# http://localhost:5000
```

**功能**:
- 即時攝像頭捕獲（30 fps）
- Haar Cascade 臉部檢測
- 3 ROI 即時可視化
- 心率推論（~10 fps）
- BVP 波形和心率趨勢圖

**參考**: `webapp/README.md`

---

## ⚠️ 關鍵注意事項（基於血淚教訓）

### 1. ❌ 絕對不要用 O3 優化（STM32CubeMX）

**教訓來源**: Zero-DCE 部署失敗經驗（詳見 `D:\MIAT\CLAUDE.md`）

**問題**:
- Balanced (O3) 導致激進內存重用
- 緩衝區重疊（輸入/輸出相同地址）
- 推論第一次調用就返回 `LL_ATON_RT_ERROR`
- 所有手動修改 `network_*.c` 嘗試均失敗

**正確做法**:
- ✅ 使用 **Time (O2)** 或 **Default (O1)**
- ✅ Memory Pools 設為 **Auto**
- ✅ 信任 X-CUBE-AI 自動分配，不手動修改

### 2. ❌ 不要跳過分層採樣（量化校準）

**錯誤示例**:
```python
# ❌ 隨機數據校準
calibration_data = np.random.randn(100, 3, 36, 36)
```

**正確做法**:
```python
# ✅ 使用真實訓練數據 + 分層採樣
data = torch.load('data/ubfc_processed.pt')
labels = data['labels']
hr_bins = np.digitize(labels, bins=[40, 60, 80, 100, 120, 160])

for bin_id in range(1, 7):
    bin_indices = np.where(hr_bins == bin_id)[0]
    selected = np.random.choice(bin_indices, 35, replace=False)
    # 每個心率範圍都有代表
```

**結果對比**:
- 隨機數據: MAE +3.20 BPM, R² -0.37 (FAIR)
- 分層採樣: MAE +0.24 BPM, R² -0.03 (EXCELLENT)

### 3. ❌ 不要手動修改 `network_*.c`

**問題**:
- 自動生成文件，每次重新生成會被覆蓋
- 手動修改緩衝區地址無效（NPU DMA 不認）

**正確做法**:
- ✅ 在 STM32CubeMX 配置層面解決問題
- ✅ 使用腳本自動化修改（如必須）
- ✅ 保留修改腳本以便重新生成後重新應用

### 4. ❌ 不要用 ONNX → TFLite（對於複雜模型）

**問題**:
- Temporal Conv1D 可能不兼容
- 轉換鏈過長，容易出錯

**正確做法**:
- ✅ PyTorch → **Keras** → TFLite（Spatial CNN）
- ✅ PyTorch → **純 C 導出**（Temporal Fusion）
- ✅ 避免依賴 ONNX 中間格式

---

## 📊 性能指標（預期）

### 訓練階段

| 指標 | 訓練集 | 驗證集 |
|------|--------|--------|
| MAE (BPM) | ~3.5 | ~4.65 |
| RMSE (BPM) | ~5.0 | ~6.63 |
| MAPE (%) | ~3.5 | ~4.2 |
| R² | ~0.90 | ~0.86 |

### 量化階段

| 模型 | MAE | 退化 | 質量 |
|------|-----|------|------|
| FP32 Original | 4.65 BPM | - | - |
| Spatial INT8 (僅) | ~4.80 BPM | +0.15 BPM | EXCELLENT |
| **Total (Spatial INT8 + Temporal FP32)** | **~4.85 BPM** | **+0.20 BPM** | **EXCELLENT** ✅ |

**對比**: 如果全 INT8 量化，預期退化 +2-5 BPM (FAIR/POOR)

### STM32N6 部署（預期）

| 指標 | 數值 |
|------|------|
| Spatial CNN 推論 | ~20 ms/次（NPU） |
| Temporal Fusion 推論 | ~5 ms（CPU） |
| 總延遲（包含 8 幀捕獲） | ~500 ms |
| 幀率（心率更新頻率） | ~2 Hz |
| 內存占用 | < 200 KB SRAM |

---

## 🗂️ 檔案位置速查

### 服務器端（訓練與量化）

**路徑**: `/mnt/data_8T/ChenPinHao/server_training/`

```
server_training/
├── data/ubfc_processed.pt          # 預處理數據（~500 MB）
├── checkpoints/best_model.pth      # 訓練最佳模型（~80 KB）
├── models/
│   ├── spatial_cnn_int8.tflite     # Spatial CNN INT8（~20 KB）
│   └── spatial_cnn.pth              # Spatial CNN FP32
└── temporal_fusion.pth              # Temporal Fusion FP32
```

### 本地端（開發與部署）

**路徑**: `D:\MIAT\rppg\`

```
rppg/
├── models/
│   └── spatial_cnn_int8.tflite     # 從服務器下載
├── stm32_rppg/temporal_fusion/
│   ├── temporal_fusion.h
│   ├── temporal_fusion.c
│   └── temporal_fusion_weights_exported.c  # 從服務器下載
├── webapp/models/
│   └── best_model.pth              # Web 用 6D 模型
└── 1_preprocessing/data/
    └── ubfc_processed.pt           # 可選（本地驗證用）
```

---

## 🛠️ 常見問題排查

### Q1: 訓練不收斂（Loss > 100）

**可能原因**:
1. 標籤計算錯誤（PPG → HR）
2. 學習率過高
3. 數據分布異常

**檢查步驟**:
```bash
python validate_data.py --mode preprocessed
# 檢查標籤分布:
#   Min: 40-50 BPM
#   Max: 120-150 BPM
#   Mean: 70-90 BPM
#   Std: 8-15 BPM
```

**解決方案**:
- 如果標籤異常 → 檢查 `preprocess_data.py` 中的 bandpass filter 和 peak detection
- 如果標籤正常 → 降低學習率（1e-3 → 1e-4）

### Q2: 量化後精度下降嚴重（MAE +5 BPM）

**可能原因**:
1. 校準數據不正確（隨機數據或分布不均）
2. 量化配置錯誤

**檢查步驟**:
```python
# 確認校準數據分布
print("Calibration HR distribution:")
print(f"  Min: {calibration_labels.min()}")
print(f"  Max: {calibration_labels.max()}")
print(f"  Mean: {calibration_labels.mean()}")
# 應該涵蓋 40-160 BPM
```

**解決方案**:
- 使用 `4_quantization/spatial_cnn/export_tflite_split_v2.py` 中的分層採樣
- 確保 QDQ 格式 + Per-channel 量化

### Q3: STM32CubeMX 導入失敗（INTERNAL ERROR）

**錯誤信息**:
```
INTERNAL ERROR: Unexpected combination of configuration and input shape
```

**原因**: 模型輸入不是 4D 或使用了不支持的 op

**解決方案**:
1. 確認使用 `spatial_cnn_int8.tflite`（不是 6D/4D 統一模型）
2. 驗證輸入形狀: `(1, 3, 36, 36)` int8
3. 使用 Netron 檢查模型結構（無不支持 op）

### Q4: NPU 推論返回 ERROR (ret=0)

**症狀**:
```c
LL_ATON_RT_RunEpochBlock() 第一次調用返回 0
```

**90% 可能原因**: 使用了 O3 優化

**解決方案**:
1. STM32CubeMX → X-CUBE-AI → Optimization: 改為 **O2** 或 **O1**
2. 重新 Analyze + Generate Code
3. 重新編譯

**參考**: `stm32_rppg/docs/troubleshooting.md`

---

## 📚 參考文檔

### 項目文檔

| 文檔 | 說明 |
|------|------|
| `README.md` | 項目主頁（繁體中文） |
| `DEVELOPMENT_LOG.md` | 完整開發歷史（9 個 Phase） |
| `CLAUDE.md` | **本文件** - LLM 快速上手 |

### 階段文檔（繁體中文）

| 階段 | 文檔 | 說明 |
|------|------|------|
| 1 | `1_preprocessing/README.md` | 數據下載、ROI 提取、標籤計算 |
| 2 | `2_training/README.md` | 模型訓練、超參數、指標 |
| 3 | `3_model_conversion/README.md` | 模型拆分、權重遷移 |
| 4 | `4_quantization/spatial_cnn/README.md` | Spatial CNN TFLite 量化 |
| 4 | `4_quantization/temporal_fusion/README.md` | Temporal Fusion C 導出 |
| 5 | `5_validation/README.md` | 精度驗證、ROI 測試 |

### STM32 部署文檔

| 文檔 | 說明 |
|------|------|
| `stm32_rppg/README.md` | 快速開始指南 |
| `stm32_rppg/docs/deployment_guide.md` | 完整部署流程 |
| `stm32_rppg/docs/cubemx_config.md` | STM32CubeMX 詳細配置 |
| `stm32_rppg/docs/troubleshooting.md` | 常見問題排查（含 Zero-DCE 教訓） |

### 外部資源

- **X-CUBE-AI 官方文檔**: https://www.st.com/en/embedded-software/x-cube-ai.html
- **STM32N6 產品頁**: https://www.st.com/stm32n6
- **UBFC-rPPG 數據集**: https://sites.google.com/view/ybenezeth/ubfcrppg
- **ME-rPPG 論文**: https://arxiv.org/abs/2504.01774

---

## 🎯 LLM 任務指引

### 當用戶說："我要訓練模型"

1. 確認服務器連接: `ssh miat@140.115.53.67`
2. 導航到: `cd /mnt/data_8T/ChenPinHao/server_training/`
3. 檢查數據: `ls -lh data/ubfc_processed.pt`
4. 參考: `2_training/README.md`
5. 執行訓練或監控進度

### 當用戶說："模型精度不好"

1. 檢查訓練 log（MAE, RMSE）
2. 驗證數據標籤分布（40-160 BPM）
3. 檢查是否用了分層採樣（量化）
4. 參考: `5_validation/README.md`

### 當用戶說："STM32 推論失敗"

1. **第一反應**: 檢查是否用了 O3 優化 ❗
2. 檢查模型輸入形狀（應為 `(1, 3, 36, 36)` int8）
3. 檢查 log 中的錯誤信息
4. 參考: `stm32_rppg/docs/troubleshooting.md`

### 當用戶說："量化精度下降太多"

1. 確認用了真實數據校準（不是隨機數據）
2. 確認分層採樣（涵蓋所有 HR 範圍）
3. 確認 QDQ 格式 + Per-channel
4. 參考: `4_quantization/spatial_cnn/README.md`

### 當用戶說："我要修改模型結構"

1. 修改: `2_training/model.py` 中的 `UltraLightRPPG` 類
2. 重新訓練: `python train.py`
3. 重新拆分: `python migrate_weights.py`
4. 重新量化: `python export_tflite_split_v2.py`
5. 驗證等價性: 每步都要檢查差異 < 1e-5

---

## ✅ 檢查清單（部署前）

### 訓練階段

- [ ] 數據預處理完成（`data/ubfc_processed.pt` 存在）
- [ ] 標籤分布正常（Min 40-50, Max 120-150, Mean 70-90）
- [ ] 訓練收斂（MAE < 5 BPM, RMSE < 8 BPM）
- [ ] 驗證集精度良好（MAE < 6 BPM）

### 模型拆分階段

- [ ] 權重遷移完成（`spatial_cnn.pth`, `temporal_fusion.pth`）
- [ ] 等價性驗證通過（差異 < 1e-5）

### 量化階段

- [ ] Spatial CNN TFLite 生成（`spatial_cnn_int8.tflite`）
- [ ] 校準數據使用真實數據 + 分層採樣
- [ ] 量化精度驗證（MAE 增加 < 1.5 BPM, EXCELLENT/GOOD）
- [ ] Temporal Fusion C 權重導出（`temporal_fusion_weights_exported.c`）
- [ ] C 實現驗證（差異 < 1e-5, PERFECT）

### STM32N6 部署階段

- [ ] STM32CubeMX 導入成功（無 ERROR）
- [ ] 輸入形狀正確（`(1, 3, 36, 36)` int8）
- [ ] 優化級別設為 **O2** 或 **O1**（❗ 不是 O3）
- [ ] Memory Pools 設為 Auto
- [ ] Temporal Fusion C 代碼集成
- [ ] 應用層邏輯實現（ROI 提取 + 推論循環）
- [ ] 編譯通過（無錯誤）
- [ ] 推論成功（返回合理 HR 值）

---

---

## 🚨 STM32N6 Camera Display 重大問題發現 (2025-12-05)

### 問題描述

在 STM32N6570-DK 上部署 IMX335 camera display 時，發現 **IMX335_Init() 中寫入 MODE_SELECT = 0x00 (STREAMING) 會導致系統異常**。

### 症狀

```
[IMX335_Init] Writing MODE_SELECT reg (0x3000) = 0x00...
[IMX335_Init] MODE_SELECT written, delaying 20ms...
=== AFTER MODE_SELECT ===
start busy-wa                    ← UART 輸出被截斷
(系統掛起)
```

**關鍵發現**：
- UART printf 在某個隨機字符數被截斷（10-42 字符不等）
- 加入 `HAL_Delay()` 無效
- 立即寫回 STANDBY 也無效
- **完全跳過 MODE_SELECT 寫入則正常**

### 根本原因

當 IMX335 收到 `MODE_SELECT = 0x00` 命令：
1. **IMX335 立即啟動 MIPI CSI-2 輸出**
2. **STM32N6 的 CSI 接收器檢測到信號**
3. **但某些系統配置阻止了 CSI 正常運作**
4. **導致系統異常**（可能是中斷風暴、DMA 錯誤、或資源隔離衝突）
5. **UART 輸出被中斷**

### 官方 vs 我們的專案對比

| 項目 | 官方 (DCMIPP_ContinuousMode) | 我們 (rppg) | 結果 |
|------|-------------------------------|-------------|------|
| DCMIPP 配置 | ✅ 完全相同 | ✅ 完全相同（多了 downsize） | - |
| IMX335_Init | ✅ **寫入 MODE_SELECT** | ❌ 寫入會崩潰 | **差異** |
| SystemIsolation | ❌ 沒有 | ⚠️ **有調用** | **可疑** |
| BSP_CAMERA_HwReset | ❌ 沒有單獨調用 | ⚠️ **有調用** | **可疑** |
| LCD 初始化順序 | DCMIPP → IMX335 → LCD | **LCD → DCMIPP → IMX335** | **不同** |

### 初始化順序對比

**官方專案**：
```
1. DCMIPP_Init
2. IMX335_Probe (含 MODE_SELECT) ← 正常
3. LCD_Init
4. ISP_Init
5. DCMIPP_Start
6. ISP_Start
```

**我們的專案**：
```
1. LTDC_Init (LCD)
2. SystemIsolation_Config() ← ⚠️ 最可疑
3. DCMIPP_Init
4. BSP_CAMERA_HwReset() ← ⚠️ 官方沒有
5. IMX335_Probe (含 MODE_SELECT) ← ❌ 崩潰
6. ISP_Init ← 未執行到
```

### 最可能的原因 ⭐⭐⭐

**SystemIsolation_Config() 配置的資源隔離機制阻止了 CSI PHY 的正常運作**

當 IMX335 開始 streaming：
- CSI PHY 產生中斷或 DMA 請求
- 被 RIF (Resource Isolation Framework) 隔離機制阻擋
- 導致系統異常

### 驗證測試結果

```c
// 測試：完全跳過 MODE_SELECT 寫入
printf("[IMX335_Init] SKIP MODE_SELECT - Testing without streaming...\r\n");
// (不寫入 MODE_SELECT = 0x00)

結果：✅ 所有 printf 完整輸出，系統正常
```

**結論**：問題確定與 MODE_SELECT 啟動 sensor streaming 直接相關。

### 建議解決方案

#### 方案 1：檢查 SystemIsolation_Config() ⭐⭐⭐ (優先)

1. 臨時註解掉 `SystemIsolation_Config()`
2. 測試是否能正常寫入 MODE_SELECT
3. 如果可以，調整 RIF 配置以允許 CSI/DCMIPP 訪問

#### 方案 2：移除 BSP_CAMERA_HwReset()

官方專案沒有單獨調用，可能在其他地方已處理或不需要。

#### 方案 3：調整初始化順序

改成與官方相同：DCMIPP → IMX335 → LCD → ISP

#### 方案 4：延遲 MODE_SELECT 寫入

在 IMX335_Init() 中不寫入 MODE_SELECT，等所有系統準備好後再啟動 streaming。

### 關鍵教訓

1. ❌ **不要假設官方範例的每個步驟都可以隨意調整**
2. ❌ **SystemIsolation_Config() 可能影響外設訪問權限**
3. ❌ **初始化順序很重要，特別是涉及硬體信號的外設**
4. ✅ **遇到問題時，對比官方專案的每個細節差異**
5. ✅ **使用 printf 截斷位置來判斷系統異常發生時間點**

### 下一步行動

1. 檢查 `SystemIsolation_Config()` 具體實現
2. 嘗試註解掉並測試
3. 如果有效，調整 RIF 配置
4. 確認正確的初始化順序

---

**文檔版本**: 1.1
**創建日期**: 2025-12-01
**最後更新**: 2025-12-05 - 添加 Camera Display 問題發現
**目的**: 讓所有 LLM 快速進入 rPPG 項目開發狀態
**維護者**: Claude Code AI

**使用建議**:
- 📌 新 LLM session 先讀這份文件
- 📌 遇到問題先查「常見問題排查」
- 📌 執行任務前先看「任務指引」
- 📌 部署前務必過一遍「檢查清單」
- 🚨 遇到 Camera Display 問題先看「重大問題發現」
