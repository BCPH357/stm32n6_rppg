# rPPG 心率檢測系統

**項目**: 遠端光電容積描記法 (Remote Photoplethysmography, rPPG) 心率檢測
**目標**: 開發基於攝像頭的非接觸式心率檢測系統，並部署到 STM32N6 嵌入式平台
**當前狀態**: ✅ Web 應用完成 | 🔄 6D→4D 模型轉換完成 | ⏳ STM32N6 部署準備中

---

## 📋 快速導航

- [項目概述](#項目概述)
- [技術限制](#技術限制)
- [工作方法](#工作方法)
- [待辦事項](#待辦事項)
- [參考資源](#參考資源)

---

## 項目概述

### 核心功能

本項目實現一個完整的 rPPG 心率檢測系統，包括：

1. **數據處理與訓練**（服務器端）
   - UBFC-rPPG 數據集處理（42 subjects）
   - Multi-ROI 特徵提取（前額、左右臉頰）
   - 健壯的 PPG → HR 標籤計算（Bandpass + Peak Detection）
   - 模型訓練（MAE: 4.65 BPM）

2. **Web 應用**（即時心率監測）
   - Flask + WebSocket 後端架構
   - 攝像頭即時捕獲（30 fps）
   - Haar Cascade 臉部檢測
   - Multi-ROI 推論（~10 fps）
   - 即時圖表顯示（Chart.js）

3. **嵌入式部署**（STM32N6）
   - 6D → 4D 模型轉換（符合 X-CUBE-AI 限制）
   - INT8 量化（QDQ 格式，MAE 增加僅 +0.24 BPM）
   - NPU 加速推論

### 模型架構

**Multi-ROI rPPG 模型**（~20K 參數）：

```
Input: (B, 8, 3, 36, 36, 3)
  ↓ 6D 版本（訓練）
[Shared CNN] 提取空間特徵
  ↓
[ROI Fusion] 融合 3 個區域
  ↓
[Temporal Conv1D] 時序建模
  ↓
[FC Layers] 預測心率
  ↓
Output: (B, 1) HR (BPM)

Input: (B, 72, 36, 36)
  ↓ 4D 版本（STM32 部署）
[Reshape to 6D] → 使用相同權重 → 輸出相同結果
```

**關鍵特性**：
- ✅ Shared CNN：所有 ROI 共享權重（減少參數）
- ✅ 輕量級：僅 20K 參數（遠低於 500K 目標）
- ✅ 時間建模：Conv1D 捕捉心率時序依賴
- ✅ 雙版本：6D（訓練/Web）+ 4D（STM32）權重一致

### ROI 提取邏輯

| ROI | 位置（相對臉部框） | 顏色標記 |
|-----|-------------------|---------|
| **Forehead** | x: [0.20w, 0.80w]<br>y: [0.05h, 0.25h] | 紅色 |
| **Left Cheek** | x: [0.05w, 0.30w]<br>y: [0.35h, 0.65h] | 藍色 |
| **Right Cheek** | x: [0.70w, 0.95w]<br>y: [0.35h, 0.65h] | 橙色 |

**處理流程**：
1. Haar Cascade 檢測臉部 bbox
2. 計算 3 個 ROI 坐標
3. 裁切並調整到 36×36
4. 歸一化到 [0, 1]
5. 堆疊為 `(3, 36, 36, 3)`

---

## 技術限制

### STM32N6 & X-CUBE-AI 限制

#### 1. X-CUBE-AI 輸入維度限制

**核心問題**：X-CUBE-AI 只支持最多 **4D 張量**

**證據**（`ai_platform.h:462-469`）：
```c
#define AI_BUFFER_OBJ_INIT(format_, h_, w_, ch_, n_batches_, data_) \
{ \
  .shape = AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, (n_batches_), (ch_), (w_), (h_)), \
}
```

**影響**：
- 原始 6D 輸入 `(B, 8, 3, 36, 36, 3)` 無法直接導入
- STM32CubeMX 報錯：`INTERNAL ERROR: Unexpected combination of configuration and input shape`

**解決方案**：
- 創建 4D 版本模型：`(B, 72, 36, 36)` 其中 72 = 8×3×3（T×ROI×C）
- 4D 模型內部 reshape 回 6D 處理
- 權重完全共享（輸出差異 < 1e-5）

#### 2. 優化級別限制（基於 Zero-DCE 教訓）

**避免使用 Balanced (O3)**：
- ❌ 導致激進內存重用
- ❌ 緩衝區重疊（輸入/輸出相同地址）
- ❌ 推論第一次調用就返回 `LL_ATON_RT_ERROR`
- ❌ 所有手動修改 `network_*.c` 嘗試均失敗

**推薦配置**：
- ✅ Time (O2) 或 Default (O1)
- ✅ Memory Pools 設為 Auto（不手動修改）
- ✅ 信任 X-CUBE-AI 自動分配

#### 3. 量化限制

**Post-Training Quantization (PTQ)**：
- 需要校準數據（使用真實訓練數據，非隨機數據）
- 必須使用分層採樣（確保各 HR 範圍都有代表）
- QDQ 格式 + Per-channel 量化效果最佳
- 預期精度損失：MAE +0.5~1.5 BPM

**實際結果**（4D 模型量化）：
- MAE 增加：僅 **+0.24 BPM**（EXCELLENT）
- 模型大小：80 KB → 20 KB（4x 壓縮）

### Web 應用限制

**環境要求**：
- 光線充足（避免逆光、暗光）
- 臉部正對攝像頭（±15° 偏轉可接受）
- 保持相對靜止（輕微點頭 OK）
- 建議距離：50-100 cm

**已知問題**：
- Haar Cascade 對側臉、遮擋敏感
- 需要 8 幀才能開始推論（~0.8 秒延遲）
- 深色皮膚可能影響 BVP 信噪比

---

## 工作方法

### 服務器端訓練流程

**服務器信息**：
- 路徑：`/mnt/data_8T/ChenPinHao/server_training/`
- 連接：`ssh miat@140.115.53.67`
- 環境：`conda activate rppg_training`

#### 完整流程

```bash
# Step 1: 連接服務器
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/

# Step 2: 數據預處理（首次運行或數據變更時）
conda activate rppg_training
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data

# Step 3: 驗證數據
python validate_data.py --mode preprocessed

# Step 4: 訓練模型（6D 版本）
bash run_training.sh
# 或後台運行（防止斷線）
nohup python train.py --config config.yaml > logs/training.log 2>&1 &

# Step 5: 監控訓練
tail -f logs/training.log
```

#### 從服務器下載模型

```bash
# 下載訓練好的模型到本地
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/checkpoints/best_model.pth D:\MIAT\rppg\webapp\models\
```

### 本地量化與轉換流程

#### 1. 6D → 4D 模型轉換

```bash
# Step 1: 上傳轉換腳本到服務器
scp "D:\MIAT\rppg\server_training\convert_to_4d_for_stm32.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/

# Step 2: 在服務器上執行轉換
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/
conda activate rppg_training
python convert_to_4d_for_stm32.py

# Step 3: 下載 4D ONNX 模型到本地
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_fp32.onnx D:\MIAT\rppg\quantization\models\
```

**輸出**：
- `models/rppg_4d_fp32.onnx`（FP32 版本，用於量化）
- `models/rppg_4d_fp32.pth`（PyTorch 檢查點，可選）

#### 2. INT8 量化

```bash
# Step 1: 上傳量化腳本到服務器
scp "D:\MIAT\rppg\server_training\quantize_4d_model_v2.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/

# Step 2: 在服務器上執行量化（需要校準數據）
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/
conda activate rppg_training
python quantize_4d_model_v2.py

# Step 3: 驗證量化精度
python evaluate_quantized_model.py

# Step 4: 下載 INT8 ONNX 模型到本地
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
```

**輸出**：
- `models/rppg_4d_int8_qdq.onnx`（INT8 量化版本，用於 STM32）
- 驗證報告（MAE, RMSE, MAPE, R²）

### Web 應用部署流程

```bash
# Step 1: 確保模型存在
cd D:\MIAT\rppg\webapp
copy ..\server_training\checkpoints\best_model.pth models\best_model.pth

# Step 2: 安裝依賴
install.bat

# Step 3: 啟動服務器
start.bat

# Step 4: 訪問應用
# 瀏覽器打開 http://localhost:5000
```

### STM32N6 部署流程

詳細文檔：`DEPLOY_4D_TO_STM32.md`

#### 快速步驟

```bash
# Step 1: 準備 INT8 ONNX 模型
# 確保 D:\MIAT\rppg\quantization\models\rppg_4d_int8_qdq.onnx 存在

# Step 2: 在 STM32CubeMX 中導入模型
# - 打開 STM32CubeMX
# - 啟用 X-CUBE-AI
# - Import ONNX: rppg_4d_int8_qdq.onnx
# - 驗證輸入形狀：(1, 72, 36, 36) int8
# - 驗證輸出形狀：(1, 1) float32

# Step 3: 配置 X-CUBE-AI
# - Optimization: Time (O2) 或 Default (O1)  ← 避免 O3！
# - Runtime: Neural-ART (NPU)
# - Memory Pools: Auto
# - Analyze Model

# Step 4: 生成代碼
# - Generate Code
# - 檢查生成的 network_rppg.c

# Step 5: 編寫應用層代碼
# - 參考 DEPLOY_4D_TO_STM32.md 中的 preprocessing/postprocessing
# - ROI 提取（攝像頭 → 3 個 36×36 patches）
# - INT8 轉換（[0,255] → [-128,127]）
# - 推論調用
# - 輸出後處理（濾波、顯示）

# Step 6: 編譯與測試
# - Build Project
# - Flash to STM32N6
# - 驗證推論結果
```

#### 關鍵配置提醒

**X-CUBE-AI 配置**：
- ✅ Optimization: O1 或 O2（不要 O3）
- ✅ Runtime: Neural-ART
- ✅ Input Data Type: int8
- ✅ Output Data Type: float32
- ✅ Memory Pools: Auto

**常見問題排查**：
- 參考 `stm32n6_deployment/troubleshooting.md`
- 基於 Zero-DCE 失敗經驗整理

---

## 待辦事項

### 立即執行（服務器端）

- [ ] **監控預處理進度**
  ```bash
  cd /mnt/data_8T/ChenPinHao/server_training/
  ls -lh data/ubfc_processed.pt
  ```

- [ ] **驗證預處理數據**
  ```bash
  python validate_data.py --mode preprocessed
  # 檢查標籤分布：Min 40-50, Max 120-150, Mean 70-90, Std 8-15
  ```

- [ ] **開始訓練（如有新數據）**
  ```bash
  bash start_training_background.sh
  # 目標：MAE < 5 BPM, RMSE < 8 BPM
  ```

### 訓練完成後

- [ ] **評估模型性能**
  - 檢查訓練日誌（MAE, RMSE, MAPE）
  - 與之前版本比較
  - 確認收斂情況

- [ ] **6D → 4D 模型轉換**
  ```bash
  # 服務器端執行
  python convert_to_4d_for_stm32.py
  # 驗證輸出差異 < 1e-5
  ```

- [ ] **INT8 量化**
  ```bash
  # 服務器端執行
  python quantize_4d_model_v2.py
  python evaluate_quantized_model.py
  # 目標：MAE 增加 < 1.5 BPM（Quality: EXCELLENT/GOOD）
  ```

- [ ] **下載模型到本地**
  ```bash
  scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
  ```

### STM32N6 部署

- [ ] **在 STM32CubeMX 中導入模型**
  - 使用 `rppg_4d_int8_qdq.onnx`
  - 驗證輸入形狀：`(1, 72, 36, 36)` int8
  - Analyze 成功（無 ERROR）

- [ ] **生成代碼並編譯**
  - Optimization: O1 或 O2（避免 O3）
  - Generate Code
  - 編譯項目（無錯誤）

- [ ] **實現應用層邏輯**
  - ROI 提取代碼（攝像頭捕獲 → 3 個 ROI）
  - INT8 預處理（歸一化 + 量化）
  - 推論調用（`LL_ATON_RT_RunEpochBlock`）
  - 後處理（濾波、顯示）

- [ ] **驗證推論結果**
  - 使用已知測試影片
  - 對比 Python 推論結果
  - 確認準確度（MAE < 10 BPM）

### 可選優化

- [ ] **ROI 參數調優**
  - 實驗不同 ROI 比例和位置
  - 可視化不同光照條件下的效果

- [ ] **數據增強**
  - ROI 位置隨機抖動
  - 光照變化模擬
  - 運動模糊增強

- [ ] **融合策略優化**
  - 嘗試 attention-based fusion（學習 ROI 權重）
  - 實驗不同融合方式（加權平均、LSTM）

- [ ] **Web 應用增強**
  - 增強 ROI 檢測（MediaPipe Face Mesh）
  - 信號質量指示器（SNR 計算）
  - 歷史記錄導出（CSV/JSON）

---

## 參考資源

### 項目文檔

- **`DEVELOPMENT_LOG.md`** - 完整開發歷史（2025-01-14 至今）
- **`DEPLOY_4D_TO_STM32.md`** - 4D 模型部署指南
- **`stm32n6_deployment/`** - STM32 部署完整文檔
  - `deployment_guide.md` - 完整流程
  - `cubemx_config.md` - CubeMX 配置
  - `troubleshooting.md` - 故障排除
  - `preprocessing_code.c` / `postprocessing_code.c` - 代碼範例

### 論文與數據集

- **ME-rPPG**: https://arxiv.org/abs/2504.01774
- **UBFC-rPPG 數據集**: https://sites.google.com/view/ybenezeth/ubfcrppg
- **PURE 數據集**: https://www.tu-ilmenau.de/neurob/data-sets-code/pulse-rate-detection-dataset-pure

### STM32 技術資源

- **X-CUBE-AI 官方文檔**: https://www.st.com/en/embedded-software/x-cube-ai.html
- **STM32N6 產品頁**: https://www.st.com/stm32n6
- **Neural-ART Runtime**: https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI

### 相關項目經驗

- **Zero-DCE 部署失敗經驗**（`D:\MIAT\CLAUDE.md`）
  - 關鍵教訓：避免 O3 優化，信任工具自動配置
  - 不要手動修改生成的 `network_*.c` 代碼

---

**文檔版本**: 3.0 (Refactored)
**創建日期**: 2025-01-14
**最後更新**: 2025-01-26
**維護者**: Claude Code AI

**變更記錄**:
- v3.0 (2025-01-26): 重構為精簡版，移除歷史記錄到 DEVELOPMENT_LOG.md，增加 6D→4D 轉換方法
- v2.0 (2025-01-20): 增加 Web 應用文檔，健壯的 PPG → HR 標籤計算
- v1.0 (2025-01-14): 初始版本
