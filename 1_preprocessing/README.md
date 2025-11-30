# 📊 階段 1：數據前處理

本階段負責下載、處理和驗證 UBFC-rPPG 數據集，為模型訓練準備標準化的輸入數據。

## 🎯 目標

1. 下載 UBFC-rPPG 數據集（42 subjects）
2. 提取 Multi-ROI 特徵（前額、左右臉頰）
3. 計算健壯的 PPG → HR 標籤（Bandpass + Peak Detection）
4. 生成訓練/驗證/測試集

## 📁 檔案說明

| 檔案 | 說明 | 執行環境 |
|------|------|----------|
| `download_ubfc.sh` | 下載 UBFC-rPPG 數據集 | 服務器 |
| `preprocess_data.py` | 數據預處理主程式 | 服務器 |
| `validate_data.py` | 驗證預處理結果 | 服務器/本地 |
| `check_data_structure.py` | 檢查數據結構 | 服務器/本地 |

## 🚀 快速開始

### 在服務器上執行

```bash
# 1. 連接到服務器
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/rppg/1_preprocessing/

# 2. 激活環境
conda activate rppg_training

# 3. 下載數據集（首次運行）
bash download_ubfc.sh

# 4. 執行預處理
python preprocess_data.py \
    --dataset ubfc \
    --raw_data raw_data \
    --output data

# 5. 驗證結果
python validate_data.py --mode preprocessed
```

### 在本地執行

```bash
# 1. 安裝依賴
pip install -r ../requirements_rppg_training.txt

# 2. 從服務器下載已預處理的數據（可選）
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/rppg/1_preprocessing/data/ubfc_processed.pt ./data/

# 3. 驗證數據
python validate_data.py --mode preprocessed
```

## 📦 輸入/輸出

### 輸入
- **raw_data/UBFC-rPPG/** - 原始數據集
  - `subject1/` ~ `subject42/`
  - 每個 subject 包含：
    - `vid.avi` - 面部影片（30 fps）
    - `ground_truth.txt` - PPG 信號

### 輸出
- **data/ubfc_processed.pt** - 預處理後的 PyTorch 數據集
  - 格式：`(N, 8, 3, 36, 36, 3)`
  - N: 樣本數
  - 8: 時間窗口（8 幀）
  - 3: ROI 數量（前額、左右臉頰）
  - 36×36×3: RGB 影像

## 🔍 數據驗證

運行驗證腳本會檢查：

```bash
python validate_data.py --mode preprocessed
```

**預期輸出**：
```
[數據集統計]
  樣本數: ~15000
  輸入形狀: (8, 3, 36, 36, 3)
  標籤範圍: [40.0, 150.0] BPM
  平均 HR: ~75.0 BPM
  標準差: ~12.0 BPM
```

## ⚙️ 參數說明

### preprocess_data.py

```bash
python preprocess_data.py \
    --dataset ubfc \              # 數據集名稱
    --raw_data raw_data \         # 原始數據路徑
    --output data \               # 輸出路徑
    --window_size 8 \             # 時間窗口大小
    --roi_size 36 \               # ROI 影像大小
    --fps 30                      # 影片 FPS
```

### ROI 提取邏輯

| ROI | 位置（相對臉部框） | 顏色標記 |
|-----|-------------------|---------|
| **Forehead** | x: [0.20w, 0.80w]<br>y: [0.05h, 0.25h] | 紅色 |
| **Left Cheek** | x: [0.05w, 0.30w]<br>y: [0.35h, 0.65h] | 藍色 |
| **Right Cheek** | x: [0.70w, 0.95w]<br>y: [0.35h, 0.65h] | 橙色 |

## 📊 數據增強

目前實現的數據增強：
- ✅ ROI 位置隨機抖動（±5%）
- ✅ 光照變化模擬（亮度調整）
- ⏳ 運動模糊增強（未來版本）

## ❗ 常見問題

### Q1: 預處理速度很慢？
**A**: 正常情況下處理 42 subjects 需要 1-2 小時。可以：
- 使用服務器的 GPU 加速（如果有 CUDA）
- 減少數據增強次數
- 使用多進程處理（修改腳本）

### Q2: 標籤範圍異常？
**A**: 檢查 PPG 信號質量：
```bash
python validate_data.py --mode raw --visualize
```
應該看到標籤分布在 40-150 BPM，平均約 70-90 BPM。

### Q3: 找不到臉部？
**A**: 確認 Haar Cascade 文件存在：
```bash
ls /path/to/haarcascade_frontalface_default.xml
```

## 📝 下一步

完成數據預處理後，前往 `2_training/` 開始模型訓練。

```bash
cd ../2_training
```

---

**環境要求**: Python 3.8+, PyTorch 2.0+, OpenCV 4.8+
**參見**: `../requirements_rppg_training.txt`
