# rPPG 即時心率檢測網頁應用

基於 Multi-ROI rPPG 模型的即時心率檢測系統。

## 📋 功能特性

- ✅ 即時攝像頭捕獲（30 fps）
- ✅ 臉部檢測（Haar Cascade）
- ✅ 3 個 ROI 區域即時顯示（前額、左臉頰、右臉頰）
- ✅ BVP 波形圖即時更新
- ✅ 心率計算與顯示（MAE: 4.65 BPM）
- ✅ 心率歷史趨勢圖（30 秒）
- ✅ WebSocket 即時通訊

## 🏗️ 技術架構

**後端**:
- Python 3.8+
- Flask + Flask-SocketIO
- PyTorch (Multi-ROI rPPG 模型)
- OpenCV (臉部檢測、ROI 提取)
- SciPy (Welch PSD 心率計算)

**前端**:
- HTML5 + CSS3
- JavaScript (ES6+)
- Socket.IO Client
- Chart.js (圖表)
- WebRTC (攝像頭訪問)

## 📂 目錄結構

```
webapp/
├── app.py                 # Flask 主程式
├── inference.py           # 推論邏輯
├── model.py               # Multi-ROI 模型定義
├── requirements.txt       # Python 依賴
├── install.bat           # 安裝腳本
├── start.bat             # 啟動腳本
├── templates/            # HTML 模板
│   └── index.html
├── static/              # 靜態文件
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── models/              # 模型文件
    └── best_model.pth   # (需要複製)
```

## 🚀 安裝與使用

### Step 1: 複製模型文件

```batch
copy ..\server_training\checkpoints\best_model.pth models\best_model.pth
```

### Step 2: 安裝依賴

雙擊運行 `install.bat`，或手動執行：

```batch
pip install -r requirements.txt
```

**注意**: 如果 PyTorch 安裝失敗，可手動安裝：

```batch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: 啟動服務器

雙擊運行 `start.bat`，或手動執行：

```batch
python app.py
```

### Step 4: 打開瀏覽器

訪問：http://localhost:5000

### Step 5: 開始檢測

1. 點擊「開始檢測」按鈕
2. 授權攝像頭訪問
3. 確保臉部在畫面中央
4. 等待收集 256 幀（約 25 秒）
5. 查看即時心率和波形

## 🎨 介面說明

### 左側面板
- **攝像頭影像**: 即時視頻流，疊加 ROI 框
  - 綠色框：前額 (Forehead)
  - 藍色框：左臉頰 (Left Cheek)
  - 紅色框：右臉頰 (Right Cheek)
- **控制按鈕**: 開始、停止、重置
- **狀態訊息**: 當前處理狀態

### 右側面板
- **當前心率**: 大字體顯示 BPM
- **BVP 波形圖**: 血容積脈搏信號
- **心率趨勢圖**: 最近 30 秒的心率變化

## ⚡ 性能參數

- **推論速度**: ~10 fps（前端發送）
- **心率計算**: 需要至少 256 幀 (~25 秒)
- **準確度**: MAE 4.65 BPM (基於 UBFC 數據集)
- **輸入尺寸**: 3 × 36×36×3 (每個 ROI)
- **模型參數**: ~20K

## ⚠️ 注意事項

### 瀏覽器要求
- Chrome/Edge (推薦)
- Firefox (支援)
- Safari (可能需要 HTTPS)

### 使用環境
- 光線充足（避免逆光）
- 臉部正對攝像頭
- 保持相對靜止（頭部運動會影響精度）
- 建議距離：50-100 cm

### 已知限制
- 需要穩定的臉部檢測（Haar Cascade）
- 側臉或遮擋可能失敗
- 首次使用需等待 BVP 累積（256 幀）

## 🐛 故障排除

### 問題 1: 無法訪問攝像頭
**解決方案**:
- 確保瀏覽器已授權攝像頭訪問
- 檢查是否有其他程式佔用攝像頭
- 嘗試使用 HTTPS（非 localhost 時需要）

### 問題 2: 模型文件未找到
**解決方案**:
```batch
copy ..\server_training\checkpoints\best_model.pth models\best_model.pth
```

### 問題 3: PyTorch 安裝失敗
**解決方案**:
```batch
# CPU 版本
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 版本
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 問題 4: 檢測不到臉部
**解決方案**:
- 增加環境光線
- 調整臉部位置（居中）
- 檢查 Haar Cascade 文件是否正確載入

### 問題 5: 心率數值異常
**解決方案**:
- 確保已收集足夠的 BVP 數據（256+ 幀）
- 減少頭部運動
- 改善光線條件
- 點擊「重置」按鈕重新開始

## 📊 技術細節

### ROI 提取比例
```python
roi_ratios = {
    'forehead':    {'x': (0.20, 0.80), 'y': (0.05, 0.25)},
    'left_cheek':  {'x': (0.05, 0.30), 'y': (0.35, 0.65)},
    'right_cheek': {'x': (0.70, 0.95), 'y': (0.35, 0.65)}
}
```

### Welch PSD 參數
```python
fps = 10  # 推論幀率
hr_min = 40  # 最小心率 (BPM)
hr_max = 160  # 最大心率 (BPM)
nperseg = min(len(bvp_buffer) - 1, 256)
```

### 模型輸入
```python
window_size = 8  # 時間窗口
input_shape = (1, 8, 3, 36, 36, 3)  # (B, T, ROI, H, W, C)
```

## 📝 版本歷史

### v1.0 (2025-01-20)
- ✅ 初始版本發布
- ✅ Multi-ROI 架構
- ✅ 即時 WebSocket 通訊
- ✅ BVP 和心率趨勢圖

## 📧 聯繫與支援

如有問題或建議，請參考主項目的 CLAUDE.md 文檔。

---

**模型訓練資訊**:
- Dataset: UBFC-rPPG
- Epoch: 19
- MAE: 4.65 BPM
- RMSE: 6.63 BPM
- MAPE: 4.94%
