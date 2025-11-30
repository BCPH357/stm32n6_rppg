# rPPG 網頁應用 - 快速開始指南

## 🚀 5 分鐘快速啟動

### Step 1: 獲取模型文件

**方法 A**: 從服務器下載（推薦）

```bash
# 如果你已經訓練完成
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/checkpoints/best_model.pth models/
```

**方法 B**: 創建測試模型（僅用於測試介面）

```bash
python -c "import torch; from model import UltraLightRPPG; m = UltraLightRPPG(); torch.save({'model_state_dict': m.state_dict(), 'epoch': 0, 'mae': 999}, 'models/best_model.pth'); print('✓ Test model created')"
```

### Step 2: 安裝依賴

```bash
pip install -r requirements.txt
```

如果 PyTorch 安裝失敗：

```bash
# CPU 版本（快速安裝）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: 啟動服務器

**Windows**:
```batch
start.bat
```

**手動**:
```bash
python app.py
```

### Step 4: 打開瀏覽器

訪問: **http://localhost:5000**

### Step 5: 開始檢測

1. 點擊 **「開始檢測」** 按鈕
2. 授權攝像頭訪問
3. 將臉部對準攝像頭（正面、光線充足）
4. 等待 BVP 累積（256 幀，約 25 秒）
5. 查看即時心率！

---

## ⚙️ 詳細配置（可選）

### 調整推論幀率

編輯 `static/js/app.js`，第 136 行：

```javascript
// 當前：每 100ms = 10 fps
captureInterval = setInterval(() => {
    if (!isRunning) return;
    captureAndSend();
}, 100);  // ← 改為 50 = 20 fps，或 200 = 5 fps
```

### 調整 ROI 位置

編輯 `inference.py`，第 56-60 行：

```python
self.roi_ratios = {
    'forehead': {'x': (0.20, 0.80), 'y': (0.05, 0.25)},
    'left_cheek': {'x': (0.05, 0.30), 'y': (0.35, 0.65)},
    'right_cheek': {'x': (0.70, 0.95), 'y': (0.35, 0.65)}
}
```

### 調整心率範圍

編輯 `inference.py`，第 241 行：

```python
def calculate_hr(self, bvp_buffer, fps=10, hr_min=40, hr_max=160):
    # hr_min: 最小心率（預設 40 BPM）
    # hr_max: 最大心率（預設 160 BPM）
```

---

## 🐛 常見問題

### Q1: 瀏覽器無法訪問攝像頭？

**A**: 確保：
- Chrome/Edge 瀏覽器
- 已授權攝像頭權限
- 沒有其他程式佔用攝像頭

### Q2: 檢測不到臉部（黃色框不出現）？

**A**:
- 增加環境光線
- 調整臉部角度（正對攝像頭）
- 距離攝像頭 50-100 cm

### Q3: 心率值一直是「--」？

**A**:
- 等待 BVP 累積（至少 256 幀，約 25 秒）
- 查看右下角「BVP 緩衝: X/256」

### Q4: 心率數值不準確？

**A**:
- **如果使用測試模型（方法 B）**：這是正常的！測試模型沒有訓練，會產生隨機值
- **如果使用訓練模型**：
  - 保持臉部靜止
  - 改善光線條件
  - 點擊「重置」按鈕重新開始

### Q5: ModuleNotFoundError: No module named 'flask'

**A**:
```bash
pip install -r requirements.txt
```

### Q6: RuntimeError: Model file not found

**A**: 請先獲取模型文件（見 Step 1）

---

## 📊 預期效果

**正常運行時**：

```
FPS: 10-15
影格計數: 持續增加
BVP 緩衝: 逐漸累積到 256+
狀態: 檢測中... 或 心率檢測中...
當前心率: 60-100 BPM（正常人）
```

**介面顯示**：
- 左側：攝像頭影像 + 3 個彩色 ROI 框（綠、藍、紅）
- 右上：心率數值（大字體）
- 右中：BVP 波形圖（即時更新）
- 右下：心率趨勢圖（最近 30 秒）

---

## 📝 下一步

- 閱讀完整文檔：`README.md`
- 查看技術細節：`../CLAUDE.md`
- 調整參數以獲得更好效果
- 使用真實訓練模型（MAE 4.65 BPM）

---

**享受你的 rPPG 即時心率檢測系統！🫀**
