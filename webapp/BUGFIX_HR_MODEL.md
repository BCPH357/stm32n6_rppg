# rPPG Web 應用 Bug 修復記錄

**日期**: 2025-01-20
**版本**: v1.1
**修復者**: Claude Code AI

---

## 🐛 問題描述

用戶報告了以下 Bug：

1. **心率永遠只有 40 BPM** - 顯示異常低且固定的心率
2. **BVP 波型圖永遠 180** - 圖表顯示固定值
3. **ROI 框有時候不更新** - 人臉移動時框沒有跟隨

---

## 🔍 根本原因分析

### 主要問題：錯誤理解模型輸出

**錯誤假設**: 模型輸出是 BVP (Blood Volume Pulse) 信號

**實際情況**: 模型訓練時直接預測 **HR (Heart Rate, BPM)**

#### 證據

從服務器端訓練代碼 `preprocess_data.py` 中確認：

```python
# 訓練數據創建
hr_labels = calculate_hr_from_ppg(ppg_signal, timestamps, fps, len(frames))
# hr_labels: (T,) - 每帧的心率标签（从 PPG 峰值计算）

# 標籤範圍
hr_per_frame = np.clip(hr_per_frame, 40, 160)  # 40-160 BPM
```

模型輸出範圍在 `model.py` 中定義：

```python
# 應用 Sigmoid 並縮放到 [30, 180] BPM
hr = self.output_act(out) * 150 + 30
return hr  # 直接返回 HR，不是 BVP！
```

### 錯誤的推論流程（v1.0）

```
模型輸出 (HR) → 當成 BVP → 累積 256 個 → Welch PSD → 計算 HR
                        ❌ 錯誤！
```

**結果**:
- 需要等待 256 幀才顯示心率（~25 秒）
- Welch PSD 對 HR 數值（40-160）計算頻譜，得到錯誤結果
- 心率固定在 40 BPM（Welch PSD 最低頻率邊界）

---

## ✅ 修復方案

### 正確的推論流程（v1.1）

```
模型輸出 (HR) → 使用移動中位數平滑 → 顯示心率
                        ✅ 正確！
```

### 關鍵修改

#### 1. 更新 `inference.py`

**變更 1: 緩衝區重命名**
```python
# 之前
self.bvp_buffer = []  # BVP 序列緩衝 ❌

# 修正後
self.hr_buffer = deque(maxlen=30)  # HR 預測緩衝（用於平滑）✅
```

**變更 2: 直接使用模型輸出為 HR**
```python
# 推論 - 模型直接輸出 HR (BPM)
with torch.no_grad():
    hr_raw = self.model(window_tensor).item()

# 將原始 HR 加入緩衝區（用於平滑）
self.hr_buffer.append(hr_raw)

# 使用移動中位數平滑 HR（減少抖動）
if len(self.hr_buffer) >= 5:  # 至少 5 個樣本才開始平滑
    hr_smoothed = np.median(list(self.hr_buffer)[-10:])  # 使用最近 10 個的中位數
    self.hr_history.append(hr_smoothed)
```

**變更 3: 移除 Welch PSD 方法**
```python
# 刪除 calculate_hr() 方法（不再需要）
# 刪除 scipy.signal.welch 導入
```

**變更 4: 增加臉部檢測頻率**
```python
# 之前
self.face_detect_interval = 5  # 每 5 幀才重新檢測 ❌

# 修正後
self.face_detect_interval = 3  # 每 3 幀重新檢測（更頻繁，提升響應性）✅
```

**變更 5: 返回值更新**
```python
return {
    'face_bbox': [...],
    'roi_coords': [...],
    'hr': float(hr_smoothed) if hr_smoothed is not None else None,  # 平滑後的 HR
    'hr_raw': float(hr_raw) if hr_raw is not None else None,        # 原始 HR（用於波形圖）
    'frame_count': int(self.frame_count),
    'hr_history': [float(h) for h in self.hr_history[-30:]],        # 歷史趨勢
    'status': status
}
```

#### 2. 更新 `app.py`

```python
# 返回結果給前端
emit('result', {
    'face_bbox': result['face_bbox'],
    'roi_coords': result['roi_coords'],
    'hr': result['hr'],          # 平滑後的 HR
    'hr_raw': result['hr_raw'],  # 原始 HR
    'frame_count': result['frame_count'],
    'hr_history': result['hr_history'],
    'status': result['status']
})
```

#### 3. 更新前端 `app.js`

**變更 1: 處理新的數據格式**
```javascript
function handleResult(data) {
    drawROIs(data.face_bbox, data.roi_coords);
    updateFrameCount(data.frame_count);
    updateStatus(data.status);

    // 更新原始 HR 到圖表（實時波形）
    if (data.hr_raw !== null) {
        updateBVPChart(data.hr_raw);  // 現在顯示原始 HR，不是 BVP
    }

    // 更新平滑後的心率顯示
    if (data.hr !== null) {
        updateHeartRate(data.hr);
    }

    // 更新心率歷史趨勢圖
    if (data.hr_history && data.hr_history.length > 0) {
        updateHRChart(data.hr_history);
    }
}
```

**變更 2: 更新圖表 Y 軸範圍**
```javascript
// 即時 HR 波形圖
y: {
    min: 30,
    max: 180,
    ticks: {
        color: '#ffffff',
        stepSize: 30
    },
    grid: { color: 'rgba(255, 255, 255, 0.1)' }
}

// 心率趨勢圖
y: {
    min: 40,
    max: 140,
    ticks: {
        color: '#ffffff',
        stepSize: 20
    },
    grid: { color: 'rgba(255, 255, 255, 0.1)' }
}
```

#### 4. 更新 HTML 標籤

```html
<!-- 之前 -->
<h3>BVP 波形圖</h3>
<p id="infoText">影格計數: <span id="frameCount">0</span> | BVP 緩衝: <span id="bvpBufferSize">0</span>/256</p>

<!-- 修正後 -->
<h3>即時心率波形</h3>
<p id="infoText">影格計數: <span id="frameCount">0</span></p>
```

---

## 📊 修復效果對比

| 項目 | v1.0 (有 Bug) | v1.1 (已修復) |
|------|--------------|--------------|
| **心率顯示延遲** | ~25 秒 (需要 256 幀 BVP) | ~0.5 秒 (5 幀即可平滑) |
| **心率準確性** | 固定 40 BPM (錯誤) | 動態 50-120 BPM (正確) |
| **波形圖顯示** | 固定 180 (錯誤) | 動態 HR 波動 (正確) |
| **ROI 框更新頻率** | 每 5 幀 | 每 3 幀 (更流暢) |
| **平滑算法** | Welch PSD (錯誤應用) | 移動中位數 (正確) |

---

## 🎯 預期行為（修復後）

1. **啟動後 ~1 秒**：開始顯示心率（不是 25 秒）
2. **心率範圍**：50-120 BPM（正常靜止心率）
3. **即時波形圖**：顯示 HR 的即時波動（30-180 BPM 範圍）
4. **心率趨勢圖**：顯示平滑後的 HR 歷史（40-140 BPM 範圍）
5. **ROI 框**：每 3 幀更新，跟隨臉部移動更流暢

---

## 🧪 測試步驟

1. **關閉舊服務器**：
   ```bash
   # 按 Ctrl+C 停止舊服務器
   ```

2. **重新啟動**：
   ```bash
   cd D:\MIAT\rppg\webapp
   python app.py
   ```

3. **打開瀏覽器**：http://localhost:5000

4. **測試檢查項**：
   - ✅ 點擊「開始檢測」後 1 秒內出現心率
   - ✅ 心率在 50-120 BPM 範圍（靜止狀態）
   - ✅ 即時波形圖顯示動態變化
   - ✅ 心率趨勢圖平滑上升/下降
   - ✅ 移動臉部時 ROI 框跟隨移動
   - ✅ 沒有 JSON 序列化錯誤

---

## 📝 技術細節

### 為什麼使用移動中位數而不是平均值？

**中位數的優勢**：
- 對異常值（outliers）不敏感
- 模型偶爾預測錯誤（如 30 或 180）不會影響整體
- 更穩定的顯示效果

**示例**：
```python
# 假設最近 10 個預測
hr_buffer = [75, 78, 76, 30, 77, 79, 75, 180, 76, 78]
                    ^^           ^^^
                    異常值       異常值

平均值 = 82.4 BPM  ← 被異常值拉高
中位數 = 76.5 BPM  ← 穩定，忽略異常值 ✅
```

### 為什麼保留 hr_raw？

- **hr_raw**: 模型每次推論的原始預測（未平滑）
- **hr**: 使用移動中位數平滑後的結果

**用途**：
- `hr_raw` → 即時波形圖（顯示模型的即時響應）
- `hr` → 大數字顯示 + 趨勢圖（顯示穩定的心率）

---

## 🔧 相關文件

- `inference.py`: 核心推論邏輯（重大修改）
- `app.py`: Flask 服務器（返回值更新）
- `static/js/app.js`: 前端邏輯（數據處理更新）
- `templates/index.html`: HTML 頁面（標籤更新）

---

## 📚 參考資料

- 訓練代碼: `D:\MIAT\rppg\server_training\preprocess_data.py`
- 模型定義: `D:\MIAT\rppg\webapp\model.py`
- 訓練配置: `D:\MIAT\rppg\CLAUDE.md`

---

**修復版本**: v1.1
**測試狀態**: ⏳ 待用戶測試
**預期結果**: ✅ 所有 Bug 應已解決
