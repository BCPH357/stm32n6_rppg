# 模型輸出約束說明

## 📋 修改內容

### 新增輸出激活函數

在 `model.py` 中添加了 **Sigmoid 輸出約束**，將心率預測限制在生理範圍內。

---

## 🎯 修改細節

### 1. 初始化（`__init__`）

```python
# 心率範圍約束：[30, 180] BPM
self.output_act = nn.Sigmoid()
```

### 2. 前向傳播（`forward`）

**修改前**：
```python
bvp = self.fc(temporal_feats)  # (B, 1)
return bvp  # 無約束，範圍 (-∞, +∞)
```

**修改後**：
```python
out = self.fc(temporal_feats)       # (B, 1) - 原始輸出
hr = self.output_act(out) * 150 + 30  # (B, 1) - 縮放到 [30, 180] BPM
return hr
```

---

## 📐 數學原理

### Sigmoid 函數

```
σ(x) = 1 / (1 + e^(-x))
```

- **輸入範圍**：(-∞, +∞)
- **輸出範圍**：(0, 1)

### 縮放到心率範圍

```
hr = σ(x) * 150 + 30
```

- **當 x → -∞**：σ(x) → 0，hr → 30 BPM
- **當 x = 0**：σ(x) = 0.5，hr → 105 BPM
- **當 x → +∞**：σ(x) → 1，hr → 180 BPM

**輸出範圍**：**[30, 180] BPM**

---

## ✅ 優點

### 1. **強制生理約束**
- 心率不可能 < 30 BPM（靜息最低）
- 心率不可能 > 180 BPM（最大運動心率）
- 符合人體生理學

### 2. **訓練穩定性**
- Sigmoid 平滑可微（無梯度消失）
- 梯度在整個範圍內都存在
- 比 ReLU 或 Clamp 更適合回歸任務

### 3. **減少異常預測**
- 不會出現負心率
- 不會出現極端值（如 5 BPM 或 250 BPM）
- 提高模型魯棒性

### 4. **後處理簡化**
- 不需要手動 `np.clip(hr, 30, 180)`
- 模型輸出直接可用

---

## 🆚 其他方案對比

| 方案 | 輸出範圍 | 梯度 | 訓練難度 | 推薦度 |
|------|---------|------|---------|-------|
| **Sigmoid [30, 180]** | 嚴格 [30, 180] | ✅ 平滑 | 簡單 | ⭐⭐⭐⭐⭐ |
| **Softplus + 30** | [30, +∞) | ✅ 平滑 | 簡單 | ⭐⭐⭐⭐ |
| **ReLU + 30** | [30, +∞) | ⚠️ 不平滑 | 中等 | ⭐⭐⭐ |
| **無約束** | (-∞, +∞) | ✅ 平滑 | 中等 | ⭐⭐⭐ |
| **Tanh 變換** | [30, 180] | ✅ 平滑 | 簡單 | ⭐⭐⭐⭐ |

---

## 📊 預期影響

### 訓練階段

**Loss 計算**（MSE）：
```python
loss = MSELoss(hr_pred, hr_true)

# 假設 hr_true = 70 BPM
# 修改前：hr_pred 可能是 -10 或 250（異常）
# 修改後：hr_pred 限制在 [30, 180]，更接近真實值
```

**預期效果**：
- ✅ Loss 值更穩定
- ✅ 訓練收斂更快
- ✅ 減少異常樣本的影響

### 驗證階段

**指標計算**（MAE, RMSE, MAPE）：
```python
# 修改前
hr_pred = [-5, 25, 70, 85, 200]  # 有異常值
hr_true = [65, 68, 72, 80, 90]
MAE = 44.0  # 被異常值拉高

# 修改後
hr_pred = [42, 51, 70, 85, 120]  # 限制在 [30, 180]
hr_true = [65, 68, 72, 80, 90]
MAE = 18.2  # 更合理
```

**預期效果**：
- ✅ MAE 更能反映實際性能
- ✅ MAPE 不會因極端值爆表
- ✅ 評估指標更可靠

---

## 🔬 梯度分析

### Sigmoid 梯度

```
∂σ/∂x = σ(x) * (1 - σ(x))
```

**梯度分佈**：
- 當 x = 0：梯度最大（0.25）
- 當 x → ±∞：梯度 → 0（但仍存在）

### 對訓練的影響

```python
# 假設預測值接近邊界
hr_pred = 32 BPM  # 接近下界 30
hr_true = 35 BPM

# 梯度仍然存在（雖然較小）
# 模型仍能學習調整
```

**結論**：雖然邊界處梯度較小，但仍能有效學習。

---

## ⚠️ 注意事項

### 1. 真實標籤超出範圍

如果 UBFC 數據集中有極端值：
- 標籤 > 180 BPM → 模型最多預測 180
- 標籤 < 30 BPM → 模型最少預測 30

**解決**：
- 檢查數據集標籤分佈
- 如果有極端值，考慮：
  - 移除異常樣本
  - 調整範圍（如 [25, 200]）

### 2. 初始化影響

模型初始化時：
- FC 層輸出接近 0
- Sigmoid(0) = 0.5
- 初始預測 ≈ 105 BPM

**影響**：訓練初期 MAE 較高（如果真實值遠離 105）

---

## 📈 實驗建議

### 驗證範圍設置

**檢查數據集標籤分佈**：
```python
import torch
data = torch.load('data/ubfc_processed.pt')
labels = data['labels']

print(f'Min:  {labels.min():.2f} BPM')
print(f'Max:  {labels.max():.2f} BPM')
print(f'Mean: {labels.mean():.2f} BPM')
print(f'Std:  {labels.std():.2f} BPM')

# 檢查是否有超出 [30, 180] 的樣本
outliers = ((labels < 30) | (labels > 180)).sum()
print(f'Outliers: {outliers} / {len(labels)}')
```

**如果有大量異常值**：
```python
# 調整模型範圍
hr = self.output_act(out) * 175 + 25  # [25, 200] BPM
```

### 對比實驗

**建議同時訓練兩個版本**：
1. **有約束版本**（當前修改）
2. **無約束版本**（原始模型）

**對比指標**：
- 訓練 Loss 收斂速度
- 驗證 MAE / RMSE
- 異常預測數量（< 30 或 > 180）

---

## 🚀 使用指南

### 重新預處理後訓練

```bash
# 1. 驗證數據標籤範圍
python -c "
import torch
data = torch.load('data/ubfc_processed.pt')
labels = data['labels']
print(f'Label range: [{labels.min():.2f}, {labels.max():.2f}] BPM')
print(f'Outliers (< 30 or > 180): {((labels < 30) | (labels > 180)).sum()}')
"

# 2. 開始訓練（使用有約束的模型）
python train.py --config config.yaml
```

### 監控訓練

**正常輸出範例**：
```
Epoch 1/50
Results:
  Train Loss: 95.23    ← 應該較低（因為預測範圍受限）
  Val Loss:   102.45
  MAE:        7.34     ← 初期可能較高（初始預測 105 BPM）
  RMSE:       10.12
  MAPE:       10.5%
```

**異常信號**：
- Loss > 500：可能標籤範圍不匹配
- MAE > 50：模型輸出可能卡在邊界

---

## 📚 參考資料

### 為什麼用 Sigmoid 而不是其他？

1. **vs ReLU**：
   - ReLU 不平滑（x=0 處不可微）
   - ReLU 無上界（需要額外限制）

2. **vs Softplus**：
   - Softplus 無上界（需要 Clamp）
   - Sigmoid 天然有上下界

3. **vs Tanh**：
   - Tanh 輸出 [-1, 1]，需要縮放
   - Sigmoid 輸出 [0, 1]，更直觀

### 生理學依據

- **靜息心率**：50-80 BPM
- **最低可能心率**：~30 BPM（訓練有素的運動員）
- **最大運動心率**：220 - 年齡（通常 < 180 BPM）

UBFC 數據集主要是靜息狀態，預期範圍：**50-90 BPM**

---

## 🎓 技術細節

### 為什麼是 150 和 30？

```python
hr = σ(x) * 150 + 30
```

- **150**：範圍寬度（180 - 30）
- **30**：範圍下界

### 等價變換

```python
# 方法 1（當前）
hr = sigmoid(x) * 150 + 30

# 方法 2（等價）
hr_min, hr_max = 30, 180
hr = sigmoid(x) * (hr_max - hr_min) + hr_min

# 方法 3（使用 tanh）
hr_mid = (30 + 180) / 2  # 105
hr_range = (180 - 30) / 2  # 75
hr = tanh(x) * hr_range + hr_mid
```

---

**創建日期**: 2025-01-19
**版本**: v1.0 - Sigmoid 輸出約束
**作者**: Claude Code AI（基於用戶建議改進）
