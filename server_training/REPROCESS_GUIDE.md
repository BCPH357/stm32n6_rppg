# 重新預處理數據 - 完整指南

## 📋 變更摘要

### 核心問題
之前的訓練失敗是因為：
- **標籤是 BVP 值**（-2 到 127 範圍），不是心率 BPM
- **模型在預測 BVP**，但評估時當作心率 → 指標無意義
- **MAPE 爆表**（4247%）是因為 BVP 值接近 0 時除法錯誤

### 修復方案
修改 `preprocess_data.py`，將標籤從 **BVP 值** 改為 **心率 BPM**：

1. **使用 Welch 功率譜密度估計**計算每個樣本的心率
2. **最小 BVP 窗口**：256 幀（~8.5 秒 @ 30fps）
3. **心率範圍限制**：30-180 BPM（0.5-3.0 Hz）
4. **標籤統計**：訓練時會顯示心率範圍

### 預期結果
重新預處理後：
- 標籤範圍：**30-180 BPM**（不是 -2 到 127）
- 標籤平均值：**~70 BPM**（不是 43）
- MAE 目標：**< 5 BPM**（現在有實際意義）
- MAPE：**< 10%**（合理範圍）

---

## 🚀 執行步驟

### Step 1: 備份舊數據（可選）

```bash
ssh miat@140.115.53.67
cd /home/miat/ChenPinHao/server_training

# 備份舊的預處理數據（如果想保留）
mv data/ubfc_processed.pt data/ubfc_processed_old_bvp.pt

# 或直接刪除（節省磁碟空間 27 GB）
rm -f data/ubfc_processed.pt
```

### Step 2: 上傳修改後的文件

**在本地 Windows 執行**：

```bash
# 上傳修改後的預處理腳本
scp "D:\MIAT\rppg\server_training\preprocess_data.py" miat@140.115.53.67:/home/miat/ChenPinHao/server_training/

# 上傳修改後的訓練腳本
scp "D:\MIAT\rppg\server_training\train.py" miat@140.115.53.67:/home/miat/ChenPinHao/server_training/
```

### Step 3: 在服務器上重新預處理

```bash
ssh miat@140.115.53.67
cd /home/miat/ChenPinHao/server_training

# 啟動環境
conda activate rppg_training

# 檢查依賴（確保有 scipy）
python -c "from scipy.signal import welch; print('scipy OK')"

# 開始預處理（2-3 小時）
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
```

### Step 4: 驗證新數據

```bash
# 檢查標籤範圍（應該是心率 BPM）
python -c "
import torch
data = torch.load('data/ubfc_processed.pt')
labels = data['labels']
print(f'Label type: {data.get(\"label_type\", \"unknown\")}')
print(f'Label statistics:')
print(f'  Min:  {labels.min():.2f} BPM')
print(f'  Max:  {labels.max():.2f} BPM')
print(f'  Mean: {labels.mean():.2f} BPM')
print(f'  Std:  {labels.std():.2f} BPM')
print(f'Sample values (first 20):')
print(labels[:20])
"
```

**預期輸出**：
```
Label type: heart_rate_bpm
Label statistics:
  Min:  35.21 BPM
  Max:  165.43 BPM
  Mean: 72.84 BPM
  Std:  18.32 BPM
Sample values (first 20):
tensor([68.5, 71.2, 69.8, ..., 75.3, 72.1])
```

如果看到類似上面的結果（30-180 BPM 範圍），說明成功！

### Step 5: 開始訓練

```bash
# 方法 1: 使用現有腳本
bash start_training_background.sh

# 方法 2: 手動啟動
LOG_FILE="logs/training_hr_$(date +%Y%m%d_%H%M%S).log"
nohup python train.py --config config.yaml > "$LOG_FILE" 2>&1 &
echo $! > training.pid

# 監控訓練
tail -f logs/training_hr_*.log
```

### Step 6: 驗證訓練指標

**正常的指標應該是**：

```
Epoch 1/50
Results:
  Train Loss: 120.5432    <- MSE loss，合理範圍 50-200
  Val Loss:   145.2156
  MAE:        8.7234      <- 應該逐漸降到 < 5 BPM
  RMSE:       12.0453     <- 應該逐漸降到 < 8 BPM
  MAPE:       12.34%      <- 應該在 5-20% 範圍
  Time:       22.3s
```

---

## 📊 預期變化對比

| 指標 | 舊版（BVP） | 新版（HR） |
|------|------------|-----------|
| **標籤範圍** | -2.79 到 127.0 | 30-180 BPM |
| **標籤平均** | 42.99 | ~70-75 BPM |
| **訓練 Loss** | 874-1004 | 50-200 |
| **MAE** | 22.9（無意義） | < 5 BPM（目標） |
| **RMSE** | 30.4（無意義） | < 8 BPM（目標） |
| **MAPE** | 4247%（錯誤） | 5-15%（合理） |

---

## ⚠️ 注意事項

### 1. 樣本數量可能減少

由於心率計算需要至少 256 幀的 BVP 窗口：
- **舊版樣本數**：79,666
- **新版預估**：60,000-70,000（減少 10-20%）
- **原因**：太短的視頻片段會被跳過

這是正常的！質量比數量重要。

### 2. 預處理時間可能稍長

- **舊版**：2-3 小時
- **新版**：2.5-3.5 小時（多了 Welch 計算）
- **增加**：~15-20%

### 3. 文件大小變化

- **樣本數減少** → 文件變小
- **預估**：從 27.69 GB → 22-25 GB

### 4. 需要 scipy 依賴

確認已安裝：
```bash
conda activate rppg_training
conda list scipy

# 如果沒有，安裝：
pip install scipy
```

---

## 🔍 故障排除

### 問題 1: `scipy` 未安裝

**錯誤**：
```
ModuleNotFoundError: No module named 'scipy'
```

**解決**：
```bash
conda activate rppg_training
pip install scipy
```

### 問題 2: 預處理中途崩潰

**錯誤**：可能是 OOM（記憶體不足）

**解決**：
```bash
# 檢查記憶體使用
free -h

# 如果記憶體不足，關閉其他進程
# 或增加 swap（臨時）
```

### 問題 3: 警告 "HR calculation failed"

**警告**：
```
[WARN] HR calculation failed at idx 123: ...
```

**說明**：這是正常的！某些 BVP 片段可能太短或噪聲太大，會自動跳過。

**預期**：每個 subject 可能有 5-10 個這樣的警告，只要最終生成了樣本就沒問題。

### 問題 4: 標籤範圍不正常

**檢查後發現**：
```
Min: -5.0 BPM  <- 不合理！
Max: 250.0 BPM <- 不合理！
```

**可能原因**：
1. BVP 信號質量太差
2. Welch 參數不合適

**解決**：聯繫我進行調試。

---

## 📈 成功標準

重新預處理**成功**的標誌：

1. ✅ 預處理完成無崩潰
2. ✅ 生成 60,000+ 樣本
3. ✅ 標籤範圍：30-180 BPM
4. ✅ 標籤平均：65-80 BPM
5. ✅ 文件大小：20-26 GB
6. ✅ 訓練 Epoch 1 的 MAE < 15 BPM
7. ✅ 訓練 Epoch 5 的 MAE < 10 BPM
8. ✅ 最終 MAE < 5 BPM

---

## 🎓 技術細節

### Welch 方法參數

```python
freqs, psd = welch(
    bvp_window,
    fs=fps,                              # 採樣率（30 fps）
    nperseg=min(len(bvp_window)-1, 256), # 每段長度
    nfft=int(1e5 / fps)                  # FFT 點數
)
```

### 心率範圍限制

```python
# 生理合理範圍
hr_min = 30 BPM   →  0.5 Hz
hr_max = 180 BPM  →  3.0 Hz

# 只在這個範圍內找峰值
valid_idx = (freqs >= 0.5) & (freqs <= 3.0)
peak_freq = freqs[valid_idx][np.argmax(psd[valid_idx])]
hr_bpm = peak_freq * 60
```

### 為什麼需要 256 幀？

- **頻率分辨率** = fs / N = 30 / 256 ≈ 0.117 Hz
- **對應心率分辨率** ≈ 7 BPM
- 256 幀（~8.5 秒）是可靠心率估計的最小窗口

---

## 📞 需要幫助？

如果遇到任何問題，提供以下信息：

1. **錯誤訊息**（完整的 log）
2. **標籤統計**（`python -c "..."` 的輸出）
3. **樣本數量**（預處理後生成了多少樣本）
4. **訓練 log**（前 5 個 epoch 的指標）

---

**創建日期**: 2025-01-19
**版本**: v2.0 - 心率預測版本
**作者**: Claude Code AI
