# 獲取最新訓練的模型

## 問題診斷

當前 `webapp/models/best_model.pth` 模型**有問題**：
- FC 層輸出總是 ~225（遠超正常範圍）
- Sigmoid(225) ≈ 1.0
- 最終 HR 總是 180 BPM

**原因**：模型文件可能是舊的或損壞的。

---

## 解決方案：從服務器複製最新模型

### 步驟 1：檢查服務器上的最新模型

```bash
ssh miat@140.115.53.67

cd /mnt/data_8T/ChenPinHao/server_training/checkpoints/

# 查看最新的 checkpoint
ls -lht | head -10

# 應該會看到類似：
#   best_model.pth (最佳模型)
#   model_epoch_46.pth (最新 epoch)
```

### 步驟 2：複製最新模型到本地

**在 Windows 本地執行**：

```bash
# 複製 best_model.pth
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/checkpoints/best_model.pth "D:\MIAT\rppg\webapp\models\best_model.pth"

# 輸入密碼後等待傳輸完成
```

**或者使用 WinSCP / FileZilla 等工具**：
1. 連接到 `140.115.53.67`
2. 進入 `/mnt/data_8T/ChenPinHao/server_training/checkpoints/`
3. 下載 `best_model.pth`
4. 保存到 `D:\MIAT\rppg\webapp\models\best_model.pth`

### 步驟 3：驗證新模型

```bash
cd D:\MIAT\rppg\webapp
python test_model.py
```

**預期結果**：
```
Zero input  → 30-50 BPM (不是 180)
One input   → 150-180 BPM (合理)
Random      → 60-120 BPM (合理)
Real-like   → 60-100 BPM (合理)

[OK] Model responds to different inputs correctly!
```

---

## 調試模型內部（可選）

如果問題仍然存在，執行：

```bash
python test_model_debug.py
```

檢查 "After FC layers (before Sigmoid)" 的輸出：
- **正常範圍**：-5 到 +5 之間
- **異常**：> 10 或 < -10

---

## 臨時解決方案（如果無法立即取得模型）

如果暫時無法從服務器複製模型，可以使用未訓練的模型進行測試：

```python
# 創建一個隨機輸出的模型（僅用於測試 UI）
import torch
from model import UltraLightRPPG

model = UltraLightRPPG(window_size=8, num_rois=3)

# 保存隨機初始化的權重
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': 0,
    'mae': 999.0
}, 'models/random_model.pth')

print("Random model created for testing")
```

然後修改 `inference.py` 使用 `random_model.pth`。

**注意**：這個模型輸出會是隨機的，僅用於測試 UI 功能。

---

## 檢查模型文件是否損壞

```python
import torch

try:
    checkpoint = torch.load('models/best_model.pth', map_location='cpu')
    print("Keys in checkpoint:", checkpoint.keys())
    print("Epoch:", checkpoint.get('epoch', 'N/A'))
    print("MAE:", checkpoint.get('mae', 'N/A'))
    print("Model state dict size:", len(checkpoint['model_state_dict']))
    print("[OK] Checkpoint loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load checkpoint: {e}")
```

---

## 下一步

1. **立即執行**：從服務器複製最新模型
2. **驗證**：執行 `test_model.py`
3. **重啟應用**：`python app.py`
4. **測試 UI**：http://localhost:5000

應該可以看到正常變化的心率值（50-120 BPM），而不是固定的 180。
