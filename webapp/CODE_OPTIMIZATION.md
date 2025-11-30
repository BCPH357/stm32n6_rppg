# 代碼優化記錄 - Model Import 重構

**日期**: 2025-01-20
**優化目標**: 避免代碼重複，讓 webapp 直接使用 server_training 的 model.py

---

## 問題背景

之前的架構中，`webapp/model.py` 和 `server_training/model.py` 是兩份幾乎完全相同的代碼（117 行 vs 176 行），存在以下問題：

1. **代碼重複** - 同一個模型定義維護兩份
2. **同步困難** - 修改模型需要同時更新兩個文件
3. **容易出錯** - 可能導致訓練和推論使用不同的模型定義
4. **維護成本高** - 任何架構變更都需要雙倍工作

---

## 解決方案

### 重構 webapp/model.py

將 `webapp/model.py` 改為 **wrapper 模塊**，直接從 `server_training/model.py` 導入 `UltraLightRPPG`。

**核心實現** (使用 importlib 避免循環導入):

```python
import sys
import os
import importlib.util

# 將 server_training 目錄加入 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
server_training_dir = os.path.join(parent_dir, 'server_training')

if server_training_dir not in sys.path:
    sys.path.insert(0, server_training_dir)

# 使用 importlib 動態載入模型模塊（避免命名衝突）
spec = importlib.util.spec_from_file_location(
    "server_training_model",
    os.path.join(server_training_dir, "model.py")
)
server_training_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_training_model)

# 導出 UltraLightRPPG
UltraLightRPPG = server_training_model.UltraLightRPPG
```

---

## 技術細節

### 為何使用 importlib？

**問題**: 直接使用 `from model import UltraLightRPPG` 會導致循環導入：

```
inference.py
    → from model import UltraLightRPPG  (webapp/model.py)
        → from model import UltraLightRPPG  (嘗試導入自己！)
            → ImportError: circular import
```

**解決**: `importlib.util.spec_from_file_location` 允許我們：
1. 明確指定要載入的文件路徑
2. 使用自定義模塊名（`server_training_model`）避免命名衝突
3. 動態執行模塊載入

### API 兼容性

重構後，其他模塊的 import 語法**完全不需要修改**：

```python
# inference.py (無需修改)
from model import UltraLightRPPG

# app.py (無需修改)
from inference import HeartRateDetector
```

---

## 優化成果

| 項目 | 優化前 | 優化後 |
|------|--------|--------|
| **webapp/model.py** | 117 行（完整模型定義） | 46 行（wrapper） |
| **代碼重複** | 有（兩份 model.py） | 無（只有 server_training） |
| **維護成本** | 高（雙倍工作） | 低（單一源頭） |
| **同步風險** | 有（可能不一致） | 無（保證一致） |
| **API 兼容性** | N/A | ✅ 完全兼容 |

**代碼減少**: ~71 行 (-60%)

---

## 測試驗證

### 測試 1: 模型導入

```bash
cd D:\MIAT\rppg\webapp
python model.py
```

**結果**:
```
============================================================
Testing Model Import from server_training
============================================================
Server training dir: D:\MIAT\rppg\server_training
Module imported from: server_training_model
Model class: UltraLightRPPG
Total parameters: 20,193

[OK] Model import successful!
```

✅ **通過**

### 測試 2: Inference 模塊

```bash
cd D:\MIAT\rppg\webapp
python -c "from inference import HeartRateDetector; print('Import successful')"
```

**結果**:
```
Import successful
```

✅ **通過**

### 測試 3: 完整 Inference 測試

```bash
cd D:\MIAT\rppg\webapp
python inference.py
```

**結果**:
```
============================================================
Initializing Heart Rate Detector
============================================================
[OK] Model loaded: models/best_model.pth
   Epoch: 49
   MAE: 3.4121 BPM
Model loaded on: cpu
Haar Cascade loaded: ...
[OK] Detector initialized successfully
============================================================

Processing 10 dummy frames...
[Frame 1-10 processed successfully]

[OK] Test completed!
```

✅ **通過**

---

## 文件變更清單

| 文件 | 變更類型 | 說明 |
|------|---------|------|
| `webapp/model.py` | ✏️ 重寫 | 從 117 行 → 46 行（wrapper） |
| `webapp/model.py.backup` | ➕ 新增 | 備份原始版本 |
| `webapp/inference.py` | ✅ 無變更 | API 完全兼容 |
| `webapp/app.py` | ✅ 無變更 | API 完全兼容 |
| `server_training/model.py` | ✅ 無變更 | 保持不變（單一源頭） |

---

## 未來維護

### 模型架構修改

現在只需要修改 **一個文件**：

```bash
# 修改模型定義
vim D:\MIAT\rppg\server_training\model.py

# webapp 自動使用最新版本（無需任何修改）
```

### 添加新模型

如果未來需要支持多個模型變體：

```python
# server_training/model.py
class UltraLightRPPG:
    pass

class UltraLightRPPG_V2:  # 新模型
    pass

# webapp/model.py (自動支持)
UltraLightRPPG = server_training_model.UltraLightRPPG
UltraLightRPPG_V2 = server_training_model.UltraLightRPPG_V2  # 添加一行即可
```

---

## 經驗總結

### ✅ 優點

1. **單一源頭 (Single Source of Truth)** - 模型定義只有一份
2. **零成本同步** - 修改立即生效，無需手動同步
3. **向後兼容** - 現有代碼無需修改
4. **易於測試** - 可獨立測試 wrapper 功能
5. **代碼簡潔** - 減少 60% 代碼量

### ⚠️ 注意事項

1. **路徑依賴** - 依賴正確的目錄結構（webapp 和 server_training 在同一父目錄）
2. **循環導入** - 必須使用 importlib 而非直接 import
3. **模塊名** - wrapper 模塊名（`server_training_model`）不應與其他模塊衝突

### 📚 最佳實踐

- **保留備份** - 重大重構前備份原始文件
- **充分測試** - 驗證所有依賴模塊
- **文檔記錄** - 清楚記錄重構動機和方法
- **漸進式** - 一次重構一個模塊，逐步驗證

---

**重構完成**: 2025-01-20
**測試狀態**: ✅ All Passed
**建議**: 未來項目可參考此模式避免代碼重複
