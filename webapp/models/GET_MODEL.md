# 如何獲取模型文件

網頁應用需要訓練好的模型文件 `best_model.pth`。

## 方法 1: 從本地服務器訓練目錄複製（推薦）

如果你已經在服務器上訓練完成模型：

```bash
# 在服務器上
cd /mnt/data_8T/ChenPinHao/server_training
scp checkpoints/best_model.pth [你的用戶名]@[你的IP]:D:/MIAT/rppg/webapp/models/
```

或者從 Windows 本地複製（如果已經下載到本地）：

```batch
copy checkpoints\best_model.pth D:\MIAT\rppg\webapp\models\best_model.pth
```

## 方法 2: 從服務器直接下載

使用 SCP 或其他文件傳輸工具：

```bash
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/checkpoints/best_model.pth models/
```

## 方法 3: 使用未訓練的模型（僅測試）

如果只是想測試網頁應用功能（不關心準確度），可以創建一個空的模型文件：

```python
import torch
from model import UltraLightRPPG

model = UltraLightRPPG(window_size=8, num_rois=3)
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'val_loss': 999,
    'mae': 999
}, 'models/best_model.pth')
```

**注意**: 未訓練的模型會產生隨機的心率值，僅用於測試介面！

## 確認模型文件

正確的目錄結構應該是：

```
webapp/
└── models/
    └── best_model.pth  ← 應該在這裡
```

檢查文件大小（應該約 70-100 KB）：

```batch
dir models\best_model.pth
```

## 如果還沒有訓練模型

請先完成模型訓練：

```bash
cd D:\MIAT\rppg\server_training
python train.py --config config.yaml
```

訓練完成後，模型會保存在 `checkpoints/best_model.pth`。
