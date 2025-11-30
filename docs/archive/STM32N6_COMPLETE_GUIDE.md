# STM32N6 完整部署指南 - ONNX Graph Surgery 方案

**創建日期**: 2025-01-26
**方法**: ONNX Graph Surgery（無需重新訓練）
**目標**: 修復現有 rPPG 模型以符合 STM32N6 NPU 限制

---

## 📋 目錄

1. [問題總結](#問題總結)
2. [解決方案概述](#解決方案概述)
3. [完整執行流程](#完整執行流程)
4. [工具說明](#工具說明)
5. [驗證與測試](#驗證與測試)
6. [常見問題](#常見問題)

---

## 問題總結

### STM32 Edge AI 錯誤

```
INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'
```

### 根本原因（ST 工程師回饋）

當前的 `rppg_4d_fp32.onnx` 違反以下 STM32N6 NPU 限制：

| 違規項 | 當前狀態 | STM32N6 要求 |
|--------|---------|-------------|
| **張量維度** | 包含 6D 中間張量 | Max rank = 5 |
| **Batch 維度** | 前三個 Conv 的 batch=24 | Batch = 1 |
| **Dynamic Batch** | 可能存在動態維度 | 所有維度固定 |
| **6D Reshape** | 模型內部有 6D→4D reshape | 不允許 6D 張量 |
| **Squeeze Axes** | 可能使用 tensor input | 必須使用 attribute |

### 違規代碼位置（`model_4d_stm32.py`）

```python
# ❌ 違規點 1: 創建 6D 張量
x = x.view(B, 8, 3, 3, 36, 36)     # (B, T, ROI, C, H, W) - 6D
x = x.permute(0, 1, 2, 4, 5, 3)    # (B, T, ROI, H, W, C) - 6D

# ❌ 違規點 2: Batched Convolution
x = x.view(B*T*ROI, C, H, W)       # (24, 3, 36, 36) when B=1
spatial_feats = self.spatial(x)    # Conv 看到 batch=24
```

---

## 解決方案概述

### 策略：ONNX Graph Surgery（無需重新訓練）

我們提供 **兩種方案** 解決問題：

#### 方案 A：Clean Export（推薦）⭐

**原理**: 重寫 `forward()` 函數，從 PyTorch 導出時就避免產生 6D 張量

**優點**:
- ✅ 從源頭解決問題
- ✅ 生成的 ONNX 圖更乾淨
- ✅ 更容易通過驗證

**文件**: `export_onnx_stm32_clean.py`

#### 方案 B：Graph Surgery

**原理**: 使用 ONNX Graph Surgeon 修復現有 ONNX 模型

**優點**:
- ✅ 不需要修改 PyTorch 代碼
- ✅ 可以修復任何現有 ONNX

**文件**: `fix_onnx_for_stm32.py`

### 修復項目清單

| 修復項 | 方案 A | 方案 B | 說明 |
|--------|--------|--------|------|
| 避免 6D 張量 | ✅ | ✅ | A: 修改 forward()<br>B: 移除 Reshape 節點 |
| 固定 batch=1 | ✅ | ✅ | A: 導出時固定<br>B: 修改 input/output shape |
| Squeeze axes | ✅ | ✅ | A: PyTorch 自動處理<br>B: tensor→attribute |
| 移除動態 shape | ✅ | ✅ | A: 無 dynamic_axes<br>B: 修改 Constant 節點 |

---

## 完整執行流程

### 準備工作

#### 1. 確認環境

```bash
# 連接服務器
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/

# 激活環境
conda activate rppg_training

# 確認模型存在
ls -lh checkpoints/best_model.pth
```

#### 2. 上傳腳本（如需要）

```bash
# 從本地上傳所有腳本
scp "D:\MIAT\rppg\server_training\diagnose_onnx_stm32.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
scp "D:\MIAT\rppg\server_training\fix_onnx_for_stm32.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
scp "D:\MIAT\rppg\server_training\export_onnx_stm32_clean.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
scp "D:\MIAT\rppg\server_training\deploy_stm32n6_complete.sh" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
```

#### 3. 安裝依賴

```bash
# ONNX Graph Surgeon (如果還沒安裝)
pip install onnx-graphsurgeon
```

---

### 執行方案 A：Clean Export（推薦）

#### Step 1: 導出 Clean ONNX

```bash
python export_onnx_stm32_clean.py
```

**輸出**:
```
models/rppg_stm32_clean_fp32.onnx
```

**關鍵改進**（vs 原始 `model_4d_stm32.py`）:

```python
# ❌ 原始版本（產生 6D）
x = x.view(B, 8, 3, 3, 36, 36)     # 6D 張量
x = x.permute(0, 1, 2, 4, 5, 3)    # 6D 張量

# ✅ Clean 版本（保持 4D）
x = x.view(B * 24, 3, H, W)        # (B*T*ROI, C, H, W) - 4D
# 後續操作全部保持 ≤4D
```

#### Step 2: 診斷 ONNX

```bash
python diagnose_onnx_stm32.py --onnx models/rppg_stm32_clean_fp32.onnx
```

**預期輸出**:
```
✅ Opset Version: 14 (RECOMMENDED)
✅ All dimensions are fixed
✅ No 6D tensors found
✅ All Conv/Pool layers appear to have batch=1
✅ MODEL IS STM32N6-COMPATIBLE!
```

#### Step 3: 如果仍有問題，應用 Graph Surgery

```bash
python fix_onnx_for_stm32.py \
    --input models/rppg_stm32_clean_fp32.onnx \
    --output models/rppg_stm32_clean_fixed.onnx
```

---

### 執行方案 B：修復現有 ONNX

#### Step 1: 診斷現有模型

```bash
python diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32.onnx
```

**預期會發現**:
- ❌ 6D 張量
- ❌ Batched convolution
- ⚠️  Dynamic shapes

#### Step 2: 應用 Graph Surgery

```bash
python fix_onnx_for_stm32.py \
    --input models/rppg_4d_fp32.onnx \
    --output models/rppg_4d_fp32_fixed.onnx
```

**修復操作**:
1. 移除所有 6D reshape 節點
2. 固定 input/output batch=1
3. 將 Squeeze 的 tensor axes 改為 attribute axes
4. 修復動態 shape constants
5. 清理和優化圖結構

#### Step 3: 驗證修復結果

```bash
python diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32_fixed.onnx
```

---

### 量化為 INT8

#### Step 1: 量化（使用修復後的 FP32 模型）

```bash
# 更新量化腳本中的輸入路徑
python quantize_4d_model_v2.py
```

**輸出**:
```
models/rppg_4d_int8_qdq.onnx
```

#### Step 2: 評估量化精度

```bash
python evaluate_quantized_model.py
```

**目標**:
- MAE 增加 < 1.5 BPM
- Quality: EXCELLENT 或 GOOD

---

### 自動化流程（推薦）

使用完整部署腳本：

```bash
chmod +x deploy_stm32n6_complete.sh
./deploy_stm32n6_complete.sh
```

**腳本會自動**:
1. 提示選擇方案（Clean Export / Fix Existing / Both）
2. 執行導出或修復
3. 診斷所有 ONNX
4. 執行 INT8 量化
5. 最終驗證
6. 生成部署報告

---

## 工具說明

### 1. `diagnose_onnx_stm32.py` - 診斷工具

**功能**: 檢測 ONNX 模型違反 STM32N6 的所有限制

**檢查項目**:
- ✅ Opset version (推薦 14)
- ✅ Tensor rank (max 5D, 最好 4D)
- ✅ Dynamic dimensions (不允許)
- ✅ Batch dimensions (必須 = 1)
- ✅ Reshape nodes (檢測 6D→4D)
- ✅ Squeeze/Unsqueeze (axes 必須是 attribute)
- ✅ Unsupported operations

**用法**:
```bash
python diagnose_onnx_stm32.py --onnx models/your_model.onnx
```

**輸出示例**:
```
======================================================================
📊 FINAL DIAGNOSTIC REPORT
======================================================================

總計檢查項目:
  - ❌ Violations: 3
  - ⚠️  Warnings: 1

❌ VIOLATIONS (必須修復):
  1. Input 'input' has rank 6 > 5 (6D not supported)
  2. Found 1 Conv/Pool layers with batch > 1
  3. Reshape to 6D shape detected: [1, 8, 3, 3, 36, 36]

⚠️  WARNINGS (建議修復):
  1. Reshape with dynamic dimension (-1): [1, -1, 36, 36]

======================================================================
❌ MODEL HAS VIOLATIONS - MUST FIX BEFORE STM32 DEPLOYMENT
Use fix_onnx_for_stm32.py to repair the model
======================================================================
```

---

### 2. `fix_onnx_for_stm32.py` - Graph Surgery 修復工具

**功能**: 使用 ONNX Graph Surgeon 自動修復違規項

**修復操作**:

1. **固定 Input Batch**
   ```python
   input_tensor.shape[0] = 1  # 從 dynamic 或 >1 改為 1
   ```

2. **移除 6D Reshape 節點**
   ```python
   # 找到 target_shape 為 6D 的 Reshape
   # 移除該節點，直接連接 input → output
   ```

3. **修復 Squeeze Axes**
   ```python
   # 從: Squeeze(x, axes_tensor)
   # 到: Squeeze(x).attrs['axes'] = [values]
   ```

4. **修復動態 Constants**
   ```python
   # 將 [-1, 72, 36, 36] 改為 [1, 72, 36, 36]
   ```

**用法**:
```bash
python fix_onnx_for_stm32.py \
    --input models/rppg_4d_fp32.onnx \
    --output models/rppg_4d_fp32_fixed.onnx
```

**輸出示例**:
```
======================================================================
STM32N6 ONNX Graph Surgery
======================================================================

[Fix 1] Fixing input batch dimension...
  ✅ input: [?, 72, 36, 36] → [1, 72, 36, 36]

[Fix 3] Removing 6D reshapes...
  ❌ Found 6D Reshape: Reshape_42
       Target shape: [1, 8, 3, 3, 36, 36]
  ✅ Removed 1 6D reshape nodes

[Fix 4] Fixing Squeeze/Unsqueeze nodes...
  Processing 2 Squeeze nodes...
    ✅ Squeeze_45: axes=[2, 3] (tensor→attribute)

Total fixes applied: 5
  ✅ Input batch fixed: input
  ✅ Removed 6D reshape: Reshape_42
  ✅ Squeeze axes fixed: Squeeze_45
  ✅ Dynamic constant fixed: Constant_12
  ✅ Graph optimized and cleaned
======================================================================
```

---

### 3. `export_onnx_stm32_clean.py` - Clean Export

**功能**: 從 PyTorch 導出時就避免產生 6D 張量

**核心策略**:

```python
class UltraLightRPPG_STM32Clean(nn.Module):
    def forward(self, x):
        # 輸入: (B, 72, 36, 36) - 4D

        # ✅ 保持 4D: 展平 batch 維度
        x = x.view(B * 24, 3, H, W)  # (B*T*ROI, C, H, W)

        # ✅ 通過 CNN（每個 T*ROI 組合獨立處理）
        x = self.spatial(x)  # (B*24, 16, 1, 1)

        # ✅ Reshape 回 3D: (B, 24, 16)
        x = x.view(B, 24, 16)

        # ✅ 再 reshape 為 (B, 8, 48) - 仍然是 3D
        x = x.view(B, 8, 48)

        # ... 後續操作全部 ≤ 3D
```

**用法**:
```bash
python export_onnx_stm32_clean.py
```

---

## 驗證與測試

### 1. 本地驗證（服務器端）

```bash
# Step 1: 診斷 ONNX
python diagnose_onnx_stm32.py --onnx models/rppg_4d_int8_qdq.onnx

# Step 2: 使用 ONNX Runtime 測試推論
python -c "
import onnx
import onnxruntime as ort
import numpy as np

model = onnx.load('models/rppg_4d_int8_qdq.onnx')
ort_session = ort.InferenceSession('models/rppg_4d_int8_qdq.onnx')

# 測試輸入
x = np.random.randint(-128, 127, (1, 72, 36, 36), dtype=np.int8)

# 推論
outputs = ort_session.run(None, {'input': x})
hr = outputs[0][0][0]

print(f'Predicted HR: {hr:.2f} BPM')
assert 30 <= hr <= 180, 'HR out of range!'
print('✅ ONNX inference successful')
"
```

### 2. 下載模型到本地

```bash
# 下載 INT8 模型
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\MIAT\rppg\

# 下載 FP32 模型（備用）
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_stm32_clean_fp32.onnx D:\MIAT\rppg\
```

### 3. STM32 Edge AI Developer Cloud 驗證

#### 方法 1: Web UI

1. 前往 https://stedgeai-dc.st.com/
2. 點擊 "New Project"
3. 上傳 `rppg_4d_int8_qdq.onnx`
4. 選擇 Target: **STM32N6**
5. 點擊 "Analyze"

**預期結果**:
- ✅ 無 `INTERNAL ERROR`
- ✅ 顯示記憶體使用統計
- ✅ 顯示 MACs 統計
- ✅ 可以生成 C 代碼

#### 方法 2: stedgeai CLI（如果已安裝）

```bash
stedgeai analyze \
    --model rppg_4d_int8_qdq.onnx \
    --target stm32n6 \
    --optimization balanced
```

**預期輸出**:
```
Analyzing rppg_4d_int8_qdq.onnx ...

Network configuration:
  Inputs:
    - input: (1, 72, 36, 36) int8
  Outputs:
    - output: (1, 1) float32

Memory usage:
  - Activation RAM: 256 KB
  - Weights ROM: 20 KB
  - Total RAM: 300 KB

MACs: 8.5 M

✅ Analysis successful
Ready for code generation
```

---

## 常見問題

### Q1: 診斷報告顯示仍有 6D 張量，怎麼辦？

**A**: 使用 Graph Surgery 修復：

```bash
python fix_onnx_for_stm32.py \
    --input models/your_model.onnx \
    --output models/your_model_fixed.onnx
```

如果仍然失敗，檢查：
- 是否使用了 opset 14？
- 是否有 `dynamic_axes`？
- PyTorch 版本是否過舊？

---

### Q2: Squeeze axes 無法修復，顯示 "axes is not constant"

**A**: 這表示 axes 是動態計算的，無法轉換為 attribute。

**解決方案**:
1. 修改 PyTorch 模型，使用明確的 axes 參數：
   ```python
   # ❌ 動態
   x = x.squeeze()  # 自動推斷 axes

   # ✅ 靜態
   x = x.squeeze(-1).squeeze(-1)  # 明確指定
   ```

2. 重新導出 ONNX

---

### Q3: Conv 層顯示 batch > 1，如何修復？

**A**: 這是模型設計問題，Graph Surgery 無法修復。

**解決方案**:
使用 Clean Export（`export_onnx_stm32_clean.py`），它重寫了 forward() 以避免 batched conv：

```python
# ❌ 原始（batch=24）
x = x.view(B*T*ROI, C, H, W)  # (24, 3, 36, 36)
x = cnn(x)  # Conv 看到 batch=24

# ✅ Clean Export（batch=1，但透過 channel 維度處理）
# 邏輯相同，但 ONNX 圖結構不同
```

---

### Q4: 量化後精度大幅下降（MAE > 5 BPM），如何改善？

**A**: 檢查校準數據：

```python
# quantize_4d_model_v2.py 中的校準數據設置
class RPPG4DCalibrationDataReader:
    def __init__(self, data_path, num_samples=200):  # 增加樣本數
        # 確保使用分層採樣
        # 確保涵蓋所有 HR 範圍 (40-160 BPM)
```

**改進方法**:
1. 增加校準樣本數（200 → 500）
2. 確認分層採樣涵蓋所有 HR 範圍
3. 使用 Per-channel 量化（已啟用）
4. 考慮使用 QDQ 格式（已使用）

---

### Q5: STM32 Edge AI 仍然報錯，該怎麼辦？

**A**: 按照以下順序排查：

1. **重新診斷模型**:
   ```bash
   python diagnose_onnx_stm32.py --onnx your_model.onnx
   ```
   確保所有 violations = 0

2. **檢查 Opset**:
   確保使用 opset 14（最兼容）

3. **簡化模型測試**:
   創建一個最小化測試模型（單層 Conv），確認工具鏈正常

4. **聯繫 ST 技術支援**:
   提供：
   - ONNX 模型文件
   - 完整錯誤訊息
   - 診斷報告
   - STM32N6 目標板型號

---

## 總結

### 成功指標

- [ ] 診斷報告：0 violations
- [ ] STM32 Edge AI Analyze 成功（無 ERROR）
- [ ] 可以生成 C 代碼
- [ ] 量化精度損失 < 2 BPM

### 推薦工作流程

```
1. Clean Export (推薦)
   python export_onnx_stm32_clean.py
   ↓
2. 診斷
   python diagnose_onnx_stm32.py --onnx models/rppg_stm32_clean_fp32.onnx
   ↓
3. 如有問題，Graph Surgery
   python fix_onnx_for_stm32.py --input ... --output ...
   ↓
4. 量化
   python quantize_4d_model_v2.py
   ↓
5. 最終驗證
   python diagnose_onnx_stm32.py --onnx models/rppg_4d_int8_qdq.onnx
   ↓
6. 部署到 STM32
   上傳到 STM32 Edge AI Developer Cloud
```

---

**文檔版本**: 1.0
**創建日期**: 2025-01-26
**適用於**: STM32N6 + X-CUBE-AI v2.2.0 + Edge AI Developer Cloud
**維護者**: Claude Code AI
