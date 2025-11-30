# 修正 ONNX Dynamic Batch 問題 - STM32 Edge AI 兼容指南

## 問題描述

### 錯誤訊息

在 STM32 Edge AI Developer Cloud 或 X-CUBE-AI 分析 ONNX 模型時出現：

```
INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'
```

### 根本原因

STM32 Edge AI Core v2.2.0 的 ONNX Parser **不支援 dynamic batch**。

當模型的 input shape 第一維為動態時：

```protobuf
input {
  dim {
    dim_param: "batch"    <- Dynamic batch（問題所在）
  }
  dim {
    dim_value: 72
  }
  dim {
    dim_value: 36
  }
  dim {
    dim_value: 36
  }
}
```

Parser 嘗試讀取 `dim.dim_value` 但得到 `None`，導致錯誤。

---

## 解決方案

### 方案 1：重新轉換模型（推薦）⭐

使用改進的轉換腳本，直接生成固定 batch 的 ONNX。

#### Step 1: 上傳新腳本到服務器

```bash
scp "D:\MIAT\rppg\server_training\convert_to_4d_for_stm32_v2.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
```

#### Step 2: 在服務器上執行轉換

```bash
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/
conda activate rppg_training

# 刪除舊的 ONNX（可選）
rm -f models/rppg_4d_fp32.onnx

# 使用 v2 腳本重新轉換
python convert_to_4d_for_stm32_v2.py
```

**關鍵改進**：

```python
# v1 版本（有問題）
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    dynamic_axes={
        'input': {0: 'batch'},    # <- 這會產生 dynamic batch
        'output': {0: 'batch'}
    }
)

# v2 版本（修正）
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=14,  # STM32N6 最佳相容性
    # 移除 dynamic_axes，固定 batch=1
)
```

#### Step 3: 驗證 ONNX

腳本會自動檢查是否仍有 dynamic batch：

```
[Step 6] Checking input/output shapes...
   Input name: input
     dim[0]: 1           <- 固定為 1（OK）
     dim[1]: 72
     dim[2]: 36
     dim[3]: 36
   [OK] All dimensions are fixed: [1, 72, 36, 36]
```

#### Step 4: 下載到本地

```bash
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_fp32.onnx D:\MIAT\rppg\quantization\models\
```

---

### 方案 2：修正現有模型

如果已經有 ONNX 模型，可以使用修正腳本直接處理。

#### Step 1: 上傳修正腳本

```bash
scp "D:\MIAT\rppg\server_training\fix_onnx_dynamic_batch.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
```

#### Step 2: 執行修正

```bash
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/server_training/
conda activate rppg_training

python fix_onnx_dynamic_batch.py
```

**腳本功能**：

1. ✅ 固定 input/output 的 batch=1
2. ✅ 修正所有 value_info（中間張量）
3. ✅ 修正 Constant nodes 中的動態 shape
4. ✅ 驗證修正結果

**輸出**：

```
[SUCCESS] Dynamic Batch Fixed!

Fixed ONNX model: models/rppg_4d_fp32_fixed.onnx

Input 'input': [1, 72, 36, 36]
Output 'output': [1, 1]
[OK] All shapes are fixed (no dynamic batch)
```

#### Step 3: 下載修正後的模型

```bash
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_fp32_fixed.onnx D:\MIAT\rppg\quantization\models\
```

---

## 驗證修正

### 方法 1：使用 onnx Python 庫

```python
import onnx

model = onnx.load("rppg_4d_fp32.onnx")

for input_tensor in model.graph.input:
    print(f"Input: {input_tensor.name}")
    for i, dim in enumerate(input_tensor.type.tensor_type.shape.dim):
        if dim.HasField('dim_value'):
            print(f"  dim[{i}]: {dim.dim_value} (fixed)")
        elif dim.HasField('dim_param'):
            print(f"  dim[{i}]: {dim.dim_param} (DYNAMIC - BAD!)")
```

**預期輸出**（修正成功）：

```
Input: input
  dim[0]: 1 (fixed)
  dim[1]: 72 (fixed)
  dim[2]: 36 (fixed)
  dim[3]: 36 (fixed)
```

### 方法 2：使用 Netron 可視化

1. 打開 https://netron.app/
2. 上傳 ONNX 模型
3. 點擊 input 節點
4. 檢查 type：應該顯示 `int8[1,72,36,36]` 而非 `int8[batch,72,36,36]`

### 方法 3：在 STM32 Edge AI Developer Cloud 測試

1. 上傳修正後的 ONNX
2. 點擊 "Analyze"
3. 應該**不再出現** `INTERNAL ERROR` 錯誤
4. 顯示模型分析結果（MACs, Parameters, Memory）

---

## 量化流程更新

修正 dynamic batch 後，繼續進行 INT8 量化：

### Step 1: 更新量化腳本中的路徑（如果使用方案 2）

編輯 `quantize_4d_model_v2.py`，將輸入路徑改為：

```python
# 修改前
fp32_path = models_dir / "rppg_4d_fp32.onnx"

# 修改後（如果使用 fix_onnx_dynamic_batch.py）
fp32_path = models_dir / "rppg_4d_fp32_fixed.onnx"
```

### Step 2: 執行量化

```bash
cd /mnt/data_8T/ChenPinHao/server_training/
conda activate rppg_training

python quantize_4d_model_v2.py
python evaluate_quantized_model.py
```

### Step 3: 下載 INT8 模型

```bash
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
```

---

## STM32 Edge AI Developer Cloud 配置

### 導入模型

1. **Login**: https://stedgeai-dc.st.com/
2. **New Project** → Upload ONNX
3. **Analyze Model**:
   - Target: STM32N6
   - Optimization: **O1** 或 **O2**（避免 O3）
   - Compression: Optional

### 驗證輸入形狀

在 Analysis 結果中，確認：

```
Input Tensors:
  - input: int8[1,72,36,36]  <- 固定 batch=1

Output Tensors:
  - output: float32[1,1]
```

### 可能的警告（可忽略）

```
WARNING: Input batch size is 1
```

這是正常的，STM32 部署通常使用 batch=1。

---

## 技術細節

### Dynamic Batch 的產生原因

在 PyTorch 轉 ONNX 時，使用 `dynamic_axes` 參數：

```python
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    dynamic_axes={
        'input': {0: 'batch'},   # 第 0 維設為動態
        'output': {0: 'batch'}
    }
)
```

這會在 ONNX protobuf 中生成：

```protobuf
tensor_type {
  shape {
    dim {
      dim_param: "batch"    # 動態維度，沒有 dim_value
    }
    ...
  }
}
```

### STM32 Parser 的限制

X-CUBE-AI 的 ONNX Parser 假設所有維度都有 `dim_value`：

```python
# STM32 Parser 偽代碼
for dim in input_shape:
    value = dim.dim_value      # 如果是 dim_param，這裡返回 None
    buffer_size += value       # None.get_value() -> ERROR
```

### 為何 batch=1 是合理的

在嵌入式部署中：

- ✅ 即時推論通常一次處理一個樣本（單張圖像、單個音頻片段）
- ✅ 記憶體有限（batch=1 降低 RAM 需求）
- ✅ 延遲優先（batch>1 會增加延遲）
- ✅ 簡化部署（無需處理批次維度）

---

## 常見問題

### Q1: 為何不能直接在 STM32 上支援 dynamic batch？

**A**: STM32 NPU 的記憶體分配是靜態的，必須在編譯時確定所有張量大小。Dynamic batch 需要動態記憶體分配，嵌入式系統通常不支援。

### Q2: 如果我需要處理多個樣本怎麼辦？

**A**: 在應用層循環調用推論：

```c
for (int i = 0; i < num_samples; i++) {
    // 填充輸入緩衝區（單個樣本）
    memcpy(input_buffer, samples[i], input_size);

    // 執行推論
    LL_ATON_RT_RunEpochBlock(&network);

    // 讀取輸出
    output[i] = output_buffer[0];
}
```

### Q3: Opset 14 vs 13，有何差異？

**A**:

| Opset | STM32N6 支援 | 特性 |
|-------|-------------|------|
| **13** | ✅ 良好 | PyTorch 1.10+ 默認 |
| **14** | ✅ **最佳** | 更多運算支援，推薦 |
| **15+** | ⚠️ 部分 | 可能有兼容性問題 |

建議使用 **Opset 14**。

### Q4: 如果修正後仍然報錯？

**A**: 可能的原因：

1. **Reshape/Squeeze/Unsqueeze 節點仍有動態 shape**
   - 解決：使用 `fix_onnx_dynamic_batch.py` 修正 Constant nodes

2. **模型包含不支援的運算**
   - 檢查 X-CUBE-AI 支援列表：https://www.st.com/en/embedded-software/x-cube-ai.html

3. **Opset 版本過高**
   - 降級到 Opset 14：使用 ONNX simplifier

4. **模型結構過於複雜**
   - 簡化模型架構
   - 移除不必要的運算

---

## 參考資源

### 官方文檔

- **X-CUBE-AI User Manual**: https://www.st.com/resource/en/user_manual/um2526-getting-started-with-xcubeai-expansion-package-for-artificial-intelligence-ai-stmicroelectronics.pdf
- **STM32 Edge AI Developer Cloud**: https://stedgeai-dc.st.com/
- **ONNX Specification**: https://github.com/onnx/onnx/blob/main/docs/IR.md

### 相關工具

- **Netron** (ONNX 可視化): https://netron.app/
- **ONNX Simplifier**: https://github.com/daquexian/onnx-simplifier
- **onnx-tool**: https://github.com/ThanatosShinji/onnx-tool

---

## 總結

**核心要點**：

1. ✅ STM32 不支援 dynamic batch，必須固定為常數（通常是 1）
2. ✅ 使用 `convert_to_4d_for_stm32_v2.py` 重新轉換（推薦）
3. ✅ 或使用 `fix_onnx_dynamic_batch.py` 修正現有模型
4. ✅ 驗證所有維度都是 `dim_value`，無 `dim_param`
5. ✅ 使用 Opset 14 確保最佳相容性

**成功指標**：

- [ ] ONNX input shape: `[1, 72, 36, 36]`（固定）
- [ ] STM32 Edge AI Analyze 成功（無 INTERNAL ERROR）
- [ ] 顯示模型統計（MACs, Memory）
- [ ] 可以生成 C 代碼

---

**文檔版本**: 1.0
**創建日期**: 2025-01-26
**適用於**: STM32N6 + X-CUBE-AI + Edge AI Developer Cloud
