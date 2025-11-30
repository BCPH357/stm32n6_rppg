# 部署 4D rPPG 模型到 STM32N6 的完整流程

## 問題背景

**問題**: X-CUBE-AI 只支持最多 4D 張量，但我們的模型輸入是 6D
- ❌ 原始輸入: `(B, 8, 3, 36, 36, 3)` = 6D
- ✅ 修改後輸入: `(B, 72, 36, 36)` = 4D (合併 T×ROI×C 到 channel 維度)

**關鍵點**:
- ✅ **不需要重新訓練** - 只是輸入形狀不同，權重完全相同
- ✅ **需要重新導出 ONNX** - 從 6D 輸入改為 4D 輸入
- ✅ **需要重新量化** - 基於新的 4D ONNX 模型

---

## Step 1: 上傳文件到服務器

```bash
# 上傳轉換腳本
scp "D:\MIAT\rppg\server_training\convert_to_4d_for_stm32.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/

# 上傳量化腳本
scp "D:\MIAT\rppg\server_training\quantize_4d_model.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
```

---

## Step 2: 在服務器上執行轉換

```bash
# 連接到服務器
ssh miat@140.115.53.67

# 進入工作目錄
cd /mnt/data_8T/ChenPinHao/server_training/

# 激活環境
conda activate rppg_training

# 執行轉換（6D → 4D ONNX FP32）
python convert_to_4d_for_stm32.py
```

**預期輸出**:
```
======================================================================
Convert 6D Model to 4D ONNX for STM32 (Server-Side)
======================================================================

[Step 1] Loading trained 6D model...
   [OK] Loaded: checkpoints/best_model.pth
   Parameters: 20,193

[Step 2] Creating 4D model (shared weights)...
   [OK] 4D model created

[Step 3] Verifying equivalence...
   6D output: 72.3456 BPM
   4D output: 72.3456 BPM
   Difference: 0.000001 BPM
   [OK] Models are equivalent

[Step 4] Exporting to ONNX...
   [OK] Exported: models/rppg_4d_fp32.onnx

[Step 5] Verifying ONNX...
   Input shape: [0, 72, 36, 36] (batch=0 means dynamic)
   Output shape: [0, 1]
   [OK] ONNX validation passed

======================================================================
[SUCCESS] Conversion Complete!
======================================================================
```

---

## Step 3: 量化為 INT8

```bash
# 仍在服務器上
cd /mnt/data_8T/ChenPinHao/server_training/

# 安裝依賴（如果還沒安裝）
pip install onnxruntime onnx

# 執行量化
python quantize_4d_model.py
```

**預期輸出**:
```
======================================================================
Quantize 4D Model to INT8 for STM32
======================================================================

[Input] models/rppg_4d_fp32.onnx
   Size: 80.50 KB

[Method 2] Static Quantization (Recommended)...
   Generating 200 calibration samples...
   [OK] Calibration data ready: 200 samples
   [OK] Saved: models/rppg_4d_int8.onnx

[Verification] Comparing FP32 vs INT8...
   Mean Absolute Error: 1.23 BPM
   Max Absolute Error:  3.45 BPM
   [OK] Quantization quality: GOOD (MAE < 5 BPM)

======================================================================
[SUCCESS] Quantization Complete!
======================================================================

Model sizes:
   FP32: 80.50 KB
   INT8: 22.30 KB
   Compression: 3.61x

Quantization error:
   Mean: 1.23 BPM
   Max:  3.45 BPM
```

---

## Step 4: 下載到本地

```bash
# 在本地 Windows 執行
scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8.onnx D:\MIAT\rppg\
```

---

## Step 5: 導入到 STM32CubeMX

### 5.1 打開 X-CUBE-AI

1. 打開 STM32CubeMX
2. 載入你的 STM32N6 項目（或創建新項目）
3. 在左側導航欄選擇 **Software Packs** → **Select Components**
4. 啟用 **X-CUBE-AI**

### 5.2 添加模型

1. 在 X-CUBE-AI 配置頁面，點擊 **Add network**
2. 選擇 `D:\MIAT\rppg\rppg_4d_int8.onnx`
3. 點擊 **Analyze**

### 5.3 配置參數（重要！）

**基於 Zero-DCE 失敗經驗的關鍵配置**:

| 參數 | 推薦值 | 說明 |
|------|-------|------|
| **Optimization** | **Time (O2)** 或 **Default (O1)** | ✅ 穩定 |
| **Optimization** | ~~Balanced (O3)~~ | ❌ **避免使用！會導致緩衝區重疊** |
| **Compression** | None 或 Lossless | 可選 |
| **Runtime** | Neural-ART | STM32N6 NPU |
| **Memory Pools** | Auto | 讓工具自動分配 |

**為什麼避免 O3？**
- 導致激進的內存重用和緩衝區重疊
- 推論第一次調用就返回 `LL_ATON_RT_ERROR`
- 所有手動修復嘗試均失敗（參考 `D:\MIAT\CLAUDE.md`）

### 5.4 分析結果

點擊 **Analyze** 後，應該看到：

```
[Analyze Results]
✅ Model validated successfully
✅ Input shape: (1, 72, 36, 36) - 4D tensor
✅ Output shape: (1, 1)
✅ Total Memory: ~200-500 KB (depending on optimization)
✅ RAM: ~150-300 KB
✅ Flash: ~20-30 KB (INT8 weights)
```

**如果出現錯誤**:
- ❌ "Unexpected combination of configuration and input shape" → 模型仍是 6D，請確認使用 `rppg_4d_int8.onnx`
- ❌ "Unsupported operator" → 檢查 opset 版本（應該是 13）
- ❌ "Memory allocation failed" → 降低優化級別（O2 → O1）

### 5.5 生成代碼

1. 點擊 **Generate Code**
2. 等待代碼生成完成
3. 打開生成的項目

---

## Step 6: 驗證推論

### 6.1 準備測試數據

在 STM32 應用代碼中，需要準備 4D 輸入：

```c
// app_x-cube-ai.c

// 輸入緩衝區: (1, 72, 36, 36) = 93,312 個 int8 值
static int8_t input_buffer[1 * 72 * 36 * 36];  // 93,312 bytes

// 輸出緩衝區: (1, 1) = 1 個 float32 值
static float output_buffer[1];  // 4 bytes

void prepare_input_data() {
    // 從攝像頭獲取 8 幀圖像，每幀提取 3 個 ROI
    // 每個 ROI 是 36×36×3 (RGB)

    for (int t = 0; t < 8; t++) {          // 8 個時間步
        for (int roi = 0; roi < 3; roi++) { // 3 個 ROI
            for (int c = 0; c < 3; c++) {   // 3 個通道 (RGB)
                for (int h = 0; h < 36; h++) {
                    for (int w = 0; w < 36; w++) {
                        // 計算在 4D 張量中的索引
                        int channel_idx = t * 3 * 3 + roi * 3 + c;  // 0-71
                        int idx = channel_idx * 36 * 36 + h * 36 + w;

                        // 填充數據（INT8 範圍 [-128, 127]）
                        // 實際應從圖像提取並正規化
                        input_buffer[idx] = (int8_t)((pixel_value - 128));
                    }
                }
            }
        }
    }
}

void run_inference() {
    // 運行推論
    ai_run(network, input_buffer, output_buffer);

    // 輸出心率
    float heart_rate = output_buffer[0];  // 30-180 BPM
    printf("Heart Rate: %.2f BPM\n", heart_rate);
}
```

### 6.2 測試推論

1. 編譯項目
2. 燒錄到 STM32N6
3. 運行測試
4. 檢查 log 輸出

**成功標誌**:
```
✅ Network initialized
✅ Inference completed
✅ Heart Rate: 72.45 BPM
```

**失敗標誌**:
```
❌ LL_ATON_RT_RunEpochBlock() returned 0 (ERROR)
❌ Network initialization failed
```

如果失敗，參考 `D:\MIAT\rppg\CLAUDE.md` 的故障排除部分。

---

## 故障排除

### 問題 1: STM32CubeMX Analyze 錯誤

**錯誤**: "Unexpected combination of configuration and input shape"

**原因**: 仍在使用 6D ONNX 模型

**解決**:
```bash
# 檢查模型形狀
python -c "import onnx; m=onnx.load('rppg_4d_int8.onnx'); print([d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])"

# 應該輸出: [0, 72, 36, 36]  （batch=0 表示動態）
# 如果是 [0, 8, 3, 36, 36, 3] → 錯誤！使用了舊模型
```

### 問題 2: 量化精度太差

**症狀**: MAE > 10 BPM

**解決**:
1. 增加校準樣本數量（200 → 500）
2. 使用真實數據而非隨機數據
3. 考慮使用 QDQ (Quantize-Dequantize) 格式

### 問題 3: 推論返回錯誤

**症狀**: `LL_ATON_RT_RunEpochBlock()` 返回 0

**解決**:
1. **降低優化級別**: O3 → O2 → O1
2. **檢查內存配置**: 確保 AXISRAM 足夠
3. **驗證輸入數據**: 確保 INT8 範圍正確
4. **參考 Zero-DCE 經驗**: 詳見 `D:\MIAT\CLAUDE.md`

---

## 總結

### ✅ 完成步驟

- [ ] Step 1: 上傳腳本到服務器
- [ ] Step 2: 執行 `convert_to_4d_for_stm32.py` → 生成 `rppg_4d_fp32.onnx`
- [ ] Step 3: 執行 `quantize_4d_model.py` → 生成 `rppg_4d_int8.onnx`
- [ ] Step 4: 下載 `rppg_4d_int8.onnx` 到本地
- [ ] Step 5: 導入到 STM32CubeMX (使用 O1 或 O2)
- [ ] Step 6: 生成代碼並驗證推論

### 🎯 關鍵要點

1. **不需要重新訓練** - 權重完全相同，只是輸入形狀改變
2. **需要重新導出 ONNX** - 從 6D 改為 4D
3. **需要重新量化** - 基於新的 4D ONNX
4. **避免使用 O3 優化** - 會導致緩衝區重疊問題
5. **信任工具的自動配置** - 不要手動修改生成的代碼

### 📚 參考文檔

- 本文件: `D:\MIAT\rppg\DEPLOY_4D_TO_STM32.md`
- Zero-DCE 經驗: `D:\MIAT\CLAUDE.md`
- rPPG 項目記錄: `D:\MIAT\rppg\CLAUDE.md`

---

**最後更新**: 2025-11-26
**作者**: Claude Code AI
