# STM32N6 部署故障排除指南

基於 Zero-DCE 項目失敗經驗的完整故障排除手冊。

---

## 🚨 常見問題與解決方案

### 問題 1: 推論返回 LL_ATON_RT_ERROR (ret=0)

**症狀**:
```c
LL_ATON_RT_RetValues_t ret = LL_ATON_RT_RunEpochBlock(&network);
// ret = 0 (LL_ATON_RT_ERROR)，第一次調用就失敗
```

**根本原因**（基於 Zero-DCE 經驗）:
1. **優化級別 O3 導致緩衝區重疊**
2. **NPU 初始化失敗**
3. **輸入數據格式錯誤**

**解決方案 A：降低優化級別** ⭐ 推薦

```
步驟:
1. 重新打開 STM32CubeMX 項目
2. X-CUBE-AI → Model Settings → Optimization
3. 從 Balanced (O3) 改為 Time (O2) 或 Default (O1)
4. 重新生成代碼
5. 重新編譯和測試
```

**解決方案 B：檢查輸入數據格式**

```c
// ❌ 錯誤：直接使用 uint8 [0, 255]
memcpy(input_buffer, rgb_data, size);

// ✅ 正確：轉換為 int8 [-128, 127]
for (int i = 0; i < size; i++) {
    input_buffer[i] = (int8_t)(rgb_data[i] - 128);
}
```

**解決方案 C：驗證內存配置**

```c
// 檢查輸入/輸出緩衝區地址
printf("Input buffer:  0x%08X\n", (uint32_t)ai_input[0].data);
printf("Output buffer: 0x%08X\n", (uint32_t)ai_output[0].data);

// 確保地址在有效範圍（AXISRAM）
// STM32N6: 0x24000000 - 0x243BFFFF
```

---

### 問題 2: 緩衝區地址重疊

**症狀**（從 Zero-DCE 項目）:
```
network_zerodce.c 分析:
  Input_8_out_0:        addr_base = 0x342e0000, size = 110592
  Transpose_142_out_0:  addr_base = 0x342e0000, size = 110592
  ← 完全重疊！

推論結果:
  - 輸出被輸入覆蓋
  - 永遠返回 0 或垃圾數據
```

**根本原因**: O3 優化的激進內存重用策略

**解決方案**:

**❌ 錯誤方法：手動修改生成的代碼**
```c
// D:\stm32n6-DK_inference_AI\Appli\X-CUBE-AI\App\network.c

// ❌ 不要這樣做！
.addr_base = {(unsigned char *)(0x34350000UL)},  // 手動改地址
```

**原因**:
- 每次重新生成代碼會被覆蓋
- 可能引入新的錯誤
- X-CUBE-AI 自動分配更可靠

**✅ 正確方法：降低優化級別**
```
STM32CubeMX:
  Balanced (O3) → Time (O2)

結果:
  - 自動使用獨立緩衝區
  - 無重疊問題
  - 穩定運行
```

---

### 問題 3: 模型分析失敗

**症狀**:
```
X-CUBE-AI Analyze 錯誤:
  ❌ Error: Cannot parse model
  ❌ Error: Unsupported operator
```

**可能原因**:
1. ONNX opset 版本不兼容
2. QDQ 節點不被識別
3. 模型結構不支持

**解決方案 A：檢查 ONNX opset**

```bash
# 檢查當前 opset
python -c "
import onnx
model = onnx.load('../quantization/models/rppg_int8_qdq.onnx')
print(f'Opset version: {model.opset_import[0].version}')
"

# 應該輸出: Opset version: 13
```

**如果不是 13，重新導出**:
```bash
cd ../quantization
python export_onnx.py --opset 13
python quantize_onnx.py
```

**解決方案 B：驗證 ONNX 格式**

```bash
python -c "
import onnx
model = onnx.load('../quantization/models/rppg_int8_qdq.onnx')
onnx.checker.check_model(model)
print('✅ ONNX model is valid')
"
```

**解決方案 C：檢查支持的算子**

參考：https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html

常見不支持算子：
- Dynamic shapes
- Control flow (If, Loop)
- 某些 Resize modes

---

### 問題 4: 推論結果異常

**症狀**:
```
輸出心率:
  - 全部為 0 BPM
  - 全部為 NaN
  - 超出範圍（< 30 或 > 180 BPM）
  - 每次推論結果相同（無論輸入）
```

**解決方案 A：檢查輸入數據**

```c
// 在推論前打印輸入數據
printf("Input samples:\n");
for (int i = 0; i < 10; i++) {
    printf("  [%d] = %d\n", i, input_buffer[i]);
}

// 預期：int8 範圍 [-128, 127]，有變化
```

**解決方案 B：檢查輸出數據**

```c
// 檢查輸出緩衝區
float* output_ptr = (float*)ai_output[0].data;
printf("Output HR: %.2f BPM\n", output_ptr[0]);

// 檢查是否為 NaN 或 inf
if (isnan(output_ptr[0]) || isinf(output_ptr[0])) {
    printf("❌ Error: Invalid output\n");
}
```

**解決方案 C：對比 ONNX 模型輸出**

```python
# 使用相同輸入在 ONNX 模型上測試
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('rppg_int8_qdq.onnx')

# 使用固定輸入（例如全 10）
test_input = np.full((1, 8, 3, 36, 36, 3), 10, dtype=np.float32)

# 推論
output = sess.run(None, {'input': test_input})[0]
print(f"ONNX output: {output[0,0]:.2f} BPM")

# 在 STM32 上使用相同輸入（全 10-128=-118）
# 輸出應該相近（差異 < 1 BPM）
```

---

### 問題 5: 編譯錯誤

**症狀**:
```
Build error:
  undefined reference to `ai_network_create`
  undefined reference to `LL_ATON_RT_Init`
```

**原因**: Neural-ART 庫未正確鏈接

**解決方案**:

1. **檢查 X-CUBE-AI 組件**:
   ```
   CubeMX → Software Packs → Select Components
   確認勾選: Neural-ART Runtime
   ```

2. **重新生成代碼**:
   ```
   CubeMX → Generate Code
   ```

3. **清理並重新編譯**:
   ```
   STM32CubeIDE:
     Project → Clean
     Project → Build Project
   ```

---

### 問題 6: 內存不足

**症狀**:
```
Build error:
  region `RAM' overflowed by XXX bytes

或運行時:
  HardFault_Handler
```

**原因**:
- 模型太大（激活值佔用過多 RAM）
- 優化級別過低（內存使用大）

**解決方案 A：提高優化級別**

```
CubeMX → X-CUBE-AI → Optimization:
  Default (O1) → Time (O2)

結果:
  - 減少內存使用（通過重用）
  - 但避免 O3（會導致其他問題）
```

**解決方案 B：使用外部記憶體**

```
CubeMX → X-CUBE-AI → Advanced Settings:
  Activation Pool: External RAM (如果有）
```

**解決方案 C：簡化模型**

```
減少模型複雜度:
  - 時間窗口: 8 幀 → 4 幀
  - ROI 數量: 3 → 2
  - ROI 尺寸: 36×36 → 24×24
```

---

## 📊 調試工具

### 1. 內存檢查工具

```c
// 檢查可用 RAM
extern uint8_t _heap_start;
extern uint8_t _heap_end;
uint32_t heap_size = (uint32_t)&_heap_end - (uint32_t)&_heap_start;
printf("Heap size: %lu bytes\n", heap_size);

// 檢查棧使用
extern uint8_t _stack_start;
uint32_t stack_ptr;
__asm volatile ("MRS %0, MSP" : "=r" (stack_ptr));
uint32_t stack_used = (uint32_t)&_stack_start - stack_ptr;
printf("Stack used: %lu bytes\n", stack_used);
```

### 2. 推論時間測量

```c
// 高精度計時
uint32_t cyccnt_start = DWT->CYCCNT;
ai_network_run(network, ai_input, ai_output);
uint32_t cyccnt_end = DWT->CYCCNT;

uint32_t cycles = cyccnt_end - cyccnt_start;
float time_ms = (float)cycles / (SystemCoreClock / 1000.0f);
printf("Inference time: %.2f ms\n", time_ms);
```

### 3. 緩衝區完整性檢查

```c
// 檢查緩衝區邊界
#define CANARY_VALUE 0xDEADBEEF
uint32_t canary_before = CANARY_VALUE;
uint8_t buffer[SIZE];
uint32_t canary_after = CANARY_VALUE;

// 推論後檢查
if (canary_before != CANARY_VALUE || canary_after != CANARY_VALUE) {
    printf("❌ Buffer overflow detected!\n");
}
```

---

## 🔧 Zero-DCE 失敗經驗總結

### 關鍵教訓

1. **永遠不要使用 O3 優化**
   - Zero-DCE 項目在 O3 下完全失敗
   - 所有解決方案（包括手動修改緩衝區地址）都無效
   - O2 或 O1 是唯一穩定選擇

2. **不要手動修改生成的代碼**
   - 修改 `network_zerodce.c` 中的緩衝區地址無效
   - NPU 仍然使用內部硬編碼地址
   - 每次重新生成會覆蓋修改

3. **信任 X-CUBE-AI 自動配置**
   - 自動內存分配比手動更可靠
   - 工具了解 NPU 的限制和要求

4. **從最保守配置開始**
   - 先用 O1 確保能運行
   - 再逐步優化（O1 → O2）
   - 只有在性能充足時才考慮 O3

---

## 參考資源

- ST Community Forum: https://community.st.com/t5/stm32-mcus-ai/bd-p/stm32-mcus-ai
- X-CUBE-AI Documentation: https://www.st.com/en/embedded-software/x-cube-ai.html
- Neural-ART Operator Support: https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html

---

**版本**: 1.0
**基於**: Zero-DCE 項目失敗經驗（2025-01-11）
**創建日期**: 2025-01-20
