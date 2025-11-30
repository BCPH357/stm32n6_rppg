# STM32N6 部署指南 - rPPG Multi-ROI INT8 模型

完整的 INT8 ONNX 模型到 STM32N6 NPU 部署流程。

---

## 📋 前提條件

### 已完成的步驟

✅ **量化完成**：
- INT8 ONNX 模型：`../quantization/models/rppg_int8_qdq.onnx`
- 精度驗證：MAE 增加 < 2 BPM（可接受）

### 必要工具

1. **STM32CubeMX** (最新版本)
   - 下載：https://www.st.com/en/development-tools/stm32cubemx.html

2. **X-CUBE-AI** 10.2 或更高
   - 安裝在 STM32CubeMX 中

3. **STM32CubeIDE** (用於編譯和調試)
   - 下載：https://www.st.com/en/development-tools/stm32cubeide.html

4. **STM32N6-DK** 開發板
   - 硬件連接：USB-C 連接到 PC

---

## 🚀 部署流程

### Step 1: STM32CubeMX 項目配置

#### 1.1 創建新項目

1. 打開 **STM32CubeMX**
2. `File` → `New Project`
3. 選擇開發板：
   - `Board Selector` → 搜索 `STM32N6570-DK`
   - 選擇並點擊 `Start Project`

#### 1.2 基礎配置

**System Core:**
- `SYS`:
  - Timebase Source: `TIM6`
- `CORTEX_M55`:
  - 保持默認設置
  - CPU ICache: Enabled
  - CPU DCache: Enabled

**Clock Configuration:**
- SYSCLK: 600 MHz（最大性能）
- Neural-ART NPU Clock: 800 MHz

---

### Step 2: X-CUBE-AI 配置

#### 2.1 添加 X-CUBE-AI 軟體包

1. `Software Packs` → `Select Components`
2. 展開 `STMicroelectronics.X-CUBE-AI`
3. 勾選：
   - `Core`
   - `Application`
   - `Neural-ART Runtime`

#### 2.2 配置 AI 模型

**導航**：`Software Packs` → `STMicroelectronics.X-CUBE-AI` → `Model Settings`

**Model 1 Configuration:**

```yaml
Model Settings:
  Model Name: rppg_multi_roi
  Model File: [Browse] → 選擇 ../quantization/models/rppg_int8_qdq.onnx
  Series: STM32N6

Optimization:
  ⚠️  關鍵配置！
  Level: Time (O2)               # 或 Default (O1)

  ❌ 避免 Balanced (O3)！
  原因：基於 Zero-DCE 失敗經驗，O3 會導致：
    - 激進內存重用
    - 緩衝區地址重疊
    - 推論返回錯誤 (LL_ATON_RT_ERROR)

  Compression: None              # INT8 已壓縮

Runtime:
  Runtime: Neural-ART            # STM32N6 專用 NPU runtime

Validation:
  Mode: Random                   # 快速驗證
  Number of Random Inputs: 10

Advanced Settings:
  Input Data Type: int8          # ⚠️  關鍵！匹配量化格式
  Output Data Type: float32      # 建議用 float（便於後處理）
  Memory Pools: Auto             # 讓 X-CUBE-AI 自動分配
```

#### 2.3 分析模型

1. 點擊 **Analyze** 按鈕
2. 等待分析完成（約 1-2 分鐘）
3. 檢查分析報告：

**預期結果**：
```
Model Summary:
  Model: rppg_multi_roi (INT8)
  Input: input (1, 8, 3, 36, 36, 3) - int8
  Output: output (1, 1) - float32

  Parameters: ~20,193
  Activations: ~110 KB
  RAM Usage: ~200-300 KB
  Flash Usage: ~20-30 KB

  Estimated Inference Time: 5-15 ms @ 800 MHz NPU

  ✅ Model is compatible with STM32N6
```

**如果出現錯誤**：
- 檢查 ONNX 模型格式（opset 13）
- 確認 INT8 QDQ 格式
- 參考 `troubleshooting.md`

#### 2.4 驗證模型

1. 點擊 **Validate on Desktop** 按鈕
2. 檢查驗證結果：
   - 推論成功完成
   - 輸出範圍合理（30-180 BPM）

---

### Step 3: 生成代碼

#### 3.1 項目設置

**Project Manager:**
- Project Name: `rppg_inference`
- Project Location: 選擇工作目錄
- Toolchain: `STM32CubeIDE`

#### 3.2 生成代碼

1. 點擊 **Generate Code** 按鈕
2. 等待生成完成
3. 選擇 **Open Project** 打開 STM32CubeIDE

---

### Step 4: 實現推論邏輯

#### 4.1 生成的代碼結構

```
rppg_inference/
├── Core/
│   ├── Src/
│   │   ├── main.c              # 主程式
│   │   └── app_x-cube-ai.c     # AI 應用代碼
│   └── Inc/
│       └── app_x-cube-ai.h
├── X-CUBE-AI/
│   └── App/
│       ├── network_rppg_multi_roi.c    # 生成的網絡代碼
│       └── network_rppg_multi_roi.h
└── Middlewares/
    └── ST/
        └── AI/                  # Neural-ART runtime
```

#### 4.2 輸入數據預處理

在 `app_x-cube-ai.c` 中添加預處理邏輯：

參考：`preprocessing_code.c`

**關鍵步驟**：
1. 從攝像頭獲取 RGB 幀（640×480×3）
2. 臉部檢測（Haar Cascade 或簡化版）
3. 提取 3 個 ROI（前額、左右臉頰）
4. Resize 到 36×36×3
5. 轉換為 INT8：`pixel_int8 = (uint8_t)pixel - 128`
6. 組織為 (8, 3, 36, 36, 3) 時間窗口

#### 4.3 推論執行

```c
// 參考生成的 app_x-cube-ai.c
void MX_X_CUBE_AI_Process(void)
{
    // 1. 填充輸入數據（8×3×36×36×3 = 279,936 int8）
    fill_input_buffer(ai_input_buffer);

    // 2. 運行推論
    ai_i32 nbatch = ai_network_run(network, ai_input, ai_output);

    if (nbatch != 1) {
        printf("Error: Inference failed\n");
        return;
    }

    // 3. 讀取輸出（float32）
    float hr_bpm = ((float*)ai_output[0].data)[0];

    // 4. 範圍檢查
    if (hr_bpm < 30.0f || hr_bpm > 180.0f) {
        printf("Warning: HR out of range: %.2f BPM\n", hr_bpm);
    } else {
        printf("Heart Rate: %.2f BPM\n", hr_bpm);
    }
}
```

詳細代碼參考：`postprocessing_code.c`

---

### Step 5: 編譯與燒錄

#### 5.1 編譯項目

在 STM32CubeIDE 中：
1. `Project` → `Build Project`
2. 確認編譯成功（0 errors）
3. 檢查內存使用：
   ```
   Memory region         Used Size
   RAM                   ~250 KB
   FLASH                 ~150 KB
   ```

#### 5.2 燒錄到開發板

1. 連接 STM32N6-DK 到 PC（USB-C）
2. `Run` → `Debug` (或按 F11)
3. 確認程式啟動

---

### Step 6: 驗證推論

#### 6.1 初步測試

使用固定輸入測試推論：

```c
// 填充測試數據（全 10）
memset(ai_input_buffer, 10, 279936);  // 8*3*36*36*3

// 運行推論
ai_network_run(network, ai_input, ai_output);

// 檢查輸出
float hr = ((float*)ai_output[0].data)[0];
printf("Test HR: %.2f BPM\n", hr);
```

**預期結果**：
- 推論成功完成（不返回錯誤）
- 輸出在合理範圍（30-180 BPM）

#### 6.2 實際數據測試

1. 整合攝像頭輸入
2. 實現 ROI 提取（簡化版或移植 OpenCV）
3. 累積 8 幀時間窗口
4. 運行推論並顯示心率

---

### Step 7: 性能優化

#### 7.1 測量推論時間

```c
uint32_t start_tick = HAL_GetTick();
ai_network_run(network, ai_input, ai_output);
uint32_t end_tick = HAL_GetTick();

printf("Inference time: %lu ms\n", end_tick - start_tick);
```

**目標**：< 15 ms/幀

#### 7.2 如果性能不足

**選項 A：提高優化級別**
- 嘗試從 O1 → O2（但避免 O3！）

**選項 B：簡化預處理**
- 降低 ROI 解析度（36×36 → 24×24）
- 簡化臉部檢測算法

**選項 C：模型簡化**
- 減少時間窗口（8 幀 → 4 幀）
- 減少 ROI 數量（3 → 2）

---

## ⚠️  關鍵注意事項（基於 Zero-DCE 經驗）

### 1. 優化級別選擇

```
❌ Balanced (O3):
   - 激進內存重用
   - 96+ 緩衝區共享同一起始地址
   - 導致緩衝區重疊和推論錯誤

✅ Time (O2):
   - 性能優先，內存使用較大
   - 穩定性好

✅ Default (O1):
   - 保守配置，最穩定
   - 首次部署推薦
```

### 2. 不要手動修改生成代碼

**錯誤做法**：
```c
// ❌ 不要手動修改 network_rppg_multi_roi.c 中的緩衝區地址
.addr_base = {(unsigned char *)(0x34350000UL)},  // 手動修改
```

**正確做法**：
- 讓 X-CUBE-AI 自動分配內存
- 如果遇到緩衝區問題，降低優化級別
- 參考 `troubleshooting.md` 中的 Zero-DCE 教訓

### 3. 輸入數據格式

**INT8 範圍映射**：
```c
// RGB [0, 255] → INT8 [-128, 127]
for (int i = 0; i < size; i++) {
    input_int8[i] = (int8_t)(rgb_buffer[i] - 128);
}
```

**錯誤**：直接使用 [0, 255] → 推論結果異常

---

## 📊 預期性能

### STM32N6 NPU 性能

```
硬件配置（VOS Low）:
- Cortex-M55: 600 MHz
- Neural-ART NPU: 800 MHz
- NPU SRAM: 800 MHz

推論性能:
- 單幀推論：5-15 ms
- FPS: 66-200 fps（理論最大）
- 實際應用：~30 fps（8 幀窗口 + 預處理）
- 功耗: ~300-500 mW
```

### 精度預期

| 場景 | PyTorch FP32 | ONNX INT8 | STM32N6 INT8 |
|------|-------------|-----------|-------------|
| **MAE** | 4.65 BPM | 5.5-7.0 BPM | 5.5-7.5 BPM |
| **RMSE** | 6.63 BPM | 7.5-9.0 BPM | 7.5-9.5 BPM |

**說明**：STM32N6 精度應與 ONNX INT8 相近（差異 < 0.5 BPM）

---

## 🔧 下一步

### 整合到完整系統

1. **攝像頭整合**：
   - 使用 DCMI 接口連接攝像頭
   - 配置 DMA 自動傳輸幀數據

2. **顯示輸出**：
   - LCD 顯示心率數值
   - 或透過 UART 傳輸到 PC

3. **優化功耗**：
   - 動態頻率調整（DFS）
   - 低功耗模式（Sleep/Stop）

4. **Web 整合**：
   - STM32N6 推論 + WiFi 傳輸
   - 與現有 Web 應用整合

---

## 參考文件

- `cubemx_config.md` - 詳細 CubeMX 配置截圖
- `preprocessing_code.c` - 完整預處理代碼
- `postprocessing_code.c` - 完整後處理代碼
- `troubleshooting.md` - 故障排除指南
- `../quantization/README.md` - 量化流程文檔

---

**版本**: 1.0
**創建日期**: 2025-01-20
**維護者**: Claude Code AI
