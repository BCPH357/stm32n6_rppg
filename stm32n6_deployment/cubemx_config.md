# STM32CubeMX 配置詳細指南

本文檔提供 STM32N6 rPPG 項目的詳細 CubeMX 配置步驟。

---

## X-CUBE-AI 配置截圖說明

### Model Settings 配置

```
┌─────────────────────────────────────────────────────────────┐
│ X-CUBE-AI - Model Settings                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Model Configuration                                          │
│   Model Name:            rppg_multi_roi                      │
│   Model Type:            ONNX                                │
│   Model File:            [Browse...] rppg_int8_qdq.onnx     │
│                                                              │
│ Target Platform                                              │
│   Series:                STM32N6                             │
│   Board:                 STM32N6570-DK (optional)            │
│                                                              │
│ Optimization                                                 │
│   ┌───────────────────────────────────────────────┐         │
│   │ ⚠️  關鍵配置！                                  │         │
│   │                                                │         │
│   │ ( ) Balanced (O3)    ← ❌ 避免！              │         │
│   │ (●) Time (O2)        ← ✅ 推薦                │         │
│   │ ( ) Default (O1)     ← ✅ 最穩定              │         │
│   │ ( ) RAM (O2s)                                 │         │
│   └───────────────────────────────────────────────┘         │
│                                                              │
│   Compression:           None                                │
│                                                              │
│ Runtime                                                      │
│   Runtime:               Neural-ART                          │
│   Version:               10.2.0 (或更高)                     │
│                                                              │
│ Validation (Optional)                                        │
│   Mode:                  Random                              │
│   Number of Inputs:      10                                  │
│                                                              │
│ [Analyze] [Validate on Desktop] [Generate Code]             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Settings

```
┌─────────────────────────────────────────────────────────────┐
│ X-CUBE-AI - Advanced Settings                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Data Format                                                  │
│   Input Data Type:       int8          ← ⚠️  匹配量化     │
│   Output Data Type:      float32       ← 推薦              │
│                                                              │
│ Memory Configuration                                         │
│   Memory Pools:          Auto          ← 自動分配          │
│   Activation Buffer:     Auto                                │
│   Weight Buffer:         Auto                                │
│                                                              │
│ Code Generation                                              │
│   Inference API:         ai_network_run                      │
│   Library:               Static                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 完整配置步驟

### Step 1: 新建項目

1. **啟動 STM32CubeMX**

2. **Board Selector**
   - 搜索：`STM32N6570-DK`
   - 選擇開發板
   - 點擊 `Start Project`

3. **初始化確認**
   - 彈出對話框：`Initialize all peripherals with their default Mode?`
   - 選擇：`Yes`

---

### Step 2: System Core 配置

#### 2.1 SYS Configuration

```
Pinout & Configuration → System Core → SYS

Settings:
  Debug:                    Serial Wire
  Timebase Source:          TIM6  ← 重要！不使用 SysTick
```

#### 2.2 CORTEX_M55 Configuration

```
Pinout & Configuration → System Core → CORTEX_M55

Settings:
  CPU ICache:               Enabled
  CPU DCache:               Enabled
  MPU Settings:             Disabled (或根據需求配置)
```

---

### Step 3: Clock Configuration

```
Clock Configuration 標籤頁

關鍵時鐘設置:
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  Input Clock:          HSI (64 MHz)                         │
│        ↓                                                    │
│     PLL1                                                    │
│        ↓                                                    │
│  SYSCLK:              600 MHz    ← 最大性能               │
│        ↓                                                    │
│  AHB:                 600 MHz                               │
│  APB1:                150 MHz                               │
│  APB2:                150 MHz                               │
│  APB3:                150 MHz                               │
│  APB4:                150 MHz                               │
│        ↓                                                    │
│  Neural-ART NPU:      800 MHz    ← NPU 時鐘               │
│  NPU SRAM:            800 MHz                               │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**配置步驟**:
1. SYSCLK: 選擇 `PLLCLK`
2. 調整 PLL 參數達到 600 MHz
3. NPU 時鐘自動配置為 800 MHz

---

### Step 4: X-CUBE-AI 軟體包添加

#### 4.1 打開軟體包管理器

```
Software Packs → Select Components
```

#### 4.2 安裝 X-CUBE-AI（如果未安裝）

1. `Manage Software Packs`
2. 搜索 `X-CUBE-AI`
3. 選擇最新版本（10.2+）
4. 點擊 `Install Now`
5. 等待安裝完成

#### 4.3 選擇組件

```
在 Software Packs → Select Components 中勾選:

STMicroelectronics.X-CUBE-AI
  ├─ [✓] Core                  ← AI 核心庫
  ├─ [✓] Application           ← 應用模板
  └─ [✓] Neural-ART Runtime    ← STM32N6 NPU runtime
```

點擊 `OK` 確認。

---

### Step 5: X-CUBE-AI Model 配置

#### 5.1 打開 AI 配置頁面

```
Software Packs → STMicroelectronics.X-CUBE-AI → Mode
```

勾選 `Enabled`

#### 5.2 Model Settings

```
Software Packs → STMicroelectronics.X-CUBE-AI → Model Settings

Model 1:
  ┌────────────────────────────────────────────────────┐
  │ Model Name:        rppg_multi_roi                   │
  │ Model File:        [Browse...]                      │
  │                    → 選擇:                           │
  │                    D:\MIAT\rppg\quantization\       │
  │                    models\rppg_int8_qdq.onnx        │
  │                                                      │
  │ Series:            STM32N6                           │
  │                                                      │
  │ Optimization:      Time (O2)     ← 推薦            │
  │                    或 Default (O1) ← 最穩定        │
  │                                                      │
  │ Compression:       None                              │
  │                                                      │
  │ Runtime:           Neural-ART                        │
  │                                                      │
  │ Validation:        Random (10 inputs)                │
  └────────────────────────────────────────────────────┘
```

#### 5.3 點擊 Analyze

1. 點擊 `Analyze` 按鈕
2. 等待分析（1-2 分鐘）
3. 查看分析報告

**預期報告內容**:
```
╔═══════════════════════════════════════════════════════════╗
║ X-CUBE-AI - Analysis Report                               ║
╠═══════════════════════════════════════════════════════════╣
║                                                            ║
║ Model: rppg_multi_roi (INT8)                               ║
║                                                            ║
║ Network Information:                                       ║
║   Input tensor:      input                                 ║
║     Shape:           (1, 8, 3, 36, 36, 3)                  ║
║     Data type:       int8                                  ║
║     Size:            279,936 bytes                         ║
║                                                            ║
║   Output tensor:     output                                ║
║     Shape:           (1, 1)                                ║
║     Data type:       float32                               ║
║     Size:            4 bytes                               ║
║                                                            ║
║ Memory Footprint:                                          ║
║   Weights:           ~20 KB                                ║
║   Activations:       ~110 KB                               ║
║   Total RAM:         ~250 KB                               ║
║                                                            ║
║ Performance Estimate (800 MHz NPU):                        ║
║   Inference time:    5-15 ms                               ║
║   Throughput:        66-200 fps                            ║
║                                                            ║
║ ✅ Model is compatible with STM32N6!                      ║
║                                                            ║
╚═══════════════════════════════════════════════════════════╝
```

**如果分析失敗**:
- 檢查 ONNX 文件路徑
- 確認 opset 版本（應為 13）
- 參考 `troubleshooting.md`

---

### Step 6: Project Manager 配置

```
Project Manager 標籤頁

Project:
  Project Name:          rppg_inference
  Project Location:      D:\Projects\STM32\rppg_inference
  Toolchain / IDE:       STM32CubeIDE

Code Generator:
  [✓] Generate peripheral initialization as a pair of '.c/.h' files per peripheral
  [✓] Delete previously generated files when not re-generated
  [ ] Keep User Code when re-generating

HAL Settings:
  [ ] Use HAL (默認)
```

---

### Step 7: 生成代碼

1. **保存項目**
   - `File` → `Save Project`

2. **生成代碼**
   - 點擊 `GENERATE CODE` 按鈕（右上角）
   - 或按 `Alt + K`

3. **等待生成**
   - 進度條顯示生成過程
   - 約 30-60 秒

4. **打開項目**
   - 彈出對話框：`Open Project`
   - 選擇 `Open Folder` 或 `Open with STM32CubeIDE`

---

## 記憶體配置注意事項

### 默認記憶體映射（STM32N6）

```
Memory Layout:
┌────────────────────────────────────────────────────────────┐
│ AXISRAM1 (0x24000000):  1024 KB  │ CPU RAM (通用)         │
│ AXISRAM2 (0x24100000):  1024 KB  │ CPU RAM                │
│ AXISRAM3 (0x24200000):   432 KB  │ NPU RAM                │
│ AXISRAM4 (0x24270000):   448 KB  │ NPU RAM                │
│ AXISRAM5 (0x242e0000):   448 KB  │ NPU RAM                │
│ AXISRAM6 (0x24350000):   432 KB  │ NPU RAM                │
├────────────────────────────────────────────────────────────┤
│ Flash     (0x70000000):  4096 KB │ 模型權重 + 程式碼      │
└────────────────────────────────────────────────────────────┘
```

### X-CUBE-AI 自動分配

**建議**：讓 X-CUBE-AI 自動分配記憶體（`Memory Pools: Auto`）

**原因**（基於 Zero-DCE 經驗）：
- ✅ 自動避免緩衝區重疊
- ✅ 優化內存使用
- ❌ 手動修改容易出錯

**如果需要手動配置**（不推薦）：
```
Advanced Settings → Memory Configuration:
  Activation Pool:     AXISRAM3 (0x24200000)
  Weight Pool:         Flash (0x70000000)
```

---

## 驗證配置

### 桌面驗證（推薦）

在生成代碼前進行驗證：

1. **點擊 Validate on Desktop**
2. **查看驗證結果**:
   ```
   ✅ Validation passed
   10/10 random tests successful
   Output range: 45.23 - 132.67 BPM (合理)
   ```

3. **如果驗證失敗**:
   - 檢查輸入數據類型（int8）
   - 檢查輸出範圍
   - 參考 `troubleshooting.md`

---

## 常見配置錯誤

### 錯誤 1: 優化級別選擇錯誤

```
❌ 錯誤配置:
   Optimization: Balanced (O3)

結果:
   - 緩衝區地址重疊
   - 推論返回 LL_ATON_RT_ERROR
   - 無法正常運行

✅ 正確配置:
   Optimization: Time (O2) 或 Default (O1)
```

### 錯誤 2: 輸入數據類型不匹配

```
❌ 錯誤配置:
   Input Data Type: float32

結果:
   - 模型期望 int8，實際傳入 float32
   - 推論結果異常

✅ 正確配置:
   Input Data Type: int8 (匹配量化格式)
```

### 錯誤 3: Runtime 選擇錯誤

```
❌ 錯誤配置:
   Runtime: Cortex-M

結果:
   - 未使用 NPU 加速
   - 推論速度慢（100+ ms）

✅ 正確配置:
   Runtime: Neural-ART (STM32N6 專用)
```

---

## 參考資料

- X-CUBE-AI User Manual: https://www.st.com/en/embedded-software/x-cube-ai.html
- STM32N6 Reference Manual: https://www.st.com/stm32n6
- Neural-ART Documentation: https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI

---

**版本**: 1.0
**創建日期**: 2025-01-20
