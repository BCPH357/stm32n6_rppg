# 🫀 rPPG 心率檢測系統

**Remote Photoplethysmography (rPPG)** - 基於攝像頭的非接觸式心率檢測系統，部署於 STM32N6 嵌入式平台。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![STM32N6](https://img.shields.io/badge/STM32-N6-03234b.svg)](https://www.st.com/stm32n6)

---

## 📋 目錄

- [專案概述](#-專案概述)
- [快速開始](#-快速開始)
- [目錄結構](#-目錄結構)
- [完整流程](#-完整流程)
- [Web 應用](#-web-應用)
- [STM32N6 部署](#-stm32n6-部署)
- [效能指標](#-效能指標)
- [技術文檔](#-技術文檔)
- [常見問題](#-常見問題)
- [授權](#-授權)

---

## 🎯 專案概述

### 核心功能

本專案實現一個完整的 rPPG 心率檢測系統，包括：

1. **數據處理與訓練**（服務器端）
   - UBFC-rPPG 數據集處理（42 subjects）
   - Multi-ROI 特徵提取（前額、左右臉頰）
   - 健壯的 PPG → HR 標籤計算（Bandpass + Peak Detection）
   - 輕量級模型訓練（MAE: 4.65 BPM, ~20K 參數）

2. **Web 應用**（即時心率監測）
   - Flask + WebSocket 後端架構
   - 攝像頭即時捕獲（30 fps）
   - Haar Cascade 臉部檢測
   - Multi-ROI 推論（~10 fps）
   - 即時圖表顯示（Chart.js）

3. **嵌入式部署**（STM32N6）
   - Pattern A 分離式架構（Spatial CNN on NPU + Temporal Fusion on CPU）
   - INT8 量化（TFLite）
   - 純 C 語言實現（Temporal Fusion）
   - NPU 加速推論

### 模型架構

```
Input: (B, 8, 3, 36, 36, 3)
  ↓ 8 時間步 × 3 ROI × 36×36 RGB
[Shared Spatial CNN] 提取空間特徵 (9,840 params)
  ↓ 產生 24 × 16 特徵向量
[Temporal Fusion] 時序建模 (10,353 params)
  ↓ Conv1D + FC layers
Output: Heart Rate [30, 180] BPM
```

**總參數**: 20,193 個（遠低於 500K 目標）

---

## 🚀 快速開始

### 選項 A: Web 應用（本地測試）

```bash
# 1. 安裝依賴
cd webapp
install.bat

# 2. 啟動服務器
start.bat

# 3. 打開瀏覽器
http://localhost:5000
```

### 選項 B: 完整訓練流程（服務器端）

```bash
# 1. 連接到服務器
ssh miat@140.115.53.67
cd /mnt/data_8T/ChenPinHao/rppg/

# 2. 激活環境
conda activate rppg_training

# 3. 執行各階段
cd 1_preprocessing && python preprocess_data.py
cd ../2_training && python train.py
cd ../3_model_conversion && python migrate_weights.py
cd ../4_quantization/spatial_cnn && python export_tflite_split_v2.py
cd ../temporal_fusion && python export_temporal_fusion_weights.py
```

### 選項 C: STM32N6 部署

參見 [`stm32_rppg/README.md`](stm32_rppg/README.md) 完整部署指南。

---

## 📁 目錄結構

```
rppg/
├── 1_preprocessing/          # 數據前處理
│   ├── preprocess_data.py    # 主要腳本
│   ├── data/                 # 預處理數據
│   └── README.md
│
├── 2_training/               # 模型訓練
│   ├── model.py              # UltraLightRPPG 模型
│   ├── train.py              # 訓練主程式
│   ├── checkpoints/          # 訓練權重
│   └── README.md
│
├── 3_model_conversion/       # 模型轉換
│   ├── model_split.py        # 拆分為 Spatial CNN + Temporal Fusion
│   ├── migrate_weights.py    # 權重遷移
│   └── README.md
│
├── 4_quantization/           # 模型量化
│   ├── spatial_cnn/          # TFLite INT8 量化
│   │   ├── export_tflite_split_v2.py
│   │   └── validate_tflite.py
│   ├── temporal_fusion/      # C 權重導出
│   │   ├── export_temporal_fusion_weights.py
│   │   └── validate_c_vs_pytorch.py
│   └── README.md
│
├── 5_validation/             # 模型驗證
│   ├── evaluate_quantized_model.py
│   ├── test_roi_extraction.py
│   └── README.md
│
├── stm32_rppg/               # STM32N6 部署
│   ├── temporal_fusion/      # Temporal Fusion C 實現
│   │   ├── temporal_fusion.h
│   │   ├── temporal_fusion.c
│   │   └── temporal_fusion_weights_exported.c
│   ├── preprocessing/        # 前處理代碼範例
│   ├── postprocessing/       # 後處理代碼範例
│   ├── docs/                 # 部署文檔
│   └── README.md
│
├── webapp/                   # Web 應用
│   ├── app.py                # Flask 後端
│   ├── inference.py          # 推論邏輯
│   ├── model.py              # 模型定義
│   ├── models/               # 訓練模型
│   ├── static/               # 前端資源
│   ├── templates/            # HTML 模板
│   └── README.md
│
├── models/                   # 共享模型檔案
│   └── spatial_cnn_int8.tflite
│
├── scripts/                  # 輔助腳本
│   ├── setup_env.sh
│   └── run_training.sh
│
├── docs/                     # 文檔
│   └── archive/              # 過時文檔（歷史記錄）
│
├── CLAUDE.md                 # 專案技術概述
├── README.md                 # 本文件
├── requirements_rppg_training.txt     # 訓練環境依賴
├── requirements_tflite_export.txt     # TFLite 導出環境依賴
└── .gitignore
```

---

## 🔄 完整流程

### 階段 1: 數據前處理

**目標**: 處理 UBFC-rPPG 數據集，生成訓練數據

**執行**:
```bash
cd 1_preprocessing
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
```

**輸出**: `data/ubfc_processed.pt` (~15000 樣本)

**詳情**: [`1_preprocessing/README.md`](1_preprocessing/README.md)

---

### 階段 2: 模型訓練

**目標**: 訓練 UltraLightRPPG 模型

**執行**:
```bash
cd 2_training
python train.py --config config.yaml
```

**輸出**: `checkpoints/best_model.pth` (MAE: 4.65 BPM)

**詳情**: [`2_training/README.md`](2_training/README.md)

---

### 階段 3: 模型轉換

**目標**: 拆分為 Spatial CNN 和 Temporal Fusion

**執行**:
```bash
cd 3_model_conversion
python migrate_weights.py
```

**輸出**:
- `checkpoints/spatial_cnn.pth`
- `checkpoints/temporal_fusion.pth`

**詳情**: [`3_model_conversion/README.md`](3_model_conversion/README.md)

---

### 階段 4: 模型量化

**目標**: 量化為 STM32 部署格式

**執行**:
```bash
# Spatial CNN: TFLite INT8
cd 4_quantization/spatial_cnn
conda activate tflite_export
python export_tflite_split_v2.py

# Temporal Fusion: C 權重
cd ../temporal_fusion
conda activate rppg_training
python export_temporal_fusion_weights.py
```

**輸出**:
- `../../models/spatial_cnn_int8.tflite` (~20 KB)
- `../../stm32_rppg/temporal_fusion/temporal_fusion_weights_exported.c` (~200 KB)

**詳情**: [`4_quantization/README.md`](4_quantization/README.md)

---

### 階段 5: 模型驗證

**目標**: 驗證量化精度

**執行**:
```bash
cd 5_validation
python evaluate_quantized_model.py
```

**預期**: MAE 增加 < 1.5 BPM

**詳情**: [`5_validation/README.md`](5_validation/README.md)

---

## 🌐 Web 應用

### 功能特色

- ✅ 即時心率監測（~10 fps）
- ✅ Multi-ROI 可視化（前額、左右臉頰）
- ✅ 歷史心率圖表（Chart.js）
- ✅ 自動臉部檢測（Haar Cascade）
- ✅ WebSocket 即時通訊

### 快速啟動

```bash
cd webapp
install.bat  # 或 pip install -r requirements.txt
start.bat    # 或 python app.py
```

訪問 http://localhost:5000

### 系統需求

- 光線充足（避免逆光、暗光）
- 臉部正對攝像頭（±15° 偏轉可接受）
- 保持相對靜止
- 建議距離：50-100 cm

**詳情**: [`webapp/README.md`](webapp/README.md)

---

## 🚀 STM32N6 部署

### 部署架構

```
Camera (640×480 RGB)
    ↓ 捕獲 8 幀
ROI 提取 (3 × 36×36)
    ↓
Spatial CNN (NPU, INT8) × 24 次推論
    ↓ 產生 24 × 16 特徵
Temporal Fusion (CPU, C 語言)
    ↓
Heart Rate [30, 180] BPM
```

### 快速步驟

1. **STM32CubeMX 配置**
   - 導入 `models/spatial_cnn_int8.tflite`
   - Optimization: O1 或 O2（避免 O3）
   - Runtime: Neural-ART (NPU)

2. **整合代碼**
   - 複製 `stm32_rppg/temporal_fusion/` 到 STM32 項目
   - 實現應用層邏輯（參考 `preprocessing/` 和 `postprocessing/`）

3. **編譯與測試**
   - 編譯項目
   - Flash 到 STM32N6
   - 驗證心率輸出

**完整指南**: [`stm32_rppg/README.md`](stm32_rppg/README.md)

### 關鍵配置

| 配置項 | 推薦值 | 說明 |
|--------|--------|------|
| Optimization | O1 或 O2 | ❌ 避免 O3（基於 Zero-DCE 教訓） |
| Runtime | Neural-ART | NPU 加速 |
| Memory Pools | Auto | 信任工具自動分配 |

---

## 📊 效能指標

### 模型精度

| 指標 | 訓練模型 | 量化模型 | 說明 |
|------|----------|----------|------|
| **MAE** | 4.65 BPM | ~5.1 BPM | 平均絕對誤差 |
| **RMSE** | 7.23 BPM | ~8.0 BPM | 均方根誤差 |
| **MAPE** | 6.82% | ~7.5% | 平均百分比誤差 |
| **R²** | 0.87 | ~0.85 | 決定係數 |

### 模型大小

| 模型 | 格式 | 大小 | 壓縮率 |
|------|------|------|--------|
| Spatial CNN (FP32) | PyTorch | ~80 KB | - |
| Spatial CNN (INT8) | TFLite | ~20 KB | **4x** |
| Temporal Fusion | C 權重 | ~200 KB | - |
| **總計** | - | ~220 KB | - |

### STM32N6 性能（預估）

| 指標 | 數值 | 說明 |
|------|------|------|
| Spatial CNN 推論 | ~20 ms | NPU 加速 |
| Temporal Fusion 推論 | ~5 ms | CPU 執行 |
| 總延遲 | ~500 ms | 包含 8 幀捕獲 |
| 心率更新頻率 | ~2 Hz | 每秒 2 次 |

---

## 📚 技術文檔

### 主要文檔

- [`CLAUDE.md`](CLAUDE.md) - 專案技術概述、技術限制、開發建議
- [`1_preprocessing/README.md`](1_preprocessing/README.md) - 數據前處理詳細說明
- [`2_training/README.md`](2_training/README.md) - 模型訓練詳細說明
- [`3_model_conversion/README.md`](3_model_conversion/README.md) - 模型轉換詳細說明
- [`4_quantization/README.md`](4_quantization/README.md) - 量化詳細說明
- [`5_validation/README.md`](5_validation/README.md) - 驗證詳細說明
- [`stm32_rppg/README.md`](stm32_rppg/README.md) - STM32N6 部署完整指南
- [`webapp/README.md`](webapp/README.md) - Web 應用使用說明

### STM32 部署文檔

- [`stm32_rppg/docs/deployment_guide.md`](stm32_rppg/docs/deployment_guide.md) - 完整部署流程
- [`stm32_rppg/docs/cubemx_config.md`](stm32_rppg/docs/cubemx_config.md) - STM32CubeMX 配置
- [`stm32_rppg/docs/troubleshooting.md`](stm32_rppg/docs/troubleshooting.md) - 故障排除

### 歷史文檔

- [`docs/archive/ARCHIVE.md`](docs/archive/ARCHIVE.md) - 過時文檔說明
- [`docs/archive/DEVELOPMENT_LOG.md`](docs/archive/DEVELOPMENT_LOG.md) - 開發歷史記錄

---

## ❓ 常見問題

### Q1: 為什麼使用 Pattern A 分離式架構？

**A**: STM32N6 的 X-CUBE-AI 限制最多 4D 張量，原始 6D 輸入無法直接部署。分離式架構：
- Spatial CNN 處理單個 ROI（4D 張量）
- Temporal Fusion 在 CPU 上處理時序（純 C 實現）
- 避免複雜的 6D→4D 轉換

### Q2: 為什麼 Temporal Fusion 不用 TFLite？

**A**: 純 C 實現更靈活：
- 完全控制內存分配
- 更容易調試和優化
- 避免額外的 TFLite Runtime 開銷
- 已驗證與 PyTorch 完全等價（差異 < 1e-5 BPM）

### Q3: STM32 優化級別為什麼要避免 O3？

**A**: 基於 Zero-DCE 失敗經驗：
- O3 (Balanced) 導致激進內存重用
- 緩衝區地址重疊（輸入/輸出相同地址）
- 推論第一次調用就返回 ERROR
- 所有手動修復嘗試均失敗
- **結論**: 使用 O1 (Default) 或 O2 (Time)，信任工具自動分配

### Q4: 量化後精度損失多少？

**A**:
- Spatial CNN INT8 量化: MAE 增加約 **+0.5 BPM**（EXCELLENT）
- 總體精度損失 < 1.5 BPM（可接受範圍）
- 使用分層採樣的校準數據集確保各 HR 範圍都有代表

### Q5: Web 應用對環境有什麼要求？

**A**:
- ✅ 光線充足（自然光或均勻室內光）
- ✅ 臉部正對攝像頭（±15° 可接受）
- ✅ 保持相對靜止（輕微點頭 OK）
- ❌ 避免逆光、暗光、側臉、遮擋
- 建議距離：50-100 cm

---

## 🛠️ 環境需求

### Python 環境

**rPPG 訓練環境**（階段 1-3, 5）:
```bash
pip install -r requirements_rppg_training.txt
```
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy, SciPy, Pandas

**TFLite 導出環境**（階段 4）:
```bash
conda create -n tflite_export python=3.10
conda activate tflite_export
pip install -r requirements_tflite_export.txt
```
- TensorFlow 2.13.1
- PyTorch 2.0+

### STM32 環境

- **MCU**: STM32N6 系列
- **IDE**: STM32CubeIDE
- **工具**: STM32CubeMX + X-CUBE-AI 10.x
- **編譯器**: GCC ARM Embedded

---

## 🔗 相關資源

### 論文與數據集
- **UBFC-rPPG 數據集**: https://sites.google.com/view/ybenezeth/ubfcrppg
- **ME-rPPG**: https://arxiv.org/abs/2504.01774

### STM32 技術資源
- **X-CUBE-AI 官方文檔**: https://www.st.com/en/embedded-software/x-cube-ai.html
- **STM32N6 產品頁**: https://www.st.com/stm32n6
- **Neural-ART Runtime**: https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI

---

## 📄 授權

本專案採用 MIT License 授權。

---

## 👥 貢獻

歡迎提交 Issue 和 Pull Request！

---

**版本**: 2.0 (重構版)
**最後更新**: 2025-01-XX
**維護者**: BCPH357
**GitHub**: https://github.com/BCPH357/stm32n6_rppg
