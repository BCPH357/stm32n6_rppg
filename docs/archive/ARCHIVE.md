# 📦 Archive - 過時檔案說明

本目錄包含專案重構前的過時檔案和文檔，保留作為歷史記錄參考。

## 📁 目錄內容

### 開發歷史文檔
- **DEVELOPMENT_LOG.md** - 完整開發歷史記錄（2025-01-14 至重構前）
- **DEPLOY_4D_TO_STM32.md** - 舊版 4D 模型部署指南
- **STM32N6_COMPLETE_GUIDE.md** - 舊版完整部署指南
- **FIX_DYNAMIC_BATCH_GUIDE.md** - 動態 Batch 修復指南
- **MODEL_CONSTRAINT.md** - 模型限制說明

### LLM Prompt 檔案
- **rppg_n6_llm_prompt.md** - rPPG STM32N6 部署 Prompt
- **stm32n6_fix_prompt.md** - STM32N6 修復 Prompt
- **stm32n6Modelrevised.md** - 模型修訂 Prompt

### 過時程式碼
- **convert_6d_to_4d_onnx.py** - 舊版 6D→4D ONNX 轉換（已被 convert_to_4d_for_stm32.py 取代）
- **model_4d_stm32.py** - 舊版 4D 模型定義
- **test.py** - 測試用腳本

### 過時批次檔
- **fix_and_upload.bat** - 舊版上傳腳本
- **upload_evaluate.bat** - 舊版評估上傳腳本

### 舊版量化目錄
- **quantization_old/** - 舊版量化工具（功能重複，已整合到 4_quantization/）

## ⚠️ 注意事項

1. **請勿使用這些檔案** - 它們已被新版本取代或不再適用於當前架構
2. **僅供參考** - 如需查看歷史開發過程或特定實現細節時可參考
3. **不保證可用性** - 這些檔案可能依賴舊版環境或已不存在的目錄結構

## 📌 對應的新版檔案

| 舊檔案 | 新位置 | 說明 |
|--------|--------|------|
| `convert_6d_to_4d_onnx.py` | `3_model_conversion/convert_to_4d_for_stm32.py` | 模型轉換 |
| `quantization/` | `4_quantization/` | 量化工具 |
| `DEPLOY_4D_TO_STM32.md` | `stm32_rppg/docs/deployment_guide.md` | 部署指南 |
| `MODEL_CONSTRAINT.md` | `CLAUDE.md` (技術限制章節) | 模型限制說明 |

## 🗂️ 保留原因

這些檔案保留的原因：
1. 記錄專案演進歷史
2. 保存特定問題的解決方案參考
3. 記錄失敗嘗試的經驗教訓（避免重蹈覆轍）
4. 提供 LLM 輔助開發的 Prompt 範例

---

**重構日期**: 2025-01-XX
**原專案結構**: 參見 DEVELOPMENT_LOG.md
