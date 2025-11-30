#!/bin/bash
# STM32N6 完整部署流程
# 執行所有必要的步驟從訓練模型到 STM32-ready ONNX

set -e  # 任何錯誤立即退出

echo "========================================================================"
echo "STM32N6 完整部署流程"
echo "========================================================================"
echo ""

# 檢查環境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# 檢查必要文件
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "❌ Model checkpoint not found: checkpoints/best_model.pth"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# ============================================================================
# 選項 A: 從頭導出 Clean ONNX (推薦)
# ============================================================================

read -p "選擇導出方式: [1] Clean Export (推薦) [2] Fix Existing ONNX [3] Both: " choice

if [ "$choice" == "1" ] || [ "$choice" == "3" ]; then
    echo ""
    echo "========================================================================"
    echo "[Option A] Clean ONNX Export"
    echo "========================================================================"

    echo "[Step A1] Exporting STM32-clean ONNX..."
    python export_onnx_stm32_clean.py
    echo "✅ Clean export complete"

    echo ""
    echo "[Step A2] Diagnosing clean ONNX..."
    python diagnose_onnx_stm32.py --onnx models/rppg_stm32_clean_fp32.onnx

    CLEAN_ONNX="models/rppg_stm32_clean_fp32.onnx"

    # 如果診斷仍有問題，應用 graph surgery
    if [ $? -ne 0 ]; then
        echo ""
        echo "[Step A3] Clean export still has issues, applying graph surgery..."
        python fix_onnx_for_stm32.py \
            --input models/rppg_stm32_clean_fp32.onnx \
            --output models/rppg_stm32_clean_fixed.onnx

        CLEAN_ONNX="models/rppg_stm32_clean_fixed.onnx"
    fi

    FINAL_FP32=$CLEAN_ONNX
fi

# ============================================================================
# 選項 B: 修復現有 ONNX
# ============================================================================

if [ "$choice" == "2" ] || [ "$choice" == "3" ]; then
    echo ""
    echo "========================================================================"
    echo "[Option B] Fix Existing ONNX"
    echo "========================================================================"

    # 檢查現有 ONNX
    if [ -f "models/rppg_4d_fp32.onnx" ]; then
        echo "[Step B1] Diagnosing existing ONNX..."
        python diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32.onnx || true

        echo ""
        echo "[Step B2] Applying graph surgery..."
        python fix_onnx_for_stm32.py \
            --input models/rppg_4d_fp32.onnx \
            --output models/rppg_4d_fp32_fixed.onnx

        echo ""
        echo "[Step B3] Validating fixed ONNX..."
        python diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32_fixed.onnx

        FIXED_ONNX="models/rppg_4d_fp32_fixed.onnx"
    else
        echo "⚠️  No existing ONNX found at models/rppg_4d_fp32.onnx"
        echo "   Please use Option A (Clean Export) instead"
    fi

    # 如果選項 B，使用 fixed ONNX
    if [ "$choice" == "2" ]; then
        FINAL_FP32=$FIXED_ONNX
    fi
fi

# ============================================================================
# 量化為 INT8
# ============================================================================

if [ -n "$FINAL_FP32" ]; then
    echo ""
    echo "========================================================================"
    echo "INT8 Quantization"
    echo "========================================================================"

    echo "[Step Q1] Quantizing $FINAL_FP32 to INT8..."

    # 更新量化腳本中的輸入路徑
    QUANT_SCRIPT="quantize_4d_model_v2.py"

    # 執行量化
    python $QUANT_SCRIPT

    echo ""
    echo "[Step Q2] Evaluating quantized model..."
    python evaluate_quantized_model.py

    echo "✅ Quantization complete"
fi

# ============================================================================
# 最終檢查
# ============================================================================

echo ""
echo "========================================================================"
echo "Final Validation"
echo "========================================================================"

INT8_ONNX="models/rppg_4d_int8_qdq.onnx"

if [ -f "$INT8_ONNX" ]; then
    echo "[Final Check] Diagnosing INT8 ONNX..."
    python diagnose_onnx_stm32.py --onnx $INT8_ONNX

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================================================"
        echo "✅ DEPLOYMENT READY!"
        echo "========================================================================"
        echo ""
        echo "模型文件:"
        echo "  - FP32: $FINAL_FP32"
        echo "  - INT8: $INT8_ONNX"
        echo ""
        echo "檔案大小:"
        ls -lh $FINAL_FP32 $INT8_ONNX
        echo ""
        echo "下一步:"
        echo "  1. 下載模型到本地:"
        echo "     scp user@server:$(pwd)/$INT8_ONNX ."
        echo ""
        echo "  2. 上傳到 STM32 Edge AI Developer Cloud"
        echo "     https://stedgeai-dc.st.com/"
        echo ""
        echo "  3. 或使用 stedgeai CLI:"
        echo "     stedgeai analyze --model $INT8_ONNX --target stm32n6"
        echo ""
        echo "========================================================================"
    else
        echo ""
        echo "❌ INT8 model still has violations"
        echo "   Please review the diagnostic report above"
    fi
else
    echo "⚠️  INT8 model not found: $INT8_ONNX"
fi

echo ""
echo "========================================================================"
echo "Deployment script complete"
echo "========================================================================"
