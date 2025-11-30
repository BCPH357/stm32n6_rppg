#!/bin/bash
# ====================================================================
# rPPG Multi-ROI Model INT8 Quantization - Complete Workflow (Linux)
# ====================================================================
#
# 此腳本將執行完整的量化流程：
# 1. 準備校準數據集
# 2. 導出 FP32 ONNX 模型
# 3. 執行 INT8 量化
# 4. 驗證量化精度
#
# 使用環境: rppg_training conda environment (服務器)
#
# ====================================================================

set -e  # 遇到錯誤立即退出

echo ""
echo "===================================================================="
echo "rPPG Multi-ROI Model INT8 Quantization Workflow"
echo "===================================================================="
echo ""

# 檢查是否在正確目錄
if [ ! -f "quantize_utils.py" ]; then
    echo "ERROR: Please run this script from the quantization directory!"
    echo "Current directory: $(pwd)"
    exit 1
fi

# 激活 conda 環境
echo "[Step 0/4] Activating conda environment: rppg_training"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rppg_training

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'rppg_training'"
    echo "Please ensure the environment exists."
    exit 1
fi
echo ""

# Step 1: 準備校準數據
echo "===================================================================="
echo "[Step 1/4] Preparing Calibration Dataset"
echo "===================================================================="
python quantize_utils.py --data ../data/ubfc_processed.pt \
                         --output calibration_data.pt \
                         --num_samples 200

if [ $? -ne 0 ]; then
    echo "ERROR: Calibration data preparation failed!"
    exit 1
fi
echo ""

# Step 2: 導出 FP32 ONNX
echo "===================================================================="
echo "[Step 2/4] Exporting FP32 ONNX Model"
echo "===================================================================="
python export_onnx.py --checkpoint ../checkpoints/best_model.pth \
                      --output models/rppg_fp32.onnx \
                      --opset 13

if [ $? -ne 0 ]; then
    echo "ERROR: FP32 ONNX export failed!"
    exit 1
fi
echo ""

# Step 3: INT8 量化
echo "===================================================================="
echo "[Step 3/4] Quantizing to INT8"
echo "===================================================================="
python quantize_onnx.py --input models/rppg_fp32.onnx \
                        --output models/rppg_int8_qdq.onnx \
                        --calibration calibration_data.pt \
                        --per_channel

if [ $? -ne 0 ]; then
    echo "ERROR: INT8 quantization failed!"
    exit 1
fi
echo ""

# Step 4: 驗證精度
echo "===================================================================="
echo "[Step 4/4] Verifying Quantization Accuracy"
echo "===================================================================="
python verify_quantization.py --fp32 models/rppg_fp32.onnx \
                               --int8 models/rppg_int8_qdq.onnx \
                               --data ../data/ubfc_processed.pt \
                               --num_samples 500

VERIFY_RESULT=$?
echo ""

# 最終報告
echo "===================================================================="
echo "Quantization Workflow Completed!"
echo "===================================================================="
echo ""

if [ $VERIFY_RESULT -eq 0 ]; then
    echo "✅ Status: SUCCESS - Quantization acceptable"
    echo ""
    echo "Next steps:"
    echo "1. Download INT8 model: models/rppg_int8_qdq.onnx"
    echo "2. Use X-CUBE-AI to convert for STM32N6"
    echo "3. Refer to: ../stm32n6_deployment/deployment_guide.md"
elif [ $VERIFY_RESULT -eq 2 ]; then
    echo "⚠️  Status: WARNING - Quantization degradation significant"
    echo ""
    echo "Suggestions:"
    echo "1. Increase calibration samples: python quantize_utils.py --num_samples 500"
    echo "2. Consider Quantization-Aware Training (QAT)"
    echo "3. Check calibration data distribution"
else
    echo "❌ Status: ERROR - Quantization failed"
    echo "Please check error messages above"
fi

echo ""
echo "===================================================================="
