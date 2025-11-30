#!/bin/bash
# STM32N6 設置和測試腳本
# 在服務器上執行: bash setup_and_test.sh

set -e

echo "========================================================================"
echo "STM32N6 ONNX 修復工具 - 設置和測試"
echo "========================================================================"

# 檢查 conda 環境
if ! command -v conda &> /dev/null; then
    echo "⚠️  conda 未找到，嘗試使用系統 Python"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "✅ conda 已找到"
    # 激活環境
    source activate rppg_training 2>/dev/null || conda activate rppg_training
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

# 顯示 Python 版本
echo ""
echo "Python 信息:"
$PYTHON_CMD --version
which $PYTHON_CMD

# 安裝依賴
echo ""
echo "========================================================================"
echo "安裝依賴"
echo "========================================================================"

echo "[1/3] 檢查 onnx..."
$PYTHON_CMD -c "import onnx; print(f'  ✅ onnx {onnx.__version__}')" || {
    echo "  ❌ onnx 未安裝"
    $PIP_CMD install onnx
}

echo "[2/3] 檢查 onnx-graphsurgeon..."
$PYTHON_CMD -c "import onnx_graphsurgeon as gs; print(f'  ✅ onnx-graphsurgeon installed')" || {
    echo "  ❌ onnx-graphsurgeon 未安裝，正在安裝..."
    $PIP_CMD install onnx-graphsurgeon
}

echo "[3/3] 檢查 numpy..."
$PYTHON_CMD -c "import numpy as np; print(f'  ✅ numpy {np.__version__}')" || {
    echo "  ❌ numpy 未安裝"
    $PIP_CMD install numpy
}

echo ""
echo "✅ 所有依賴已安裝"

# 檢查模型文件
echo ""
echo "========================================================================"
echo "檢查模型文件"
echo "========================================================================"

if [ -f "checkpoints/best_model.pth" ]; then
    echo "✅ 訓練模型存在: checkpoints/best_model.pth"
    ls -lh checkpoints/best_model.pth
else
    echo "❌ 訓練模型不存在: checkpoints/best_model.pth"
    echo "   請先訓練模型或下載已訓練的模型"
    exit 1
fi

if [ -f "models/rppg_4d_fp32.onnx" ]; then
    echo "✅ 現有 ONNX 模型: models/rppg_4d_fp32.onnx"
    ls -lh models/rppg_4d_fp32.onnx
else
    echo "⚠️  現有 ONNX 模型不存在，將使用 Clean Export 創建"
fi

# 測試腳本
echo ""
echo "========================================================================"
echo "測試診斷腳本"
echo "========================================================================"

if [ -f "models/rppg_4d_fp32.onnx" ]; then
    echo "測試診斷現有 ONNX..."
    $PYTHON_CMD diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32.onnx || true
fi

echo ""
echo "========================================================================"
echo "設置完成！"
echo "========================================================================"
echo ""
echo "下一步執行選項:"
echo ""
echo "選項 1: Clean Export (推薦)"
echo "  python export_onnx_stm32_clean.py"
echo ""
echo "選項 2: 修復現有 ONNX"
echo "  python fix_onnx_for_stm32.py --input models/rppg_4d_fp32.onnx --output models/rppg_4d_fp32_fixed.onnx"
echo ""
echo "選項 3: 自動化流程"
echo "  bash deploy_stm32n6_complete.sh"
echo ""
echo "========================================================================"
