#!/bin/bash

# rPPG Training Script
# 一鍵啟動訓練

echo "=============================="
echo "Starting rPPG Training"
echo "=============================="

# 啟動環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rppg_training

# 檢查 GPU
echo "Checking GPU..."
nvidia-smi

# 检查数据
echo ""
echo "Checking data..."
if [ ! -f "data/ubfc_processed.pt" ]; then
    echo "❌ Error: Preprocessed UBFC data not found"
    echo "Please run preprocessing first:"
    echo "  python preprocess_data.py --dataset ubfc --raw_data raw_data --output data"
    exit 1
fi

echo "✅ Found preprocessed data: data/ubfc_processed.pt"

# 創建輸出目錄
mkdir -p checkpoints
mkdir -p logs

# 開始訓練
echo ""
echo "Starting training..."
python train.py --config config.yaml 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo "Check results in checkpoints/ and logs/"
