#!/bin/bash

echo "=============================="
echo "Starting rPPG Training (Background)"
echo "=============================="

cd /mnt/data_8T/ChenPinHao/server_training/

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
    echo "Error: Preprocessed UBFC data not found"
    echo "Please run preprocessing first"
    exit 1
fi

echo "Found preprocessed data: data/ubfc_processed.pt"
DATA_SIZE=$(du -h data/ubfc_processed.pt | cut -f1)
echo "   Size: $DATA_SIZE"

# 創建輸出目錄
mkdir -p checkpoints logs

# 生成日誌文件名
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# 開始訓練（後台）
echo ""
echo "Starting training in background..."
echo "Log file: $LOG_FILE"

nohup python train.py --config config.yaml > "$LOG_FILE" 2>&1 &

# 保存 PID
echo $! > training.pid
PID=$(cat training.pid)

echo ""
echo "Training started!"
echo "   PID: $PID"
echo ""
echo "Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "   ps -p $PID"
echo ""
echo "Stop training:"
echo "   kill $PID"
echo ""
echo "GPU usage:"
echo "   watch -n 2 nvidia-smi"
echo ""
