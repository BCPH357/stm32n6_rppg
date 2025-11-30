#!/bin/bash
################################################################################
# 重啟訓練腳本
# 1. 停止當前訓練進程
# 2. 使用修復版 train_fixed.py 重新開始
################################################################################

echo "======================================"
echo "Restarting Training with Fixed Script"
echo "======================================"
echo ""

cd /home/miat/ChenPinHao/server_training/

# 啟動環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rppg_training

# 檢查是否有正在運行的訓練
if [ -f "training.pid" ]; then
    OLD_PID=$(cat training.pid)
    echo "Found existing training PID: $OLD_PID"

    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Stopping old training process..."
        kill $OLD_PID
        sleep 2

        # 強制終止（如果還在運行）
        if ps -p $OLD_PID > /dev/null 2>&1; then
            echo "Force killing..."
            kill -9 $OLD_PID
        fi

        echo "[OK] Old training stopped"
    else
        echo "[INFO] Old PID not running"
    fi

    rm -f training.pid
fi

# 檢查 GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 檢查數據
echo ""
echo "Checking data..."
if [ ! -f "data/ubfc_processed.pt" ]; then
    echo "[ERROR] Preprocessed data not found"
    exit 1
fi

echo "[OK] Found preprocessed data: data/ubfc_processed.pt"
DATA_SIZE=$(du -h data/ubfc_processed.pt | cut -f1)
echo "   Size: $DATA_SIZE"

# 創建輸出目錄
mkdir -p checkpoints_fixed logs

# 備份舊的 checkpoints（如果存在）
if [ -d "checkpoints" ]; then
    echo ""
    echo "Backing up old checkpoints..."
    BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
    mv checkpoints "$BACKUP_DIR"
    echo "[OK] Old checkpoints moved to $BACKUP_DIR"
fi

# 使用新目錄
mv checkpoints_fixed checkpoints

# 生成日誌文件名
LOG_FILE="logs/training_fixed_$(date +%Y%m%d_%H%M%S).log"

# 開始訓練（使用修復版腳本）
echo ""
echo "Starting training with fixed script..."
echo "Log file: $LOG_FILE"
echo ""

nohup python train_fixed.py --config config.yaml > "$LOG_FILE" 2>&1 &

# 保存 PID
NEW_PID=$!
echo $NEW_PID > training.pid

echo ""
echo "======================================"
echo "Training Restarted!"
echo "======================================"
echo "   PID: $NEW_PID"
echo ""
echo "What's fixed:"
echo "  1. Labels are now normalized (Z-score)"
echo "  2. MAPE calculation fixed (no division by near-zero)"
echo "  3. Added raw BVP metrics for interpretability"
echo ""
echo "Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "   ps -p $NEW_PID"
echo ""
echo "Stop training:"
echo "   kill $NEW_PID"
echo ""
echo "GPU usage:"
echo "   watch -n 2 nvidia-smi"
echo ""
