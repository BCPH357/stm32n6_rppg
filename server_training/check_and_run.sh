#!/bin/bash
################################################################################
# 檢查項目目錄結構並運行完整流程
# 用於確認下載完成後的下一步操作
################################################################################

set -e

echo "========================================"
echo "步驟 1: 檢查項目目錄結構"
echo "========================================"
echo ""

cd /home/miat/ChenPinHao/server_training/

echo "當前目錄: $(pwd)"
echo ""

# 檢查必要的目錄
echo "檢查目錄結構..."
echo ""

DIRS_OK=true

# 檢查 raw_data
if [ -d "raw_data/UBFC-rPPG" ]; then
    SUBJECTS=$(find raw_data/UBFC-rPPG/ -type d -name "subject*" | wc -l)
    echo "✅ raw_data/UBFC-rPPG/ 存在"
    echo "   └─ 找到 $SUBJECTS 個 subjects"

    if [ "$SUBJECTS" -ge 40 ]; then
        echo "   └─ ✅ Subject 數量正常 (>= 40)"
    else
        echo "   └─ ⚠️  Subject 數量不足 (< 40)"
        DIRS_OK=false
    fi
else
    echo "❌ raw_data/UBFC-rPPG/ 不存在"
    DIRS_OK=false
fi

# 檢查 data 目錄（用於存放預處理後的數據）
if [ -d "data" ]; then
    echo "✅ data/ 目錄存在"
else
    echo "⚠️  data/ 目錄不存在，將自動創建"
    mkdir -p data
fi

# 檢查 checkpoints 目錄（用於存放訓練模型）
if [ -d "checkpoints" ]; then
    echo "✅ checkpoints/ 目錄存在"
else
    echo "⚠️  checkpoints/ 目錄不存在，將自動創建"
    mkdir -p checkpoints
fi

# 檢查 logs 目錄（用於存放訓練日誌）
if [ -d "logs" ]; then
    echo "✅ logs/ 目錄存在"
else
    echo "⚠️  logs/ 目錄不存在，將自動創建"
    mkdir -p logs
fi

echo ""

# 檢查必要的 Python 腳本
echo "檢查 Python 腳本..."
echo ""

FILES_OK=true

if [ -f "preprocess_data.py" ]; then
    echo "✅ preprocess_data.py 存在"
else
    echo "❌ preprocess_data.py 不存在"
    FILES_OK=false
fi

if [ -f "train.py" ]; then
    echo "✅ train.py 存在"
else
    echo "❌ train.py 不存在"
    FILES_OK=false
fi

if [ -f "model.py" ]; then
    echo "✅ model.py 存在"
else
    echo "❌ model.py 不存在"
    FILES_OK=false
fi

if [ -f "config.yaml" ]; then
    echo "✅ config.yaml 存在"
else
    echo "❌ config.yaml 不存在"
    FILES_OK=false
fi

if [ -f "validate_data.py" ]; then
    echo "✅ validate_data.py 存在"
else
    echo "❌ validate_data.py 不存在"
    FILES_OK=false
fi

echo ""

# 顯示完整目錄結構
echo "========================================"
echo "完整目錄結構預覽"
echo "========================================"
echo ""
tree -L 2 -I '__pycache__|*.pyc|.git' || ls -lhR --max-depth=2

echo ""
echo "========================================"
echo "步驟 2: 驗證原始數據"
echo "========================================"
echo ""

if [ "$DIRS_OK" = true ] && [ "$FILES_OK" = true ]; then
    echo "運行數據驗證..."
    python validate_data.py --check raw

    VALIDATE_STATUS=$?

    if [ $VALIDATE_STATUS -eq 0 ]; then
        echo ""
        echo "✅ 原始數據驗證通過！"
    else
        echo ""
        echo "❌ 原始數據驗證失敗！請檢查上面的錯誤信息"
        exit 1
    fi
else
    echo "⚠️  跳過數據驗證（目錄或文件缺失）"
    exit 1
fi

echo ""
echo "========================================"
echo "步驟 3: 預處理數據"
echo "========================================"
echo ""

# 檢查是否已經有預處理好的數據
if [ -f "data/ubfc_processed.pt" ]; then
    echo "⚠️  發現已存在的預處理數據: data/ubfc_processed.pt"
    echo ""
    echo "選項:"
    echo "  1) 跳過預處理，直接訓練"
    echo "  2) 刪除舊數據，重新預處理"
    echo "  3) 退出，手動處理"
    echo ""
    read -p "請選擇 [1-3]: " choice

    case $choice in
        1)
            echo "跳過預處理..."
            ;;
        2)
            echo "刪除舊數據並重新預處理..."
            rm -f data/ubfc_processed.pt
            echo "開始預處理（預計 2-3 小時）..."
            python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
            ;;
        3)
            echo "退出腳本"
            exit 0
            ;;
        *)
            echo "無效選擇，退出"
            exit 1
            ;;
    esac
else
    echo "開始預處理 UBFC 數據集..."
    echo "預計時間: 2-3 小時（服務器 CPU）"
    echo ""
    echo "這會將原始視頻處理為:"
    echo "  - 提取 3 個 ROI (前額、左臉頰、右臉頰)"
    echo "  - 每個 ROI 調整為 36×36×3"
    echo "  - 創建時間窗口（8 幀）"
    echo "  - 保存為 data/ubfc_processed.pt"
    echo ""
    read -p "確認開始預處理? [y/n]: " confirm

    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
    else
        echo "取消預處理，退出"
        exit 0
    fi
fi

echo ""
echo "========================================"
echo "步驟 4: 驗證預處理數據"
echo "========================================"
echo ""

if [ -f "data/ubfc_processed.pt" ]; then
    echo "驗證預處理數據格式..."
    python validate_data.py --check processed

    VALIDATE_STATUS=$?

    if [ $VALIDATE_STATUS -eq 0 ]; then
        echo ""
        echo "✅ 預處理數據驗證通過！"
    else
        echo ""
        echo "❌ 預處理數據驗證失敗！"
        exit 1
    fi
else
    echo "❌ 找不到預處理數據: data/ubfc_processed.pt"
    exit 1
fi

echo ""
echo "========================================"
echo "步驟 5: 開始訓練"
echo "========================================"
echo ""

echo "訓練配置:"
echo "  - 數據集: UBFC-rPPG"
echo "  - 模型: Multi-ROI rPPG (3 ROIs)"
echo "  - 預計時間: 1.5-2 小時 (A6000 GPU)"
echo ""

# 顯示 config.yaml 內容
echo "當前配置 (config.yaml):"
echo "────────────────────────────────────────"
cat config.yaml
echo "────────────────────────────────────────"
echo ""

read -p "確認開始訓練? [y/n]: " confirm_train

if [ "$confirm_train" = "y" ] || [ "$confirm_train" = "Y" ]; then
    echo ""
    echo "開始訓練..."
    echo "日誌將保存到 logs/ 目錄"
    echo "模型檢查點將保存到 checkpoints/ 目錄"
    echo ""

    # 運行訓練（使用 nohup 以防 SSH 斷線）
    nohup python train.py --config config.yaml > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    TRAIN_PID=$!
    echo "訓練已在後台啟動 (PID: $TRAIN_PID)"
    echo ""
    echo "監控訓練進度:"
    echo "  tail -f logs/training_*.log"
    echo ""
    echo "或者使用 TensorBoard:"
    echo "  tensorboard --logdir=logs/ --port=6006"
    echo "  然後在瀏覽器訪問: http://140.115.53.67:6006"
    echo ""
    echo "檢查訓練是否仍在運行:"
    echo "  ps aux | grep train.py"
    echo ""
    echo "停止訓練:"
    echo "  kill $TRAIN_PID"
else
    echo ""
    echo "取消訓練，退出"
    echo ""
    echo "如需稍後訓練，運行:"
    echo "  bash run_training.sh"
    exit 0
fi

echo ""
echo "========================================"
echo "✅ 所有步驟完成！"
echo "========================================"
echo ""
echo "後續操作:"
echo "  1. 監控訓練: tail -f logs/training_*.log"
echo "  2. 訓練完成後，最佳模型保存在: checkpoints/best_model.pth"
echo "  3. 評估模型: python evaluate.py --checkpoint checkpoints/best_model.pth"
echo ""
