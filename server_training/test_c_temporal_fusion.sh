#!/bin/bash

###############################################################################
# 服務器端測試 Temporal Fusion C 代碼
#
# 執行方法:
#   chmod +x test_c_temporal_fusion.sh
#   ./test_c_temporal_fusion.sh
###############################################################################

echo "======================================================================"
echo "Temporal Fusion C 代碼測試 (服務器端)"
echo "======================================================================"

# Step 1: 導出權重
echo ""
echo "[Step 1] 導出 PyTorch 權重為 C 陣列..."
python export_temporal_fusion_weights.py

if [ $? -ne 0 ]; then
    echo "[ERROR] 權重導出失敗"
    exit 1
fi

# Step 2: 複製 C 文件到當前目錄
echo ""
echo "[Step 2] 準備 C 源文件..."

# 檢查文件是否存在（可能需要從本地上傳）
if [ ! -f "temporal_fusion.h" ]; then
    echo "[ERROR] 找不到 temporal_fusion.h"
    echo "請先上傳 C 文件到服務器："
    echo "  scp D:\\MIAT\\rppg\\stm32_deployment\\temporal_fusion.h miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/"
    echo "  scp D:\\MIAT\\rppg\\stm32_deployment\\temporal_fusion.c miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/"
    echo "  scp D:\\MIAT\\rppg\\stm32_deployment\\test_temporal_fusion.c miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/"
    exit 1
fi

# Step 3: 編譯
echo ""
echo "[Step 3] 編譯 C 代碼..."
gcc -o test_temporal_fusion \
    test_temporal_fusion.c \
    temporal_fusion.c \
    temporal_fusion_weights_exported.c \
    -lm -O2 -Wall

if [ $? -ne 0 ]; then
    echo "[ERROR] 編譯失敗"
    exit 1
fi

echo "[OK] 編譯成功"
echo "  可執行文件: test_temporal_fusion"
echo "  文件大小: $(ls -lh test_temporal_fusion | awk '{print $5}')"

# Step 4: 運行測試
echo ""
echo "[Step 4] 運行測試..."
echo "======================================================================"
./test_temporal_fusion

if [ $? -ne 0 ]; then
    echo "[ERROR] 測試執行失敗"
    exit 1
fi

# Step 5: 總結
echo ""
echo "======================================================================"
echo "[SUCCESS] 測試完成"
echo "======================================================================"
echo ""
echo "輸出文件:"
echo "  - temporal_fusion_weights_exported.c (C 權重文件)"
echo "  - test_temporal_fusion (可執行文件)"
echo ""
echo "下一步:"
echo "  1. 下載權重文件到本地："
echo "     scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/temporal_fusion_weights_exported.c D:\\MIAT\\rppg\\stm32_deployment\\"
echo "  2. 下載 spatial_cnn_int8.tflite："
echo "     scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/spatial_cnn_int8.tflite D:\\MIAT\\rppg\\webapp\\models\\"
echo "======================================================================"
