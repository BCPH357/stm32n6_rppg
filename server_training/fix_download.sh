#!/bin/bash
################################################################################
# UBFC-rPPG 下載修復腳本
# 按推薦順序嘗試三種方法
################################################################################

set +e  # 不要在錯誤時退出，允許嘗試多種方法

cd /home/miat/ChenPinHao/server_training/raw_data/UBFC-rPPG/

echo "========================================"
echo "方案 1: 重新運行 gdown（續傳）"
echo "========================================"
echo ""
echo "檢查已下載的文件..."
SUBJECTS=$(find . -type d -name "subject*" | wc -l)
echo "已找到 $SUBJECTS 個 subjects"
echo ""

# 升級 gdown 到最新版
echo "升級 gdown 到最新版..."
pip install --upgrade gdown --quiet

echo "重新運行 gdown（會自動跳過已下載的文件）..."
gdown --folder https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk --remaining-ok

# 檢查是否成功
SUBJECTS_AFTER=$(find . -type d -name "subject*" | wc -l)
echo ""
echo "現在有 $SUBJECTS_AFTER 個 subjects"

if [ "$SUBJECTS_AFTER" -ge 40 ]; then
    echo "✅ 方案 1 成功！UBFC 數據集下載完成"
    echo ""
    echo "驗證數據完整性..."
    cd /home/miat/ChenPinHao/server_training/
    python validate_data.py --check raw
    exit 0
fi

echo ""
echo "⚠️  方案 1 未完全成功（subjects: $SUBJECTS_AFTER < 40）"
echo ""

################################################################################
echo "========================================"
echo "方案 2: 使用 rclone 從官方 Google Drive 下載"
echo "========================================"
echo ""

# 檢查 rclone 是否已安裝
if ! command -v rclone &> /dev/null; then
    echo "安裝 rclone..."
    sudo apt-get update
    sudo apt-get install rclone -y
fi

echo ""
echo "⚠️  需要配置 rclone（一次性操作）"
echo ""
echo "請按照以下步驟操作："
echo ""
echo "1. 運行命令：rclone config"
echo "2. 選擇 n (New remote)"
echo "3. name> 輸入：gdrive"
echo "4. Storage> 選擇 drive (Google Drive，通常是編號 15 或 16)"
echo "5. client_id> 直接按 Enter（留空）"
echo "6. client_secret> 直接按 Enter（留空）"
echo "7. scope> 選擇 1 (Full access)"
echo "8. service_account_file> 直接按 Enter（留空）"
echo "9. Edit advanced config> 選擇 n"
echo "10. Use web browser> 選擇 n"
echo "11. 複製顯示的 URL，在本地瀏覽器打開"
echo "12. 登入 Google 帳號並授權"
echo "13. 複製驗證碼，貼回終端"
echo "14. Configure as Shared Drive> 選擇 n"
echo "15. Keep this remote> 選擇 y"
echo "16. Quit config> 選擇 q"
echo ""
echo "配置完成後，運行以下命令："
echo ""
echo "cd /home/miat/ChenPinHao/server_training/raw_data/"
echo "rclone copy --drive-shared-with-me 'gdrive:UBFC-rPPG' ./UBFC-rPPG/ --progress --transfers 16 --drive-acknowledge-abuse"
echo ""
echo "或者如果你已上傳到自己的 Google Drive："
echo "rclone copy gdrive:UBFC-rPPG ./UBFC-rPPG/ --progress --transfers 16"
echo ""

################################################################################
echo "========================================"
echo "方案 3: 手動指導"
echo "========================================"
echo ""
echo "如果以上方法都失敗，請："
echo ""
echo "A. 在本地電腦上傳 UBFC-rPPG 到你的 Google Drive："
echo "   1. 打開 https://drive.google.com"
echo "   2. 拖放本地的 UBFC-rPPG 文件夾（5-6 GB，需 30-60 分鐘）"
echo "   3. 上傳完成後，在服務器上運行："
echo "      rclone copy gdrive:UBFC-rPPG /home/miat/ChenPinHao/server_training/raw_data/UBFC-rPPG/ --progress --transfers 16"
echo ""
echo "B. 或者使用 scp 上傳（慢但可靠）："
echo "   在本地 Windows 執行："
echo "   scp -r <本地UBFC路徑> miat@140.115.53.67:/home/miat/ChenPinHao/server_training/raw_data/"
echo ""
echo "C. 或者聯繫我，我們可以嘗試其他方法"
echo ""
