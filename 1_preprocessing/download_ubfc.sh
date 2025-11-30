#!/bin/bash
################################################################################
# UBFC-rPPG Dataset Downloader
# 使用 gdown 从 Google Drive 自动下载 UBFC-rPPG 数据集
################################################################################

set -e  # Exit on error

echo "========================================"
echo "UBFC-rPPG Dataset Downloader"
echo "========================================"
echo ""

# Create directory
echo "Creating directory: raw_data/UBFC-rPPG"
mkdir -p raw_data/UBFC-rPPG
cd raw_data/UBFC-rPPG

# Download using gdown
echo ""
echo "Downloading UBFC-rPPG dataset from Google Drive..."
echo "URL: https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk"
echo ""

if ! command -v gdown &> /dev/null; then
    echo "❌ Error: gdown is not installed!"
    echo ""
    echo "Please install it using:"
    echo "  pip install gdown"
    echo ""
    exit 1
fi

# Download the folder
echo "Starting download (this may take 30-60 minutes depending on network speed)..."
gdown --folder https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk --remaining-ok

# Verify download
echo ""
echo "========================================"
echo "Verifying download..."
echo "========================================"

SUBJECTS=$(find . -type d -name "subject*" | wc -l)
echo "Found $SUBJECTS subjects"

if [ "$SUBJECTS" -ge 40 ]; then
    echo "✅ UBFC-rPPG dataset downloaded successfully!"
    echo ""
    echo "Dataset location: $(pwd)"
    echo "Total subjects: $SUBJECTS"

    # Show sample structure
    echo ""
    echo "Sample directory structure:"
    FIRST_SUBJECT=$(find . -type d -name "subject*" | head -n 1)
    if [ -n "$FIRST_SUBJECT" ]; then
        ls -lh "$FIRST_SUBJECT" | head -n 10
    fi
else
    echo "⚠️  Warning: Expected 42-43 subjects, but found $SUBJECTS"
    echo ""
    echo "Possible issues:"
    echo "  - Incomplete download (check network connection)"
    echo "  - Google Drive quota limit reached"
    echo "  - Incorrect folder structure"
    echo ""
    echo "Please try again or use manual download method (see below)."
fi

echo ""
echo "========================================"
echo "Alternative: Manual Download Method"
echo "========================================"
echo ""
echo "If gdown fails, you can download manually:"
echo ""
echo "1. Visit: https://sites.google.com/view/ybenezeth/ubfcrppg"
echo "2. Click on 'Download Dataset' link"
echo "3. Download the ZIP/TAR file to your local machine"
echo "4. Upload to server using scp or rsync:"
echo "   scp UBFC-rPPG.tar.gz username@server:/path/to/rppg_training/raw_data/"
echo "5. Extract:"
echo "   cd raw_data"
echo "   tar -xzf UBFC-rPPG.tar.gz"
echo ""
echo "Expected structure:"
echo "  raw_data/UBFC-rPPG/"
echo "    ├── subject1/"
echo "    │   ├── vid.avi"
echo "    │   └── ground_truth.txt"
echo "    ├── subject2/"
echo "    └── ..."
echo ""
